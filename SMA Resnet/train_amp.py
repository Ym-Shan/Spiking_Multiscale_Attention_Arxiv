import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torch.cuda.amp
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader
from collections import OrderedDict
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, model, layer, monitor
import torch.distributed as dist
from timm.data.mixup import Mixup


class batch_norm_2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = nn.BatchNorm3d(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)



class SMA(nn.Module):
    def __init__(self, channel=512, time=16, kernels=[1, 3, 5, 7], C_reduction=16, T_reduction=4, AZO=False, RCR=4, RTR=4):
        super().__init__()
        self.AZO = AZO
        self.RCR = RCR
        self.RTR = RTR
        self.C_attention_out = channel // C_reduction
        self.T_attention_out = time // T_reduction
        self.S_Convs = nn.ModuleList([])
        for i in kernels:
            self.S_Convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', layer.Conv2d(channel, channel, kernel_size=i, padding=i // 2)),
                    ('bn', batch_norm_2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )

        # --------------------------------channel---------------------------------
        self.C_before_attention_conv = layer.Conv2d(channel, self.C_attention_out, 1)
        self.C_after_attention_convs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.C_after_attention_convs.append(layer.Conv2d(self.C_attention_out, channel, 1))

        # --------------------------------time---------------------------------
        self.T_before_attention_conv = nn.Conv3d(time, self.T_attention_out, 1)
        self.T_after_attention_convs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.T_after_attention_convs.append(nn.Conv3d(self.T_attention_out, time, 1))

        # ----------------------------------------------------------------------
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        T, B, C, _, _ = x.size()
        S_Conv_outs = []
        for conv in self.S_Convs:
            S_Conv_outs.append(conv(x))
        multiscale_feature_map = torch.stack(S_Conv_outs, 0)    # [n, T, B, C, h, w]
        feature_map = sum(S_Conv_outs)
        feature_map = feature_map.mean(-1).mean(-1)
        feature_map = feature_map.unsqueeze(-1).unsqueeze(-1)       # [T, B, C, 1, 1]
        # --------------------------------channel---------------------------------
        C_attention_weight = []     # [n]->[T, B, C, 1, 1](inside)
        C_attention_mid_weight = self.C_before_attention_conv(feature_map)
        C_attention_mid_weight = self.relu(C_attention_mid_weight)
        for conv in self.C_after_attention_convs:
            weight = conv(C_attention_mid_weight)
            C_attention_weight.append(weight.view(T, B, C, 1, 1))
        # ---------------------------------
        C_attention_weight_stack = torch.stack(C_attention_weight, 0)   # [n, T, B, C, 1, 1]
        C_attention_weight_stack = self.softmax(C_attention_weight_stack)
        # --------------------------------time---------------------------------
        T_attention_weight = []
        feature_map_before_T_attention = feature_map.permute(1, 0, 2, 3, 4)     # [B, T, C, 1, 1]
        T_after_pool = self.avg_pool(feature_map_before_T_attention)            # [B, T, 1, 1, 1]
        T_attention_mid_weight = self.T_before_attention_conv(T_after_pool)
        T_attention_mid_weight = self.relu(T_attention_mid_weight)
        for conv in self.T_after_attention_convs:
            weight = conv(T_attention_mid_weight)
            T_attention_weight.append(weight)
        # # ---------------------------------
        T_attention_weight_stack = torch.stack(T_attention_weight, 0)       # [n, B, T, 1, 1, 1]
        T_attention_weight_stack = T_attention_weight_stack.permute(0, 2, 1, 3, 4, 5)       # [n, T, B, 1, 1, 1]
        T_attention_weight_stack = self.softmax(T_attention_weight_stack)
        # ----------------------------------------------------------------------
        # T:[n, T, B, 1, 1, 1]                 C:[n, T, B, C, 1, 1]
        # The following section is the AZO regularization method
        # ------------------------------------------------------------------------------------------------------
        if self.AZO == True:
            if train_flag == 1:
                n_C = C // self.RCR           # Take the smallest few T's
                n_T = T // self.RTR           # Take the smallest few C's
            
            
                T_min_indices = []              # [4, n_T]
                C_min_indices = []              # [4, n_C]
            
                for i in T_attention_weight_stack:      # [T, B, 1, 1, 1]
                    j = torch.sum(i, dim=1)             # Accumulate over time to facilitate finding the minimum value
                    j = j.squeeze()                     
            
                    T_min_indices.append(torch.topk(j, k=n_T, largest=False).indices)         # The minimum number of n indexes
            
                j = 0               # Indicate to what scale the processing has reached
                for i in C_attention_weight_stack:      # [T, B, C, 1, 1]
                    c = torch.sum(i, dim=1)             # Accumulate on the batch dimension to facilitate finding the minimum value, [T, C, 1, 1]
                    # Find nC channels at the time step specified in the Tmin_index
                    t = T_min_indices[j]                # [n_T]
                    select_channel = []                 # [n_T]->[B, C, 1, 1]
                    for m in range(len(t)):
                        select_channel.append(i[t[m]])
                    select_channel = torch.stack(select_channel, dim=0)         # [n_T, B, C, 1, 1]
                    k = torch.sum(select_channel, dim=0)                    # Accumulate on the n-T dimension  [B, C, 1, 1]
                    k = torch.sum(k, dim=0)                                 # Accumulate on B    [C, 1, 1]
                    k = k.squeeze()                                         # [C]
            
                    C_min_indices.append(torch.topk(k, k=n_C, largest=False).indices)         # The minimum number of n indexes
            
                    j += 1
                # ------------------------------------------------------------------------------------------------------
            
                attention_module_out = (T_attention_weight_stack * C_attention_weight_stack
                                        * multiscale_feature_map)
            
                after_resplace = attention_module_out.clone()       # [n, T, B, C, h, w]
            
                for i in range(len(T_min_indices)):  # scale
                    non_zero_T_min_indices = torch.stack([j for j in T_min_indices[i] if j != 0])
                    non_zero_T_min_indices = non_zero_T_min_indices[:, None]
                    C_min_indices[i] = C_min_indices[i][None, :]
                    after_resplace[i, non_zero_T_min_indices, :, C_min_indices[i]] = attention_module_out[i,
                                                                                    non_zero_T_min_indices - 1, :,
                                                                                    C_min_indices[i]]
            
                result = after_resplace.sum(0)
            
            else:
                result = (T_attention_weight_stack * C_attention_weight_stack
                                        * multiscale_feature_map).sum(0)
        else:
            result = (T_attention_weight_stack * C_attention_weight_stack
                                        * multiscale_feature_map).sum(0)

        return result


mixup_fn = Mixup(
            mixup_alpha=0.8,          # Determines the proportion of each of the two data samples when mixed. The larger the alpha value, the greater the possible difference between the mixed samples.
            cutmix_alpha=1.0,         # In CutMix, not the entire image is mixed, but parts of the image are replaced. cutmix_alpha controls the size of these parts.
            cutmix_minmax=None,     # Defines the minimum and maximum proportion of the replaced area in CutMix. If None, the size is not restricted and is entirely determined by cutmix_alpha.
            prob=1.0,               # The probability of performing Mixup or CutMix. There is a certain probability of applying data mixing augmentation each time data is loaded.
            switch_prob=0.5,        # When deciding to use data augmentation, this parameter determines whether to use Mixup or CutMix. For example, if switch_prob is 0.5, there is a 50% chance to choose Mixup and a 50% chance to choose CutMix.
            mode="batch",           # Determines the way Mixup or CutMix is applied. 'batch' mode means applying augmentation to the entire batch of data. There are other modes such as 'pair' or 'elem', which apply to each pair or each element in the batch individually.
            num_classes=1000,
        )

def train(epoch, args):
    running_loss = 0
    start = time.time()
    net.train()
    correct = 0.0
    num_sample = 0
    for batch_index, (images, labels) in enumerate(ImageNet_training_loader):
        if args.gpu:
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
        
        # images, labels = mixup_fn(images, labels)

        num_sample += images.size()[0]
        optimizer.zero_grad()
        with autocast():
            outputs = net(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n_iter = (epoch - 1) * len(ImageNet_training_loader) + batch_index + 1
        if batch_index % 10 == 9:
            print(
                'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'
                .format(running_loss / 10,
                        optimizer.param_groups[0]['lr'],
                        epoch=epoch,
                        trained_samples=batch_index * args.b + len(images),
                        total_samples=len(ImageNet_training_loader.dataset)))
            print('training time consumed: {:.2f}s'.format(time.time() -
                                                           start))
            if args.local_rank == 0:
                writer.add_scalar('Train/avg_loss', running_loss / 10, n_iter)
                writer.add_scalar('Train/avg_loss_numpic', running_loss / 10,
                                  n_iter * args.b)
            running_loss = 0
    finish = time.time()
    if args.local_rank == 0:
        writer.add_scalar('Train/acc', correct / num_sample * 100, epoch)
    print("Training accuracy: {:.2f} of epoch {}".format(
        correct / num_sample * 100, epoch))
    print('epoch {} training time consumed: {:.2f}s'.format(
        epoch, finish - start))


@torch.no_grad()
def eval_training(epoch, args):

    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0
    real_batch = 0
    for (images, labels) in ImageNet_test_loader:
        real_batch += images.size()[0]
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.4f}%, Time consumed:{:.2f}s'
        .format(test_loss * args.b / len(ImageNet_test_loader.dataset),
                correct.float() / real_batch * 100, finish - start))

    if args.local_rank == 0:
        # add information to tensorboard
        writer.add_scalar(
            'Test/Average loss',
            test_loss * args.b / len(ImageNet_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy',
                          correct.float() / real_batch * 100, epoch)

    return correct.float() / len(ImageNet_test_loader.dataset)


# for resnet-104
class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes=1000, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1),
                                                       1)
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


if __name__ == '__main__':

    train_flag = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu',
                        action='store_true',
                        default=True,
                        help='use gpu or not')
    parser.add_argument('-b',
                        type=int,
                        default=256,
                        help='batch size for dataloader')
    parser.add_argument('-lr',
                        type=float,
                        default=0.1,
                        help='initial learning rate')
    parser.add_argument('--local-rank',
                        default=-1,
                        type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()
    print(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    SEED = 445
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    net = get_network(args)



    net.conv2_x[0].residual_function.insert(2, SMA(channel=64, C_reduction=4, time=6, T_reduction=1.5, AZO=False, RCR=4, RTR=2))
    net.conv2_x[1].residual_function.insert(2, SMA(channel=64, C_reduction=4, time=6, T_reduction=1.5, AZO=False, RCR=4, RTR=2))

    net.conv3_x[0].residual_function.insert(2, SMA(channel=128, C_reduction=4, time=6, T_reduction=1.5, AZO=False, RCR=4, RTR=2))
    net.conv3_x[1].residual_function.insert(2, SMA(channel=128, C_reduction=4, time=6, T_reduction=1.5, AZO=False, RCR=4, RTR=2))

    net.conv4_x[0].residual_function.insert(2, SMA(channel=256, C_reduction=4, time=6, T_reduction=1.5, AZO=False, RCR=4, RTR=2))
    net.conv4_x[1].residual_function.insert(2, SMA(channel=256, C_reduction=4, time=6, T_reduction=1.5, AZO=False, RCR=4, RTR=2))

    net.conv5_x[0].residual_function.insert(2, SMA(channel=512, C_reduction=4, time=6, T_reduction=1.5, AZO=False, RCR=4, RTR=2))
    net.conv5_x[1].residual_function.insert(2, SMA(channel=512, C_reduction=4, time=6, T_reduction=1.5, AZO=False, RCR=4, RTR=2))

    

    

    
    net.cuda()

    functional.set_step_mode(net, step_mode='m')

    if dist.get_rank() == 0:
        print(net)
    
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank])

    # to load a pretrained model
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    # net.load_state_dict(
    #     torch.load("path", map_location=map_location))

    num_gpus = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # data preprocessing:
    ImageNet_training_loader = get_training_dataloader(
        traindir="./data/train",
        num_workers=70,
        batch_size=args.b // num_gpus,
        shuffle=False,
        sampler=1  # to enable sampler for DDP
    )

    ImageNet_test_loader = get_test_dataloader(valdir="./data/test",
                                               num_workers=70,
                                               batch_size=args.b // num_gpus,
                                               shuffle=False,
                                               sampler=1)
    # learning rate should go with batch size.
    b_lr = args.lr

    loss_function = CrossEntropyLabelSmooth()
    optimizer = optim.SGD([{
        'params': net.parameters(),
        'initial_lr': b_lr
    }],
                          momentum=0.9,
                          lr=b_lr,
                          weight_decay=1e-4)  # SGD MOMENTUM
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.EPOCH, eta_min=0, last_epoch=0)
    iter_per_epoch = len(ImageNet_training_loader)
    LOG_INFO = "ImageNet_ACC"
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net,
                                   str(args.b), str(args.lr), LOG_INFO,
                                   settings.TIME_NOW)

    # use tensorboard
    if args.local_rank == 0:
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(
            log_dir=os.path.join(settings.LOG_DIR, args.net, str(args.b),
                                 str(args.lr), LOG_INFO, settings.TIME_NOW))

    # create checkpoint folder to save model
    if args.local_rank == 0:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path,
                                       '{net}-{epoch}-{type}.pth')
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0

    for epoch in range(1, settings.EPOCH + 1):
        train_flag = 1
        train(epoch, args)
        train_scheduler.step()
        
        train_flag = 0
        acc = eval_training(epoch, args)

        if epoch > (settings.EPOCH -
                    5) and best_acc < acc and args.local_rank == 0:
            torch.save(
                net.state_dict(),
                checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue
        elif epoch >= (settings.EPOCH - 5) and args.local_rank == 0:
            torch.save(
                net.state_dict(),
                checkpoint_path.format(net=args.net,
                                       epoch=epoch,
                                       type='regular'))
            continue
        elif ((not epoch % settings.SAVE_EPOCH) and args.local_rank == 0):
            torch.save(
                net.state_dict(),
                checkpoint_path.format(net=args.net,
                                       epoch=epoch,
                                       type='regular'))
            continue

    if args.local_rank == 0:
        writer.close()
