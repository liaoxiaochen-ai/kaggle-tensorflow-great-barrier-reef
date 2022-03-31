from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Function
from torch.nn.init import calculate_gain
import sys
sys.path.append('/home/yankai/longfeiqi/ddfnet-main/')
from ddf.ddf import *

'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


'''
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# (PixelUnShuffle)
# (ConditionalBatchNorm2d)
# concat (ConcatBlock)
# sum (ShortcutBlock)
# resblock (ResBlock)
# Channel Attention (CA) Layer (CALayer)
# Residual Channel Attention Block (RCABlock)
# Residual Channel Attention Group (RCAGroup)
# Residual Dense Block (ResidualDenseBlock_5C)
# Residual in Residual Dense Block (RRDB)
# --------------------------------------------
'''
# --------------------------------------------
# Weight Normalization
# --------------------------------------------
def wn(x):
    return torch.nn.utils.weight_norm(x)

# --------------------------------------------
# Depthwise Separable Convolution
# --------------------------------------------
def DSC_augment(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True, ratio=2):
    if in_channels==out_channels:
        return ShortcutBlock(nn.Sequential(
            ShortcutBlock(wn(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=bias))),
            wn(nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels*ratio), kernel_size=1, stride=stride, padding=0, bias=bias)),
            conv(mode='L'),
            wn(nn.Conv2d(in_channels=int(in_channels*ratio), out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=bias))
        ))
    else:
        return nn.Sequential(
            ShortcutBlock(wn(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=bias))),
            wn(nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels*ratio), kernel_size=1, stride=stride, padding=0, bias=bias)),
            conv(mode='L'),
            wn(nn.Conv2d(in_channels=int(in_channels*ratio), out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=bias))
        )

def DS_convolution(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        # depthwise conv
        wn(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=bias)),
        # pointwise conv
        wn(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=bias))
    )

def DDF_convolution(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
    if in_channels == out_channels:
        return nn.Sequential(
            DDFPack(in_channels)
        )
    else:
        return nn.Sequential(
        DDFPack(in_channels),
        wn(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=bias))
        )

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        # 有普通卷积也有深度可分离卷积
        if t == 'C':
            # L.append(wn(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)))
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'c':
            L.append(DSC_augment(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            # L.append(DS_convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'D':
            L.append(DDF_convolution(in_channels=in_channels, out_channels=out_channels))
        # 全部换成深度可分离卷积
        # if t == 'c' or t == 'C':
        #     L.append(DS_convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        # IMDN-baseline，普通的卷积方法
        # if t == 'C' or t=='c':
        #     L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'd':
            L.append(DDFUpPack(in_channels=in_channels, scale_factor=4))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            # L.append(nn.PReLU(num_parameters=64, init=0.25))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


# --------------------------------------------
# inverse of pixel_shuffle
# --------------------------------------------
def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


# --------------------------------------------
# conditional batch norm
# https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
# --------------------------------------------
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


# --------------------------------------------
# Concat the output of a submodule to its input
# --------------------------------------------
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + 'concat'


# --------------------------------------------
# sum the output of a submodule to its input
# --------------------------------------------
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res


# --------------------------------------------
# simplified information multi-distillation block (IMDB)
# x + conv1(concat(split(relu(conv(x)))x3))
# --------------------------------------------
# 将IMDB里面的卷积全部换成depthwise convolution
class IMDBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='cL', d_rate=0.25, negative_slope=0.05):
        super(IMDBlock, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C' or 'c', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
    
    # 不加注意力机制的版本
    def forward(self, x):
        d1, r1 = torch.split(self.conv1(x), (self.d_nc, self.r_nc), dim=1)
        d2, r2 = torch.split(self.conv2(r1), (self.d_nc, self.r_nc), dim=1)
        d3, r3 = torch.split(self.conv3(r2), (self.d_nc, self.r_nc), dim=1)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res

class IMDBlock_RFDN(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='cL', d_rate=0.25, negative_slope=0.05):
        super(IMDBlock_RFDN, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C' or 'c', 'convolutional layer first'

        self.conv1_1 = conv(in_channels, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
        self.conv2_1 = conv(in_channels, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
        self.conv3_1 = conv(in_channels, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)

        self.conv1_3 = nn.Sequential(ShortcutBlock(conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode[0], negative_slope)), conv(mode=mode[1]))
        self.conv2_3 = nn.Sequential(ShortcutBlock(conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode[0], negative_slope)), conv(mode=mode[1]))
        self.conv3_3 = nn.Sequential(ShortcutBlock(conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode[0], negative_slope)), conv(mode=mode[1]))

        self.conv4 = conv(in_channels, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
    
    # 不加注意力机制的版本
    def forward(self, x):
        d1 = self.conv1_1(x)
        r1 = self.conv1_3(x)
        d2 = self.conv2_1(r1)
        r2 = self.conv2_3(r1)
        d3 = self.conv3_1(r2)
        r3 = self.conv3_3(r2)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res

class RFDN(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CL', d_rate=0.25, negative_slope=0.05):
        super(RFDN, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C' or 'c', 'convolutional layer first'

        self.conv1_1 = conv(in_channels, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
        self.conv2_1 = conv(in_channels, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
        self.conv3_1 = conv(in_channels, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)

        self.conv1_3 = nn.Sequential(ShortcutBlock(conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode[0], negative_slope)), conv(mode=mode[1]))
        self.conv2_3 = nn.Sequential(ShortcutBlock(conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode[0], negative_slope)), conv(mode=mode[1]))
        self.conv3_3 = nn.Sequential(ShortcutBlock(conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode[0], negative_slope)), conv(mode=mode[1]))

        self.conv4 = conv(in_channels, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
    
    # 不加注意力机制的版本
    def forward(self, x):
        d1 = self.conv1_1(x)
        r1 = self.conv1_3(x)
        d2 = self.conv2_1(r1)
        r2 = self.conv2_3(r1)
        d3 = self.conv3_1(r2)
        r3 = self.conv3_3(r2)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res

class IMDBlock_Mix(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='cL', d_rate=0.25, negative_slope=0.05):
        super(IMDBlock_Mix, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C' or 'c', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, 'CL', negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
    
    # 不加注意力机制的版本
    def forward(self, x):
        d1, r1 = torch.split(self.conv1(x), (self.d_nc, self.r_nc), dim=1)
        d2, r2 = torch.split(self.conv2(r1), (self.d_nc, self.r_nc), dim=1)
        d3, r3 = torch.split(self.conv3(r2), (self.d_nc, self.r_nc), dim=1)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res

class IMDBlock_DDF(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='DL', d_rate=0.25, negative_slope=0.05):
        super(IMDBlock_DDF, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C' or 'c' or 'D', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
    
    # 不加注意力机制的版本
    def forward(self, x):
        d1, r1 = torch.split(self.conv1(x), (self.d_nc, self.r_nc), dim=1)
        d2, r2 = torch.split(self.conv2(r1), (self.d_nc, self.r_nc), dim=1)
        d3, r3 = torch.split(self.conv3(r2), (self.d_nc, self.r_nc), dim=1)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res

# 实现通道注意力机制，这里用的是卷积核权重,每次挑出16个通道，其他64个继续卷积
class IMDBlock_weight_attention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='cL', d_rate=0.25, negative_slope=0.05):
        super(IMDBlock_weight_attention, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C' or 'c', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        # self.conv2 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        # self.conv3 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        # self.conv4 = conv(in_channels, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
    
    # 通道排序切分,用op.weight的L1 normal
    def re_organize_channel(self, x, op):
        importance = torch.sum(
            torch.abs(list(op)[1].weight.data * list(op)[0].weight.data.transpose(0,1).contiguous()),
            dim = (1, 2, 3)
        )
        importance_index = torch.sort(importance, descending=True)[1]
        d_index = importance_index[:self.d_nc]
        r_index = importance_index[self.d_nc:]
        x = op(x)
        d = x[:, d_index, ...]
        r = x[:, r_index, ...]
        # r = x
        return d, r

    def forward(self, x):
        d1, r1 = self.re_organize_channel(x, self.conv1)
        d2, r2 = self.re_organize_channel(r1, self.conv2)
        d3, r3 = self.re_organize_channel(r2, self.conv3)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res

class IMDBlock_weight_OFA(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='cL', d_rate=0.25, negative_slope=0.05):
        super(IMDBlock_weight_OFA, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C' or 'c', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(in_channels, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
    
    # 通道排序切分,用op.weight的L1 normal
    def re_organize_channel(self, x, op):
        x = op(x)
        importance = torch.sum(
            torch.abs(list(op)[1].weight.data * list(op)[0].weight.data.transpose(0,1).contiguous()),
            dim = (0, 2, 3)
        )
        importance_index = torch.sort(importance, descending=True)[1]
        d_index = importance_index[:self.d_nc]
        r_index = importance_index[self.d_nc:]
        d = x[:, d_index, ...]
        # r = x[:, r_index, ...]
        r = x
        return d, r

    def forward(self, x):
        d1, r1 = self.re_organize_channel(x, self.conv1)
        d2, r2 = self.re_organize_channel(r1, self.conv2)
        d3, r3 = self.re_organize_channel(r2, self.conv3)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res

# 实现通道注意力机制，这里用的是激活值,每次挑出16个通道，其他64个继续卷积
class IMDBlock_activation_attention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='cL', d_rate=0.25, negative_slope=0.05):
        super(IMDBlock_activation_attention, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C' or 'c', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        # self.conv2 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        # self.conv3 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        # self.conv4 = conv(in_channels, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
    
    # 通道切分排序，用x.activation的L1 normal
    def re_organize_channel(self, x, op):
        x = op(x)
        importance = torch.sum(
            torch.abs(x),
            dim = (0,2,3)
        )
        importance_index = torch.sort(importance, descending=True)[1]
        d_index = importance_index[:self.d_nc]
        r_index = importance_index[self.d_nc:]
        d = x[:, d_index, ...]
        r = x[:, r_index, ...]
        # r = x
        return d, r

    def forward(self, x):
        d1, r1 = self.re_organize_channel(x, self.conv1)
        d2, r2 = self.re_organize_channel(r1, self.conv2)
        d3, r3 = self.re_organize_channel(r2, self.conv3)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res

# 实现通道注意力机制，这里用的是SE模块,每次挑出16个通道，其他64个继续卷积
class IMDBlock_SE_attention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='cL', d_rate=0.25, negative_slope=0.05):
        super(IMDBlock_SE_attention, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C' or 'c', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(in_channels, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.se1 = CALayer()
        self.se2 = CALayer()
        self.se3 = CALayer()
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode='C', negative_slope=negative_slope)
    
    # 通道切分排序，用包含了se的CA模块，d:r=16：64
    def re_organize_channel(self, x, op, se):
        x = op(x)
        x, importance = se(x)
        importance = torch.sum(importance, dim=(0, 2, 3))
        importance_index = torch.sort(importance, descending=True)[1]
        d_index = importance_index[:self.d_nc]
        d = x[:, d_index, ...]
        r = x
        return d, r

    def forward(self, x):
        d1, r1 = self.re_organize_channel(x, self.conv1, self.se1)
        d2, r2 = self.re_organize_channel(r1, self.conv2, self.se2)
        d3, r3 = self.re_organize_channel(r2, self.conv3, self.se3)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res

# --------------------------------------------
# Enhanced Spatial Attention (ESA)
# --------------------------------------------
class ESA(nn.Module):
    def __init__(self, channel=64, reduction=4, bias=True):
        super(ESA, self).__init__()
        #               -->conv3x3(conv21)-----------------------------------------------------------------------------------------+
        # conv1x1(conv1)-->conv3x3-2(conv2)-->maxpool7-3-->conv3x3(conv3)(relu)-->conv3x3(conv4)(relu)-->conv3x3(conv5)-->bilinear--->conv1x1(conv6)-->sigmoid
        self.r_nc = channel // reduction
        self.conv1 = nn.Conv2d(channel, self.r_nc, kernel_size=1)
        self.conv21 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=1)
        self.conv2 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.r_nc, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.max_pool2d(self.conv2(x1), kernel_size=7, stride=3)  # 1/6
        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        x2 = F.interpolate(self.conv5(x2), (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        x2 = self.conv6(x2 + self.conv21(x1))
        return x.mul(self.sigmoid(x2))
        # return x.mul_(self.sigmoid(x2))


class CFRB(nn.Module):
    def __init__(self, in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1, bias=True, mode='CL', d_rate=0.5, negative_slope=0.05):
        super(CFRB, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = in_channels  # int(in_channels - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv1_d = conv(in_channels, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv1_r = conv(in_channels, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv2_d = conv(self.r_nc, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv2_r = conv(self.r_nc, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv3_d = conv(self.r_nc, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv3_r = conv(self.r_nc, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv4_d = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.act = conv(mode=mode[-1], negative_slope=negative_slope)
        self.esa = ESA(in_channels, reduction=4, bias=True)

    def forward(self, x):
        d1 = self.conv1_d(x)
        x = self.act(self.conv1_r(x)+x)
        d2 = self.conv2_d(x)
        x = self.act(self.conv2_r(x)+x)
        d3 = self.conv3_d(x)
        x = self.act(self.conv3_r(x)+x)
        x = self.conv4_d(x)
        x = self.act(torch.cat([d1, d2, d3, x], dim=1))
        x = self.esa(self.conv1x1(x))
        return x


# --------------------------------------------
# Channel Attention (CA) Layer
# --------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return x * y, y


# --------------------------------------------
# Residual Channel Attention Block (RCAB)
# --------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16, negative_slope=0.2):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        return res + x


# --------------------------------------------
# Residual Channel Attention Group (RG)
# --------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16, nb=12, negative_slope=0.2):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding, bias, mode, reduction, negative_slope)  for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        self.rg = nn.Sequential(*RG)  # self.rg = ShortcutBlock(nn.Sequential(*RG))

    def forward(self, x):
        res = self.rg(x)
        return res + x


# --------------------------------------------
# Residual Dense Block
# style: 5 convs
# --------------------------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(nc+gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(nc+2*gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(nc+3*gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv5 = conv(nc+4*gc, nc, kernel_size, stride, padding, bias, mode[:-1], negative_slope)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


# --------------------------------------------
# Residual in Residual Dense Block
# 3x5c
# --------------------------------------------
class RRDB(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(RRDB, self).__init__()

        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x


"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_ddfup
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# 
# --------------------------------------------
"""
# --------------------------------------------
# conv + ddfup (+ leakyrelu)
# --------------------------------------------
def upsample_ddfup(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='d', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4', 'd'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode+'CL', negative_slope=negative_slope)
    return up1

# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


'''
# --------------------------------------------
# NonLocalBlock2D:
# embedded_gaussian
# +W(softmax(thetaXphi)Xg)
# --------------------------------------------
'''


# --------------------------------------------
# non-local block with embedded_gaussian
# https://github.com/AlexHex7/Non-local_pytorch
# --------------------------------------------
class NonLocalBlock2D(nn.Module):
    def __init__(self, nc=64, kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='maxpool', negative_slope=0.2):

        super(NonLocalBlock2D, self).__init__()

        inter_nc = nc // 2
        self.inter_nc = inter_nc
        self.W = conv(inter_nc, nc, kernel_size, stride, padding, bias, mode='C'+act_mode)
        self.theta = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

        if downsample:
            if downsample_mode == 'avgpool':
                downsample_block = downsample_avgpool
            elif downsample_mode == 'maxpool':
                downsample_block = downsample_maxpool
            elif downsample_mode == 'strideconv':
                downsample_block = downsample_strideconv
            else:
                raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
            self.phi = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
            self.g = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
        else:
            self.phi = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')
            self.g = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
