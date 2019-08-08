import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from torch_core import Module
BN_MOMENTUM = 0.1


def conv_bn_relu(ni:int, nf:int, ks:int=3, stride:int=1)->nn.Sequential:
    "Create a seuence Conv2d->BatchNorm2d->LeakyReLu layer."
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks//2),
        nn.BatchNorm2d(nf, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True))

def conv_bn(ni:int, nf:int, ks:int=3, stride:int=1)->nn.Sequential:
    "Create a seuence Conv2d->BatchNorm2d-"
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks//2),
        nn.BatchNorm2d(nf, momentum=BN_MOMENTUM))


def init_cnn(m):
    if getattr(m, 'bias', None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    for l in m.children(): init_cnn(l)


class BasicBlock(Module):
    def __init__(self, ni: int, nout: int, downsample=None):
        self.downsample = downsample
        self.conv1 = conv_bn_relu(ni, nout, ks=3)
        self.conv2 = conv_bn(nout, nout, ks=3)

    def forward(self, x):
        conv_out = nn.Sequential(
            self.conv1,
            self.conv2)
        if self.downsample is not None:
            x = self.downsample(x)
        return nn.ReLU(x + conv_out)


class Stem(Module):
    def __init__(self):
        self.conv1 = conv_bn_relu(ni=3, nf=64, ks=3, stride=2)
        self.conv2 = conv_bn_relu(ni=64, nf=64, ks=3, stride=2)
        init_cnn(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ResidualBlock(Module):
    """
    For the 1st stage:
    according to the hrnet paper:
    > The 1st stage contains 4 residual units
    > where each unit is formed by a bottleneck with the width 64,
    > and is followed by one 3×3 convolution reducing the width
    > of feature maps to C
    """

    def __init__(self, ni:int, nout:int,  expansion=4, downsample=None):
        self.downsample = downsample
        self.conv1 = conv_bn_relu(ni, nout, ks=1)
        self.conv2 = conv_bn_relu(nout, nout, ks=3)
        self.conv3 = conv_bn(nout, nout * expansion)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        conv_out = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3)
        return nn.ReLU(identity + conv_out)


class MultiResBlock(Module):
    def __init__(self, channels_in):
        pass
    def forward(self, *input):
        pass


class HRNet(Module):
    def __init__(self, stem, stage1, stage2, stage3, stage4):
        self.stem = stem
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.stage4 = stage4

        init_cnn(self)


    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x

