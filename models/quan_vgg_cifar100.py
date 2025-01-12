"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import math
import torch
import torch.nn as nn
from models.quantization import *

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100, n_bits=8):
        super().__init__()
        self.features = features
        self.n_bits = n_bits

        self.classifier = nn.Sequential(
            quan_Linear(512, 4096, n_bits=self.n_bits),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            quan_Linear(4096, 4096, n_bits=self.n_bits),
            nn.ReLU(inplace=True),
            nn.Dropout(),

        )
        self.linear = quan_Linear(4096, num_class, n_bits=self.n_bits)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        output = self.linear(output)

        return output

class VGG_mid(nn.Module):

    def __init__(self, features, num_class=100, n_bits=8):
        super().__init__()
        self.features = features
        self.n_bits = n_bits

        self.classifier = nn.Sequential(
            quan_Linear(512, 4096, n_bits=self.n_bits),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            quan_Linear(4096, 4096, n_bits=self.n_bits),
            nn.ReLU(inplace=True),
            nn.Dropout(),

        )
        self.linear = quan_Linear(4096, num_class, n_bits=self.n_bits)
        self.mid_dim = 4096

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def make_layers(cfg, batch_norm=False, n_bits=8):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [quan_Conv2d(input_channel, l, kernel_size=3, padding=1, n_bits=n_bits)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg11_bn_quan_cifar100(num_classes=100, n_bits=8, output_act='linear'):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg=cfg['A'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)

def vgg11_bn_quan_cifar100_mid(num_classes=100, n_bits=8, output_act='linear'):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG_mid(make_layers(cfg=cfg['A'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)

def vgg13_bn_quan_cifar100(num_classes=100, n_bits=8, output_act='linear'):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg=cfg['B'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)

def vgg13_bn_quan_cifar100_mid(num_classes=100, n_bits=8, output_act='linear'):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG_mid(make_layers(cfg=cfg['B'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)