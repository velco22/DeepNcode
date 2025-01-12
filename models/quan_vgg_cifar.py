'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

from models.quantization import *

__all__ = [
    'VGG_cifar10', 'vgg11_quan_cifar10', 'vgg11_bn_quan_cifar10', 'vgg13_cifar10', 'vgg13_bn_cifar10', 'vgg16_cifar10', 'vgg16_bn_cifar10',
    'vgg19_bn_cifar10', 'vgg19_cifar10', 'vgg11_bn_quan_cifar10_mid', 'VGG_cifar10_mid',
]


class VGG_cifar10(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, n_bits=8):
        super(VGG_cifar10, self).__init__()
        self.features = features
        self.n_bits = n_bits
        self.classifier = nn.Sequential(
            nn.Dropout(),
            quan_Linear(512, 512, n_bits=self.n_bits),
            nn.ReLU(True),
            nn.Dropout(),
            quan_Linear(512, 512, n_bits=self.n_bits),
            nn.ReLU(True),
        )
        self.linear = quan_Linear(512, 10, n_bits=self.n_bits)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.linear(x)
        return x


class VGG_cifar10_mid(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, n_bits=8):
        super(VGG_cifar10_mid, self).__init__()
        self.features = features
        self.n_bits = n_bits
        self.classifier = nn.Sequential(
            nn.Dropout(),
            quan_Linear(512, 512, n_bits=self.n_bits),
            nn.ReLU(True),
            nn.Dropout(),
            quan_Linear(512, 512, n_bits=self.n_bits),
            nn.ReLU(True),

        )
        self.linear = quan_Linear(512, 10, n_bits=self.n_bits)
        self.mid_dim = 512

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False, n_bits=8):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = quan_Conv2d(in_channels, v, kernel_size=3, padding=1, n_bits=n_bits)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11_quan_cifar10(num_classes=10, n_bits=8, output_act='linear'):
    """VGG 11-layer model (configuration "A")"""
    return VGG_cifar10(make_layers(cfg=cfg['A'], batch_norm=False, n_bits=n_bits), n_bits=n_bits)


def vgg11_bn_quan_cifar10(num_classes=10, n_bits=8, output_act='linear'):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG_cifar10(make_layers(cfg=cfg['A'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)

def vgg11_bn_quan_cifar10_mid(num_classes=10, n_bits=8, output_act='linear'):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG_cifar10_mid(make_layers(cfg=cfg['A'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)



def vgg13_cifar10(num_classes=10, n_bits=8, output_act='linear'):
    """VGG 13-layer model (configuration "B")"""
    return VGG_cifar10(make_layers(cfg=cfg['B'], batch_norm=False, n_bits=n_bits), n_bits=n_bits)


def vgg13_bn_cifar10(num_classes=10, n_bits=8, output_act='linear'):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG_cifar10(make_layers(cfg=cfg['B'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)


def vgg16_cifar10(num_classes=10, n_bits=8, output_act='linear'):
    """VGG 16-layer model (configuration "D")"""
    return VGG_cifar10(make_layers(cfg=cfg['D'], batch_norm=False, n_bits=n_bits), n_bits=n_bits)


def vgg16_bn_cifar10(num_classes=10, n_bits=8, output_act='linear'):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG_cifar10(make_layers(cfg=cfg['D'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)


def vgg19_cifar10(num_classes=10, n_bits=8, output_act='linear'):
    """VGG 19-layer model (configuration "E")"""
    return VGG_cifar10(make_layers(cfg=cfg['E'], batch_norm=False, n_bits=n_bits), n_bits=n_bits)


def vgg19_bn_cifar10(num_classes=10, n_bits=8, output_act='linear'):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG_cifar10(make_layers(cfg=cfg['E'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)