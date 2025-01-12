'''
Modified from https://github.com/pytorch/vision.git
https://pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/
https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
'''
import math
from collections import OrderedDict

import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo

from models.quantization import *


from functools import partial
from typing import Any, cast, Dict, List, Optional, Union



__all__ = [
    'VGG', 'VGG_mid', 'vgg11_bn_imagenet', 'vgg11_bn_imagenet_mid',  'vgg16_bn_imagenet', 'vgg16_bn_imagenet_mid',

]

model_urls = {
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5, n_bits=8
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.n_bits = n_bits
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            quan_Linear(512 * 7 * 7, 4096, n_bits=n_bits),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            quan_Linear(4096, 4096, n_bits=n_bits),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            # nn.Linear(4096, num_classes),
        )
        self.linear = quan_Linear(4096, num_classes, n_bits=n_bits)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    # if m.bias is not None:
                    #     nn.init.constant_(m.bias, 0)

                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.linear(x)

        return x

class VGG_mid(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5, n_bits=8
    ) -> None:
        super(VGG_mid, self).__init__()
        self.features = features
        self.n_bits = n_bits
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            quan_Linear(512 * 7 * 7, 4096, n_bits=n_bits),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            quan_Linear(4096, 4096, n_bits=n_bits),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            # nn.Linear(4096, num_classes),
        )
        self.linear = quan_Linear(4096, num_classes, n_bits=n_bits)
        self.mid_dim = 4096

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    # if m.bias is not None:
                    #     nn.init.constant_(m.bias, 0)

                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                # elif isinstance(m, nn.Linear):
                #     nn.init.normal_(m.weight, 0, 0.01)
                #     nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

def make_layers_vgg(cfg: List[Union[str, int]], batch_norm: bool = False, n_bits: int = 8) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = quan_Conv2d(in_channels, v, kernel_size=3, padding=1, n_bits=n_bits)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # 9
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # 11
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], # 14
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'], # 17
}

def vgg11_bn_imagenet(num_classes=1000, n_bits=8, pretrained=True, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    # return VGG_imagenet(make_layers(cfg=cfg['A'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)
    model = VGG(make_layers_vgg(cfg=cfg['A'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg11_bn'])
        model_dict = model.state_dict()
        print(pretrained_dict.keys())
        print('-'*100)
        print(model_dict.keys())

        pretrained_dict['linear.weight'] = pretrained_dict['classifier.6.weight']
        pretrained_dict['linear.bias'] = pretrained_dict['classifier.6.bias']
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        modul_dict = OrderedDict()
        for k, v in model_dict.items():
            modul_dict['module.' + str(k)] = v

        print('-'*100)
        print('-'*100)
        print(model.state_dict().keys())
        print('-'*100)
        print(modul_dict.keys())

    return model, modul_dict


def vgg11_bn_imagenet_mid(num_classes=1000, n_bits=8, pretrained=True, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    # return VGG_imagenet(make_layers(cfg=cfg['A'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)
    model = VGG_mid(make_layers_vgg(cfg=cfg['A'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg11_bn'])
        model_dict = model.state_dict()
        print(pretrained_dict.keys())
        print('-'*100)
        print(model_dict.keys())

        pretrained_dict['linear.weight'] = pretrained_dict['classifier.6.weight']
        pretrained_dict['linear.bias'] = pretrained_dict['classifier.6.bias']
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        modul_dict = OrderedDict()
        for k, v in model_dict.items():
            modul_dict['module.' + str(k)] = v

    return model, modul_dict



# def vgg13_imagenet(num_classes=1000, n_bits=8):
#     """VGG 13-layer model (configuration "B")"""
#     return VGG_imagenet(make_layers(cfg=cfg['B'], batch_norm=False, n_bits=n_bits), n_bits=n_bits)
#
#
# def vgg13_bn_imagenet(num_classes=1000, n_bits=8):
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     return VGG_imagenet(make_layers(cfg=cfg['B'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)


def vgg16_bn_imagenet(num_classes=1000, n_bits=8, pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers_vgg(cfg=cfg['D'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        model_dict = model.state_dict()
        pretrained_dict['linear.weight'] = pretrained_dict['classifier.6.weight']
        pretrained_dict['linear.bias'] = pretrained_dict['classifier.6.bias']
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        modul_dict = OrderedDict()
        for k, v in model_dict.items():
            modul_dict['module.' + str(k)] = v

    return model, modul_dict


def vgg16_bn_imagenet_mid(num_classes=1000, n_bits=8, pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG_mid(make_layers_vgg(cfg=cfg['D'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        model_dict = model.state_dict()
        print(pretrained_dict.keys())
        print('-'*100)
        print(model_dict.keys())

        pretrained_dict['linear.weight'] = pretrained_dict['classifier.6.weight']
        pretrained_dict['linear.bias'] = pretrained_dict['classifier.6.bias']
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        modul_dict = OrderedDict()
        for k, v in model_dict.items():
            modul_dict['module.' + str(k)] = v

    return model, modul_dict


# def vgg16_bn_imagenet(num_classes=1000, n_bits=8):
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     return VGG_imagenet(make_layers(cfg=cfg['D'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)
#
#
# def vgg19_imagenet(num_classes=1000, n_bits=8):
#     """VGG 19-layer model (configuration "E")"""
#     return VGG_imagenet(make_layers(cfg=cfg['E'], batch_norm=False, n_bits=n_bits), n_bits=n_bits)
#
#
# def vgg19_bn_imagenet(num_classes=1000, n_bits=8):
#     """VGG 19-layer model (configuration 'E') with batch normalization"""
#     return VGG_imagenet(make_layers(cfg=cfg['E'], batch_norm=True, n_bits=n_bits), n_bits=n_bits)