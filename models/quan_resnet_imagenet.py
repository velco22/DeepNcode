import time
from collections import OrderedDict

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .quantization import *
import math

__all__ = [
    'ResNet', 'resnet18_quan', 'resnet18_quan_mid', 'resnet34_quan', 'resnet34_quan_mid', 'resnet50_quan', 'resnet50_quan_mid', 'resnet101',
    'resnet152'
]

model_urls = {
    # original
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

    # new
    # 'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    # 'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    # 'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
}


def conv3x3(in_planes, out_planes, stride=1, n_bits=8):
    """3x3 convolution with padding"""
    return quan_Conv2d(in_planes,
                       out_planes,
                       kernel_size=3,
                       stride=stride,
                       padding=1,
                       bias=False,
                       n_bits=n_bits)


def conv1x1(in_planes, out_planes, stride=1, n_bits=8):
    """1x1 convolution"""
    return quan_Conv2d(in_planes,
                       out_planes,
                       kernel_size=1,
                       stride=stride,
                       bias=False,
                       n_bits=n_bits)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_bits=8):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_bits=8):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, n_bits=n_bits)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 n_bits=8,
                 zero_init_residual=False
                 ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.n_bits = n_bits
        self.conv1 = quan_Conv2d(3,
                                 64,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3,
                                 bias=False,
                                 n_bits=self.n_bits)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = quan_Linear(512 * block.expansion, num_classes, n_bits=self.n_bits)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print("init conv2d")
                # nn.init.kaiming_normal_(m.weight,
                #                         mode='fan_out',
                #                         nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                print("init bn2d")
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # if isinstance(m, quan_Conv2d):
            #     print("init quan_conv2d")
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            #     if m.bias:
            #         m.bias.data.zero_()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, n_bits=self.n_bits),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_bits=self.n_bits))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_bits=self.n_bits))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class ResNet_mid(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 n_bits=8,
                 zero_init_residual=False
                 ):
        super(ResNet_mid, self).__init__()
        self.inplanes = 64
        self.n_bits = n_bits
        self.conv1 = quan_Conv2d(3,
                                 64,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3,
                                 bias=False,
                                 n_bits=self.n_bits)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = quan_Linear(512 * block.expansion, num_classes, n_bits=self.n_bits)
        self.mid_dim = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print("init conv2d")
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                print("init bn2d")
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, n_bits=self.n_bits),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_bits=self.n_bits))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_bits=self.n_bits))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.linear(x)

        return x


def resnet18_quan(num_output=10, n_bits=8, pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:  
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs, n_bits=n_bits)
    if pretrained:

        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()

        if n_bits == 8:
            pretrained_dict['linear.weight'] = pretrained_dict['fc.weight']
            pretrained_dict['linear.bias'] = pretrained_dict['fc.bias']
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


        elif n_bits == 4:
            best_dict = torch.load('.\\results\\imagenet\\resnet18_quan4_\\model_best.pth.tar')['state_dict']

            modul_dict = OrderedDict()
            for k, v in best_dict.items():
                modul_dict[str(k)[7:]] = v

            model.load_state_dict(modul_dict)

            return model, best_dict



def resnet18_quan_mid(num_output=10, n_bits=8, pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_mid(BasicBlock, [2, 2, 2, 2], **kwargs, n_bits=n_bits)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()

        if n_bits == 8:
            # best_dict = torch.load('.\\results\\imagenet\\resnet18_quan4\\state_dict.pth.tar')
            print(model)
            print("------------------")
            print("pretrained_dict")
            print(pretrained_dict.keys())
            print("------------------")
            # print(best_dict.keys())
            # print("------------------")
            # print(model_dict.keys())
            # time.sleep(100)


            pretrained_dict['linear.weight'] = pretrained_dict['fc.weight']
            pretrained_dict['linear.bias'] = pretrained_dict['fc.bias']
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

        elif n_bits == 4:
            best_dict = torch.load('.\\results\\imagenet\\resnet18_quan4_\\model_best.pth.tar')['state_dict']

            modul_dict = OrderedDict()
            for k, v in best_dict.items():
                modul_dict[str(k)[7:]] = v

            model.load_state_dict(modul_dict)

            return model, best_dict


def resnet34_quan(num_output=10, n_bits=8, pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs, n_bits=n_bits)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        pretrained_dict['linear.weight'] = pretrained_dict['fc.weight']
        pretrained_dict['linear.bias'] = pretrained_dict['fc.bias']
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

def resnet34_quan_mid(num_output=10, n_bits=8, pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_mid(BasicBlock, [3, 4, 6, 3], **kwargs, n_bits=n_bits)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        pretrained_dict['linear.weight'] = pretrained_dict['fc.weight']
        pretrained_dict['linear.bias'] = pretrained_dict['fc.bias']
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

def resnet50_quan(num_output=10, n_bits=8, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs, n_bits=n_bits)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict['linear.weight'] = pretrained_dict['fc.weight']
        pretrained_dict['linear.bias'] = pretrained_dict['fc.bias']
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

def resnet50_quan_mid(num_output=10, n_bits=8, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_mid(Bottleneck, [3, 4, 6, 3], **kwargs, n_bits=n_bits)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict['linear.weight'] = pretrained_dict['fc.weight']
        pretrained_dict['linear.bias'] = pretrained_dict['fc.bias']
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

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
