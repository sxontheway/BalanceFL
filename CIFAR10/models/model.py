
from utils import *
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, planes):
        super(LambdaLayer, self).__init__()
        self.planes = planes

    def forward(self, x):
        return self.lamda(x)

    def lamda(self, x):
        return F.pad(
            x[:, :, ::2, ::2], 
            (0, 0, 0, 0, self.planes // 4, self.planes // 4), 
            "constant", 
            0,)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)  # change from ReLU to LeakyReLU
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


import math
class ResNet(nn.Module):

    def __init__(self, block, layers, l2_norm, channels=[64,128,256,512], use_fc=False, dropout=False):
        self.inplanes = channels[0]
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)  # change from ReLU to LeakyReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.l2_norm = l2_norm
        self.use_fc = use_fc
        self.use_dropout = dropout
        self.beta = 0.5

        out_dim = channels[3]
        if self.use_fc:
            print('Using fc.')
            self.fc_add = nn.Linear(out_dim*block.expansion, out_dim)

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, *args):
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

        self.x = x  

        if self.use_fc:
            x = F.relu(self.fc_add(x))    # change from ReLU to LeakyReLU

        if self.use_dropout:
            x = self.dropout(x)

        if self.l2_norm:
            norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8
            x = x/norm

        # Tukey's transformation
        # x = torch.pow(x, self.beta)    # 尝试改变cov？ 试下10 shot，差别大一点？

        return x


class DotProduct_Classifier(nn.Module):
    def __init__(self, num_classes, feat_dim, l2_norm=False, bias=True, scale=1, *args):
        super(DotProduct_Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(feat_dim, num_classes, bias=bias)
        self.l2_norm = l2_norm
        self.weight_scale = Parameter(torch.ones(num_classes))
        self.scale = scale
        self.apply(_weights_init)
        if l2_norm == True:
            self.weight_norm()
        self.stage1_training()

    def weight_norm(self):
        norm = torch.norm(self.fc.weight.data, p=2, dim=1, keepdim=True) + 1e-8
        self.fc.weight.data = self.fc.weight.data/norm

    def stage1_training(self):
        self.weight_scale.requires_grad = False
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = True

    def stage2_training(self):
        self.weight_scale.requires_grad = True
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = False

    def update_weight(self):
        self.fc.weight.data = self.fc.weight.data * self.weight_scale.unsqueeze(1)
        self.weight_scale.data = torch.ones(self.num_classes).to(self.fc.weight.data.device)
    
    def forward(self, x, *args):
        x = self.fc(x) * self.weight_scale
        if self.scale:
            x = self.scale * x
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)  # change from ReLU to LeakyReLU
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            """
            For CIFAR10: the ResNet paper uses option A.
            For others: Use option B, use 1*1 conv for downsampling
            """
            if downsample is None:
                if option == "A":
                    # since planes is larger than in_planes, the shortcuted weight need expansion
                    self.shortcut = LambdaLayer(planes)
                elif option == "B":
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(
                            in_planes,
                            self.expansion * planes,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(self.expansion * planes),
                    )
            else:
                self.shortcut = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x) #x还是out
        out = F.relu(out)     # change from ReLU to LeakyReLU

        return out


import numpy as np
class BBN_ResNet_Cifar(nn.Module):
    def __init__(self, block, num_blocks, l2_norm, feat_dim=128):
        super(BBN_ResNet_Cifar, self).__init__()
        if feat_dim == 128:
            channels = [16,32,64]
        elif feat_dim == 256:
            channels = [32,64,128]
        elif feat_dim == 512:
            channels = [64,128,256]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])

        # self.in_planes will be changed in self._make_layer()
        self.in_planes = channels[0]
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1)      # output size (N*channels[0]*32*32)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2)      # output size (N*channels[1]*16*16)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2] - 1, stride=2)  # output size (N*channels[2]*8*8)
        self.cb_block = block(self.in_planes, self.in_planes, stride=1)         # output size (N*channels[2]*8*8)
        self.rb_block = block(self.in_planes, self.in_planes, stride=1)         # output size (N*channels[2]*8*8)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l2_norm = l2_norm
        self.apply(_weights_init)

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)["state_dict_best"]['feat_model']

        new_dict = OrderedDict()

        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "fc" not in k and "classifier" not in k:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def _make_layer(self, block, planes, num_blocks, stride, add_flag=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))   # change from ReLU to LeakyReLU
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if "feature_cb" in kwargs:
            out = self.cb_block(out)
            return out
        elif "feature_rb" in kwargs:
            out = self.rb_block(out)
            return out

        out1 = self.cb_block(out)
        out2 = self.rb_block(out)
        out = torch.cat((out1, out2), dim=1)
        
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)

        if self.l2_norm:
            norm = torch.norm(out, p=2, dim=1, keepdim=True) + 1e-8
            out = out/norm
        return out


class BasicBlock_new(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BBN_ResNet_Cifar_new(nn.Module): 
    def __init__(self, block, num_blocks, l2_norm, feat_dim=128):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if feat_dim == 128:
            channels = [16,32,64,128]
        elif feat_dim == 256:
            channels = [32,64,128,256]
        elif feat_dim == 512:
            channels = [64,128,256,512]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

