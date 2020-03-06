import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import dataloader, dataset
from torchvision import models
from torchsummary import summary


class ResBlock(nn.Module):
    def __init__(self, input_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        print(input_channel,out_channel)
        self.left = nn.Sequential(
            nn.Conv2d(input_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or input_channel != out_channel:
            print(input_channel, out_channel, stride)
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, input_channel, out_channel, stride=1):
        super(BottleNeck, self).__init__()
        self.down_sample = int(out_channel / 4)

        print(self.down_sample,input_channel,out_channel)
        self.left = nn.Sequential(
            nn.Conv2d(input_channel, self.down_sample, 1, stride=1,bias=False),
            nn.BatchNorm2d(self.down_sample),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.down_sample, self.down_sample, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.down_sample),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.down_sample, out_channel, 1, stride=1,bias=False),
        )
        self.shortcut = nn.Sequential()
        if input_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel, out_channel, 1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channel)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        print(out.shape,self.shortcut(x).shape)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        self.input_channel = 64
        super(ResNet, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Sequential(
            self.layer(block, 256, 3, 2),
            self.layer(block, 512, 4, 2),
            self.layer(block, 1024, 6, 2),
            self.layer(block, 2048, 3, 2),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(2048,num_classes)
        )

    def layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        res_layer = []
        for stride in strides:
            res_layer.append(block(self.input_channel, out_channel, stride))
            self.input_channel = out_channel
        return nn.Sequential(*res_layer)

    def forward(self, x):
        out = self.downsample(x)
        out = self.res(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# net = ResNet(ResBlock, 10).to(device)
# summary(net, (3, 32, 32))

net = ResNet(BottleNeck, 10).to(device)
summary(net, (3, 32, 32))
