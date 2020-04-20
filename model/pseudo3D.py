import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import dataloader, dataset
from torchvision import models
from torchsummary import summary


class BottleNeck(nn.Module):
    def __init__(self, input_channel, out_channel, stride=1):
        super(BottleNeck, self).__init__()
        self.down_sample = int(out_channel / 4)
        print(self.down_sample,input_channel,out_channel)
        self.left = nn.Sequential(
            nn.Conv3d(input_channel, self.down_sample, 1, stride=1,bias=False),
            nn.BatchNorm3d(self.down_sample),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.down_sample, self.down_sample, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(self.down_sample),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.down_sample, out_channel, 1, stride=1,bias=False),
        )
        self.shortcut = nn.Sequential()
        if input_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(input_channel, out_channel, (1,1,1), stride=stride, bias=True),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channel, out_channel, (1,3,3), stride=1,padding=(0,1,1), bias=True),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channel, out_channel, (3,1,1), stride=1,padding=(1,0,0), bias=True),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channel, out_channel, (1, 1, 1), stride=1, bias=True),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        print("left",out.shape)
        shortcut = self.shortcut(x)
        print("short",shortcut.shape)
        out += shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_layers, block, num_classes=10):
        self.input_channel = 64
        super(ResNet, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,3,3),(1,2,2))
        )

        self.res = nn.Sequential(
            self.layer(block, 256, num_layers[0], 2),
            self.layer(block, 512, num_layers[1], 2),
            self.layer(block, 1024, num_layers[2], 2),
            self.layer(block, 2048, num_layers[3], 2),
            nn.AvgPool3d((1,2,2)),
            nn.Flatten(),
            nn.Linear(2048, 2048),
            nn.Linear(2048,num_classes),
            nn.Softmax()
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

net = ResNet([3,4,6,3],BottleNeck, 101).to(device)
summary(net, (3,16, 112, 112))


