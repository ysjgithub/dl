import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models


def panel3X3(input_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, out_channel, 3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


def panel1X1(input_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, out_channel, 1, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )

classafication = nn.Sequential(
    *[nn.Flatten(), nn.Linear(512, 512), nn.Linear(512, 512), nn.Linear(512, 10)]
)

class VGG11(nn.Module):
    def __init__(self,lrn=False):
        super(VGG11, self).__init__()

        self.downsample = nn.Sequential(
            *[panel3X3(3,64) ,nn.LocalResponseNorm(64) if lrn else nn.Sequential()],
            nn.MaxPool2d(2, 2),

            *[panel3X3(64, 128)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(128, 256),panel3X3(256, 256)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(256, 512),panel3X3(512, 512)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(512, 512),panel3X3(512, 512)],
            nn.MaxPool2d(2, 2),

            classafication
        )

    def forward(self, x):
        out = self.downsample(x)
        return out


class VGG16(nn.Module):
    def __init__(self,size=3):
        super(VGG16,self).__init__()

        self.downsample = nn.Sequential(
            *[panel3X3(3, 64), panel3X3(64,64)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(64, 128)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(128, 256), panel3X3(256, 256),panel3X3(256, 256) if size==3 else panel1X1(356,256)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(256, 512), panel3X3(512, 512),panel3X3(512, 512) if size==3 else panel1X1(356,256)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(512, 512), panel3X3(512, 512),panel3X3(512, 512) if size==3 else panel1X1(356,256)],
            nn.MaxPool2d(2, 2),

            classafication
        )

    def forward(self,x):
        out = self.downsample(x)
        return out

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19,self).__init__()

        self.downsample = nn.Sequential(
            *[panel3X3(3, 64), panel3X3(64,64)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(64, 128)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(128, 256), panel3X3(256, 256),panel3X3(256, 256),panel3X3(256, 256)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(256, 512), panel3X3(512, 512),panel3X3(512, 512),panel3X3(512, 512)],
            nn.MaxPool2d(2, 2),

            *[panel3X3(512, 512), panel3X3(512, 512),panel3X3(512, 512),panel3X3(512, 512)],
            nn.MaxPool2d(2, 2),

            classafication
        )

    def forward(self,x):
        out = self.downsample(x)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG19().to(device)
summary(model,(3,32,32))