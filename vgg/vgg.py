import torch
import torch.nn as nn
from torchsummary import summary


def panel3X3(input_channel,out_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, out_channel,3, stride=1,padding=1, bias=True),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )

class Vgg11(nn.Module):
    def __init__(self):
        super(Vgg11,self).__init__()
        self.s = nn.Sequential(
            panel3X3(3,64),

            nn.MaxPool2d(2, 2),

            panel3X3(64,128),

            nn.MaxPool2d(2, 2),

            panel3X3(128,256),
            panel3X3(256,256),

            nn.MaxPool2d(2, 2),

            panel3X3(256, 512),
            panel3X3(512, 512),

            nn.MaxPool2d(2,2),

            panel3X3(512, 512),
            panel3X3(512, 512),

            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(512,512),
            nn.Linear(512,512),
            nn.Linear(512,10),
        )

    def forward(self, x):
        out  = self.s(x)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Vgg11().to(device)
summary(net, (3, 32, 32))