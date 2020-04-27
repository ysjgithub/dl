
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from C1onvlutionalNetworks import FC
from C1onvlutionalNetworks.ConvlutionalNetworks import Conv, Flattan, MaxPooling, SoftMax, Relu, Model


def MSE(y,t):
    # y,t 都是 10*10
    return y-t
    # 输出是 10 *10

model = Model([
        Conv(1, 32, 3, 1, 1),
        Relu(),
        MaxPooling(2, 2),
        Conv(32, 64, 3, 1, 1),
        Relu(),
        MaxPooling(2, 2),
        # Conv(64,128,3,1,1),
        # activition(),
        # MaxPooling(2, 2),
        # Conv(128,256,3,1,1),
        # activition(),
        # MaxPooling(2, 2),
        Flattan(),
        FC(3136,96),
        Relu(),
        FC(96,10),
        Relu(),
        SoftMax()
    ])


transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_loader = DataLoader(
    datasets.MNIST(root='../.data/mnist', train=True, download=True, transform=transform_train), batch_size=20,
    shuffle=True, num_workers=2, drop_last=True
)

test_loader = DataLoader(
    datasets.MNIST(root='../.data/mnist', train=False, download=True, transform=transform_test),
    batch_size=100, shuffle=False, num_workers=2, drop_last=True
)

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def test():
    correct = 0
    for x,y in test_loader:
        x = x.numpy()
        y = one_hot(np.array(y.numpy()),10)
        s = model.forword(x)
        correct+=np.sum(np.argmax(y,axis=1)==np.argmax(s,axis=1))
    print("test_correct",correct)


iter = 0
for i in range(2):
    for x,y in train_loader:
        print(iter)
        iter+=1
        x = x.numpy()
        y = one_hot(np.array(y.numpy()),10)
        s = model.forword(x)
        r = MSE(s,y)
        print(np.argmax(y,axis=1),np.argmax(s,axis=1))
        print(np.sum(np.argmax(y,axis=1)==np.argmax(s,axis=1)))
        model.backword(r)
        if iter%1000 == 0:
            test()



