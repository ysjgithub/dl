import time

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Conv(object):
    def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
        self.input_channel=input_channel
        self.output_channel = output_channel
        self.kernel=kernel
        self.stride=stride
        self.padding=padding
        self.output_maps=[]
        self.core = np.array([np.random.randn(input_channel,kernel,kernel)/np.sqrt(input_channel/2) for _ in range(output_channel)])
        self.gradient =None
        self.input_maps = None
    def forword(self,x):
        # 输入为 128 * 3 * 10 *10
        self.input_maps = x
        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        mini_batch,c,w,h = x.shape
        neww,newh = int((w-self.kernel)/self.stride+1),int((h-self.kernel)/self.stride+1)

        self.output_maps = np.zeros((mini_batch,self.output_channel,neww,newh))
        for idc in range(self.output_channel):
            core = self.core[idc,:,:,:]
            #卷积核对应mini_batch张特征图
            featuremap = np.zeros((mini_batch,neww,newh))

            for i in range(0,neww):
                for j in range(0,newh):
                    newfeaturemap = np.mean(x[:,:,i:i+self.kernel,j:j+self.kernel] * core,axis=(1,2,3)).reshape((1,20))
                    featuremap[:,int(i/self.stride),int(j/self.stride)] = newfeaturemap

        self.output_maps[:,idc,:,:] = featuremap
        #输出为128 * output_channel * 10 *10

    def update(self,y):
        # 更新权值要用到所有参与卷积的点，需要使用padding后的图像
        mini_batch,output_channel,w,h = y.shape
        x = np.pad(self.input_maps, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        # 16 * 9 * 9
        for idc in range(self.output_channel):
            #更新一个核,core input_channel * 3 * 3,使用一组特征图
            core = self.core[idc,:,:,:]
            deltacore = np.zeros(core.shape)

            # 一个mini_batch的梯度加起来
            feature = y[:,idc,:,:].reshape((mini_batch,1,w,h))
            assert feature.shape==(mini_batch,1,w,h)
            feature = np.repeat(feature,self.input_channel,axis=1)
            assert feature.shape==(mini_batch,self.input_channel,w,h)

            # 影响的对象只有一张图,features 7 * 7

            for i in range(self.kernel):
                for j in range(self.kernel):
                    zz= x[:,:,i:i + w, j:j + h] * feature
                    deltacore[:,i,j] += np.mean(zz,axis=(0,2,3))

            deltacore=deltacore/len(y)
            self.core[idc,:,:,:]-=deltacore*0.1

    def backword(self,m):
        # 输入为 output_maps，更新core的
        # 输入为 20 * 8 * 10 *10
        # 计算输入点产生的误差
        y=m
        t = time.time()
        mini_batch, ic, iw, ih = self.input_maps.shape
        y = np.pad(y, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        self.gradient = np.zeros(self.input_maps.shape)
        newcore  = self.core.reshape((self.input_channel,self.output_channel,self.kernel,self.kernel))

        for i in range(self.input_channel):
            for j in range(self.output_channel):
                newcore[i,j,:,:] = newcore[i,j,:,:].T[::-1].T[::-1]

        for idc in range(self.input_channel):
            core = newcore[idc,:,:,:]

            # 卷积核对应mini_batch张特征图
            featuremap = np.zeros((mini_batch, iw, ih))

            for i in range(iw):
                for j in range(ih):
                    newfeaturemap = np.mean(y[:, :, i:i + self.kernel, j:j + self.kernel] * core,axis=(1, 2, 3)).reshape((1, mini_batch))
                    featuremap[:, int(i / self.stride), int(j / self.stride)] = newfeaturemap

            self.gradient[:, idc, :, :] = featuremap

        print("conv back",time.time()-t)
        t = time.time()
        self.update(m)

        print("update back",time.time()-t)
        self.output_maps =[]
        #输出大小为input_maps,去掉padding


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
    batch_size=11, shuffle=False, num_workers=2, drop_last=True
)

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


model = Conv(1,5,3,1,1)






