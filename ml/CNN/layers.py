import time

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Conv(object):
    def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
        self.input_channel=input_channel
        self.output_channel = output_channel
        self.kernel=kernel
        self.stride=stride
        self.padding=padding
        self.output_maps=[]
        self.core = np.array([np.random.randn(input_channel,kernel,kernel) for _ in range(output_channel)])
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

            for i in range(neww):
                for j in range(newh):
                    newfeaturemap = np.sum(x[:,:,i:i+self.kernel,j:j+self.kernel] * core,axis=(1,2,3)).reshape((1,mini_batch))
                    featuremap[:,i,j] = newfeaturemap

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
                    deltacore[:,i,j] += np.sum(zz,axis=(0,2,3))/mini_batch
            self.core[idc,:,:,:]-=deltacore*0.005

    def backword(self,m):
        # 输入为 output_maps，更新core的
        # 输入为 20 * 8 * 10 *10
        # 计算输入点产生的误差
        y=m
        mini_batch, ic, iw, ih = self.input_maps.shape
        y = np.pad(y, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        self.gradient = np.zeros(self.input_maps.shape)
        newcore = np.zeros((self.input_channel,self.output_channel,self.kernel,self.kernel))

        for i in range(self.input_channel):
            for j in range(self.output_channel):
                newcore[i,j,:,:] = self.core[j,i,:,:].T[::-1].T[::-1]

        for idc in range(self.input_channel):
            core = newcore[idc,:,:,:]

            # 卷积核对应mini_batch张特征图
            featuremap = np.zeros((mini_batch, iw, ih))

            for i in range(iw):
                for j in range(ih):
                    newfeaturemap = np.sum(y[:, :, i:i + self.kernel, j:j + self.kernel] * core,axis=(1, 2, 3)).reshape((1, mini_batch))
                    featuremap[:, int(i / self.stride), int(j / self.stride)] = newfeaturemap

            self.gradient[:, idc, :, :] = featuremap

        self.update(m)
        self.output_maps =[]
        #输出大小为input_maps,去掉padding





# 池化层的输入是feature maps，正向传播时记录下最大点的位置
class MaxPooling(object):
    def __init__(self,kernel,stride):
        self.stride = stride
        self.kernel = kernel
        self.input_maps = None
        self.record_maps =[]
        self.output_maps = []
        self.gradient = None
    def forword(self,x):
        # 输入大小为 128 * 10 *10 *10
        self.record_maps=[]
        self.input_maps = x

        for maps in x:
            n_features,w,h=maps.shape
            neww,newh = w//self.stride,h//self.stride
            #记录最大点的特征图
            maps_record_map = [] #图片级的记录map
            new_features = []  #新的特征图
            for feature in maps:
                # 遍历行
                feature_record_map = np.zeros(feature.shape)#特征记录图
                new_feature_map = np.zeros((neww, newh))#新的特征图
                for i in range(0,w-self.stride+1,self.stride):
                    # 遍历列
                    for j in range(0,h-self.stride+1,self.stride):
                        # 池化区
                        rect = feature[i:i+self.stride,j:j+self.stride]
                        # 最大点的坐标和值
                        x,y,val = self.findmax(rect)
                        feature_record_map[i+x,j+y] =1
                        new_feature_map[int(i // self.stride), int(j // self.stride)] = val
                new_features.append(new_feature_map)
                maps_record_map.append(feature_record_map)
            self.output_maps.append(new_features)
            self.record_maps.append(maps_record_map)
        self.output_maps = np.array(self.output_maps)
        self.record_maps = np.array(self.record_maps)

    def findmax(self,rect):
        w,h = rect.shape
        x,y,val = 0,0,rect[0,0]
        for i in range(w):
            for j in range(h):
                if rect[i,j]>val:
                    x,y,val=i,j,rect[i,j]

        return x,y,val

    def backword(self,y):
        y = np.repeat(np.repeat(y,self.stride,axis=2),self.stride,axis=3)
        resshape = np.array(self.input_maps.shape) - np.array(y.shape)
        y = np.pad(y,((0,resshape[0]),(0,resshape[1]),(0,resshape[2]),(0,resshape[3])),'constant')
        self.gradient = y*self.record_maps
        self.output_maps = []


class Flattan(object):
    def __init__(self):
        self.input_maps = []
        self.output_maps = []
        self.gradient= None
    def forword(self,x):
        #input 是 128 *10* 5 *5
        self.input_maps = x
        for maps in x:
            self.output_maps.append(maps.flatten())
        self.output_maps = np.array(self.output_maps)
        #输出是 128 * 250
    def backword(self,y):
        # t = time.time()
        self.gradient = y.reshape(self.input_maps.shape)
        self.output_maps = []



class FC(object):
    def __init__(self,input_nums,output_nums,bias=False):
        self.weights = np.random.randn(input_nums, output_nums)/np.sqrt(input_nums / 2) # 250 * 10
        self.output_maps =None
        self.gradient =None

    def forword(self,x):
        self.input_maps= x
        self.output_maps = np.dot(x,self.weights)

    def update(self,y):
        mini_batch,output_nums = y.shape
        self.weights-=0.1*np.dot(self.input_maps.T,y)/mini_batch

    def backword(self,y):
        # 11 * 10
        self.gradient = np.dot(y,self.weights.T)
        self.update(y)
        # 11 * 32


class activition(object):
    def __init__(self):
        self.input_maps = []
        self.output_maps = []
        self.gradient =None
    def forword(self,x):
        self.input_maps = x
        self.output_maps = np.maximum(0,x)
    def backword(self,y):
        y[self.input_maps <= 0] = 0
        self.gradient = y



class softMax(object):
    def __init__(self):
        self.input_maps = None
        self.output_maps = None
        self.gradient = None
    def forword(self,x):
        # 输入为 128 * 10
        self.input_maps = x
        x = np.exp(x-np.max(x,axis=1,keepdims=True))
        self.output_maps = x/np.sum(x,axis=1,keepdims=True)
    def backword(self,y):
        self.gradient = y
        # print("softmax",y)




class Model(object):
    def __init__(self,seq):
        self.sequential = seq
    def forword(self,x):
        o = x
        for i in range(len(self.sequential)):
            l=self.sequential[i]
            l.forword(o)
            o=l.output_maps
        return o

    def backword(self,y):
        g = y
        for i in range(len(self.sequential)-1,-1,-1):
            l = self.sequential[i]
            l.backword(g)
            g = l.gradient


def MSE(y,t):
    # y,t 都是 10*10
    return y-t
    # 输出是 10 *10

model = Model([
        Conv(1,32,3,1,1),
        activition(),
        MaxPooling(2,2),
        Conv(32,64,3,1,1),
        activition(),
        MaxPooling(2,2),
        # Conv(64,128,3,1,1),
        # activition(),
        # MaxPooling(2, 2),
        # Conv(128,256,3,1,1),
        # activition(),
        # MaxPooling(2, 2),
        Flattan(),
        FC(3136,96),
        activition(),
        FC(96,10),
        activition(),
        softMax()
    ])



transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_loader = DataLoader(
    datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=20,
    shuffle=True, num_workers=2, drop_last=True
)

test_loader = DataLoader(
    datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
    batch_size=11, shuffle=False, num_workers=2, drop_last=True
)

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

iter = 0
for i in range(2):
    for x,y in train_loader:
        iter+=1
        x = x.numpy()
        y = one_hot(np.array(y.numpy()),10)
        s = model.forword(x)
        r = MSE(s,y)
        print(iter,np.argmax(y,axis=1),np.argmax(s,axis=1))
        print(np.sum(np.argmax(y,axis=1)==np.argmax(s,axis=1)))
        model.backword(r)






