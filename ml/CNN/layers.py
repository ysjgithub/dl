import time

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Conv(object):
    def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=0):
        self.input_channel=input_channel
        self.output_channel = output_channel
        self.kernel=kernel
        self.stride=stride
        self.padding=padding
        self.output_maps=[]
        self.core = np.array([np.random.randn(input_channel,kernel,kernel)/np.sqrt(input_channel*kernel*kernel/2) for _ in range(output_channel)])
        self.gradient =None
        self.input_maps = None
        self.delta = None
    def forword(self,x):
        # 输入为 128 * 3 * 10 *10
        self.input_maps = x
        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        self.delta = np.zeros(x.shape)
        mini_batch,c,w,h = x.shape
        neww,newh = int((w-self.kernel)/self.stride+1),int((h-self.kernel)/self.stride+1)

        #图片
        for img in x:
            #卷积核
            features = []
            for core in self.core:
                #卷积核对应一张特征图
                featuremap = np.zeros((int(neww),int(newh)))
                #遍历一张维度
                for i in range(0,neww):
                    for j in range(0,newh):
                        featuremap[int(i/self.stride),int(j/self.stride)] = np.mean(img[:,i:i+self.kernel,j:j+self.kernel] * core)
                features.append(featuremap)
            self.output_maps.append(np.array(features))
        self.output_maps = np.array(self.output_maps)
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

            for idi in range(mini_batch):
                # 一个mini_batch的梯度加起来,feature 7 * 7
                # 扩维
                feature = np.expand_dims(y[idi,idc,:,:],axis=0)
                feature = np.repeat(feature,self.input_channel,axis=0)
                # 影响的对象只有一张图,features 7 * 7

                for i in range(self.kernel):
                    for j in range(self.kernel):
                        zz= x[idi,:,i:i + w, j:j + h] * feature
                        deltacore[:,i,j] += np.mean(np.mean(zz,axis=1),axis=1)

            deltacore=deltacore/len(y)
            self.core[idc,:,:,:]-=deltacore*0.1

    def backword(self,y):
        # 输入为 output_maps，更新core的
        # 输入为 128 * 10 * 10 *10
        # 计算输入点产生的误差
        t = time.time()

        self.update(y)
        mini_batch, c, w, h = y.shape
        mini_batch, ic, iw, ih = self.input_maps.shape

        y = np.pad(y, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        self.gradient = np.zeros(self.input_maps.shape)
        # 图片
        for idx in range(mini_batch):
            img = y[idx,:,:,:] #每一张图片的特征图 64 *7*7
            # 遍历特征图，找到每个特征图产生误差的地方
            for idf in range(len(img)):
                cores = self.core[idf,:,:,:] # 32 * 3 * 3
                featuremap = np.zeros((ic,iw,ih)) # 32 * 7 * 7
                for idc in range(len(cores)):
                    newcore = cores[idc,:,:].T[::-1].T[::-1] #3*3
                    for i in range(0, w + 2 * self.padding - self.kernel, self.stride):
                        for j in range(0, h + 2 * self.padding - self.kernel, self.stride):
                            featuremap[idc,int(i / self.stride), int(j / self.stride)] = np.mean(img[:, i:i + self.kernel, j:j + self.kernel] * newcore)
                    # 得到特征图加到误差点上
                self.gradient[idx,:,:,:]+=featuremap
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
        print (self.output_maps.shape)
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
        Conv(1,8,3,1,1),
        activition(),
        MaxPooling(2,2),
        Conv(8,16,3,1,1),
        activition(),
        MaxPooling(2,2),
        # Conv(16,32,3,1,1),
        # activition(),
        # MaxPooling(2, 2),
        Flattan(),
        FC(784,32),
        activition(),
        FC(32,10),
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


for i in range(20):
    for x,y in train_loader:
        x = x.numpy()
        y = one_hot(np.array(y.numpy()),10)
        s = model.forword(x)
        r = MSE(s,y)
        print("result",np.argmax(y,axis=1),np.argmax(s,axis=1))
        model.backword(r)






