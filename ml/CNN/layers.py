
import numpy as np


class Conv(object):
    def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
        self.input_channel=input_channel
        self.output_channel = output_channel
        self.kernel=kernel
        self.stride=stride
        self.padding=padding
        self.feature_map=[]
        self.core = np.array([np.random.randn(3,kernel,kernel) for _ in range(output_channel)])
    def forword(self,x):
        mini_batch,c,w,h = x.shape
        neww,newh = int(w-self.kernel+2*self.padding)/self.stride,(h-self.kernel+2*self.padding)/self.stride
        print(neww,newh)
        #图片
        for img in x:
            #卷积核
            if self.padding != 0:
                img = np.pad(img, ((0,0),(self.padding,self.padding),(self.padding,self.padding)), 'constant')
            print(img.shape)
            print(img)
            for core in self.core:
                #卷积核对应一张特征图
                newfeaturemap = np.zeros((int(neww),int(newh)))
                #遍历一张维度
                for i in range(0,w+2*self.padding-self.kernel,self.stride):
                    for j in range(0,h+2*self.padding-self.kernel,self.stride):

                        # print(img)
                        print(img[:,i:i+self.kernel,j:j+self.kernel])
                        newfeaturemap[int(i/self.stride),int(j/self.stride)] = np.sum(img[:,i:i+self.kernel,j:j+self.kernel]*core)

            self.feature_map.append(newfeaturemap)
        self.feature_map = np.array(self.feature_map)


    def backword(self,y):
        # 已知y.shape= （2，2）

        pass

class activition(object):
    def __init__(self):
        self.inputMaps = []
        self.outputMaps = []
    def forword(self,x):
        self.inputMaps = x
        self.outputMaps = np.relu(x)
    def backword(self,y):
        assert y.shape


# 池化层的输入是feature maps，正向传播时记录下最大点的位置
class MaxPlooing(object):
    def __init__(self,kernel,stride):
        self.stride = stride
        self.kernel = kernel
        self.featuremaps = []
        self.recordMaps =[]
    def forword(self,x):
        for map in x:
            w,h=map.shape
            neww,newh = w//self.stride,h//self.stride
            # 前馈的特征图
            newfeaturemap = np.zeros((neww,newh))
            #记录最大点的特征图
            recordmap = np.zeros((w,h))
            # 遍历行
            for i in range(0,w-self.stride+1,self.stride):
                # 遍历列
                for j in range(0,h-self.stride+1,self.stride):
                    # 池化区
                    rect = map[i:i+self.stride,j:j+self.stride]
                    # 最大点的坐标和值
                    x,y,val = self.findmax(rect)
                    recordmap[i+x,j+y] = val
                    newfeaturemap[int(i//self.stride),int(j//self.stride)] = val
            self.recordMaps.append(recordmap)
            self.featuremaps.append(newfeaturemap)

    def findmax(self,rect):
        w,h = rect.shape
        x,y,val = 0,0,rect[0,0]
        for i in range(w):
            for j in range(h):
                if rect[i,j]>val:
                    x,y,val=i,j,rect[i,j]

        return x,y,val


    def backword(self,y):
        pass


class Flattan(object):
    def __init__(self):
        self.inputMaps = []
        self.outputMaps = []
    def forword(self,x):
        for maps in x:
            self.outputMaps.append(maps.flatten())

    def backword(self):
        pass

class softMax(object):
    def __init__(self):
        self.inputMaps = None
        self.outputMaps = None
    def forword(self,x):
        self.inputMaps = x
        sum = np.sum(np.exp(x))
        self.outputMaps = x/sum
    def backword(self,y):
        # 一个梯度，对应上一层的数据
        pass

image = np.array(
    [[[_ for _ in range(10)]]*10]*3
)
print(image.shape)
model = MaxPlooing(2,2)
model.forword(image)
print(model.recordMaps)

# print(image.shape)
#
# model = Conv(3,10,3,1,1)
# model.forword(image)


