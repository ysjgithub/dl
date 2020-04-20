
import numpy as np

class MyModule(object):
    def __init__(self):
        pass

    def forword(self):
        pass

    def backword(self):
        pass


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


    def backword(self):
        pass


class MaxPlooing(object):
    def __init__(self):
        pass

    def forword(self):
        pass

    def backword(self):
        pass


class Flattan(object):
    def __init__(self):
        pass
    def forweord(self):
        pass
    def backword(self):


image = np.array([
    [[[_ for _ in range(10)]]*10]*3
])
print(image.shape)

model = Conv(3,10,3,1,1)
model.forword(image)


