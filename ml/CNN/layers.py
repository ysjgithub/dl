
import numpy as np

class Conv(object):
    def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
        self.input_channel=input_channel
        self.output_channel = output_channel
        self.kernel=kernel
        self.stride=stride
        self.padding=padding
        self.output_map=[]
        self.core = np.array([np.random.randn(3,kernel,kernel) for _ in range(output_channel)])

    def forword(self,x):
        # 输入为 128 * 3 * 10 *10
        mini_batch,c,w,h = x.shape
        if self.padding != 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        neww,newh = int(w-self.kernel+2*self.padding)/self.stride+1,(h-self.kernel+2*self.padding)/self.stride+1
        #图片
        for img in x:
            #卷积核
            features = []
            for core in self.core:
                #卷积核对应一张特征图
                featuremap = np.zeros((int(neww),int(newh)))
                #遍历一张维度
                for i in range(0,w+2*self.padding-self.kernel,self.stride):
                    for j in range(0,h+2*self.padding-self.kernel,self.stride):
                        featuremap[int(i/self.stride),int(j/self.stride)] = np.sum(img[:,i:i+self.kernel,j:j+self.kernel]*core)
                features.append(featuremap)
            self.output_map.append(np.array(features))
        self.output_map = np.array(self.output_map)
        #输出为128 * output_channel * 10 *10

    def backword(self,y):
        # 输入为 output_maps，更新core的

        pass
        #输出为input_maps

class activition(object):
    def __init__(self):
        self.input_maps = []
        self.output_maps = []
    def forword(self,x):
        self.input_maps = x
        self.input_maps = np.maximum(0,x)

    def prime(self,x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    def backword(self,y):
        deltax = self.prime(y)



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
        self.record_maps=np.zeros(x.shape)
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
        self.output_maps = np.array(self.output_maps)
        #输出大小为 128 * 10 * 5 * 5

    def findmax(self,rect):
        w,h = rect.shape
        x,y,val = 0,0,rect[0,0]
        for i in range(w):
            for j in range(h):
                if rect[i,j]>val:
                    x,y,val=i,j,rect[i,j]

        return x,y,val

    def backword(self,y):
        #input 尺寸为 10 * 5 * 5,recordmap尺寸为128 *10*5*5
        y = np.repeat(np.repeat(y,self.stride,axis=2),self.stride,axis=3)
        self.gradient = y*self.record_maps


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
        self.gradient = y.reshape(self.input_maps.shape)


class softMax(object):
    def __init__(self):
        self.input_maps = None
        self.output_maps = None
        self.gradient = None
    def forword(self,x):
        # 输入为 128 * 10
        self.input_maps = x
        sum = np.sum(np.exp(x))
        self.output_maps = x/sum
    def backword(self,y):
        # 一个梯度，对应上一层的数据，输入为128 * 10
        sum = np.sum(np.exp(self.input_maps))
        self.gradient = y * (sum-y)/sum**2
        # 输出为 128*10
        pass

image = np.array(
    [[[[_ for _ in range(10)]]*10]*3]*128
)

#
model = Conv(3,10,3,1,1)
model.forword(image)
print(model.output_map.shape)
pool1 = MaxPooling(2,2)
pool1.forword(model.output_map)
print(pool1.output_maps.shape)
f1 = Flattan()
f1.forword(pool1.output_maps)
print(f1.output_maps.shape)
sm=softMax()
sm.forword(f1.output_maps)
print(sm.output_maps.shape)
r = sm.output_maps - 1
print(r.shape)
sm.backword(r)
print(sm.gradient.shape)
f1.backword(sm.gradient)
pool1.backword(f1.gradient)


