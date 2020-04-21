
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
            self.feature_map.append(np.array(features))
        self.feature_map = np.array(self.feature_map)
        #输出为128 * output_channel * 10 *10

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
class MaxPooling(object):
    def __init__(self,kernel,stride):
        self.stride = stride
        self.kernel = kernel
        self.input_maps = None
        self.record_maps =[]
        self.output_maps = []
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
                        feature_record_map[i+x,j+y] = val
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
    [[[[_ for _ in range(10)]]*10]*3]*128
)

#
model = Conv(3,10,3,1,1)
model.forword(image)
print(model.feature_map.shape)
pool1 = MaxPooling(2,2)
pool1.forword(model.feature_map)
print(pool1.output_maps.shape)


