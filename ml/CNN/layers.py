
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
        self.gradient =None
        self.input_maps = None
        self.delta = None
    def forword(self,x):
        # 输入为 128 * 3 * 10 *10
        if self.padding != 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        self.input_maps = x
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
                        featuremap[int(i/self.stride),int(j/self.stride)] = np.sum(img[:,i:i+self.kernel,j:j+self.kernel]*core)
                features.append(featuremap)
            self.output_map.append(np.array(features))
        self.output_map = np.array(self.output_map)
        #输出为128 * output_channel * 10 *10

    def update(self,y):
        for idc in range(len(self.core)):
            #更新一个核,core 3*3*3
            core = self.core[idc,:,:,:]
            # 这一批所有的梯度加起来 3*3*3
            deltacore = np.zeros(core.shape)

            for idf in range(len(y)):
                # features 10*10*10
                features = y[idf,:,:,:]
                for feature in features:
                    # feature 10 * 10
                    for i in range(3):
                        for j in range(3):
                            for seq in range(self.input_channel):
                                deltacore[seq,i,j] += np.sum(self.input_maps[idf,seq,i:i + 10, j:j + 10] * feature)
            deltacore=deltacore/len(y)
            self.core[idc,:,:,:]-=deltacore*0.3

    def backword(self,y):
        # 输入为 output_maps，更新core的
        # 输入为 128 * 10 * 10 *10
        self.update(y)
        mini_batch, c, w, h = y.shape

        if self.padding != 0:
            y = np.pad(y, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        self.gradient = np.zeros(self.input_maps.shape)
        # 图片
        for idx in range(mini_batch):
            img = y[idx,:,:,:] #每一张图片的特征图
            # 遍历特征图，找到每个特征图产生误差的地方
            for idf in range(len(img)):
                cores = self.core[idf,:,:,:]
                featuremap = np.zeros((3,10, 10))
                for idc in range(len(cores)):
                    newcore = cores[idc,:,:].T[::-1].T[::-1]
                    for i in range(0, w + 2 * self.padding - self.kernel, self.stride):
                        for j in range(0, h + 2 * self.padding - self.kernel, self.stride):
                            featuremap[idc,int(i / self.stride), int(j / self.stride)] = np.sum(img[:, i:i + self.kernel, j:j + self.kernel] * newcore)
                    # 得到特征图加到误差点上
                self.gradient[idx,:,self.padding:-self.padding,self.padding:-self.padding]+=featuremap
        #输出大小为input_maps,去掉padding

class activition(object):
    def __init__(self):
        self.input_maps = []
        self.output_maps = []
        self.gradient =None
    def forword(self,x):
        self.input_maps = x
        self.output_maps = np.maximum(0,x)

    def prime(self,x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    def backword(self,y):
        self.gradient = self.prime(y)



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
        print("flattan",x.shape)
        self.input_maps = x
        for maps in x:
            self.output_maps.append(maps.flatten())
        self.output_maps = np.array(self.output_maps)
        #输出是 128 * 250
    def backword(self,y):
        print("flattan-back",y.shape)
        self.gradient = y.reshape(self.input_maps.shape)


class FC(object):
    def __init__(self,input_nums,output_nums,bias=False):
        self.biases = np.random.rand((1,output_nums)) if bias else np.zeros((1,output_nums))
        self.weights = np.random.randn(input_nums, output_nums) # 250 * 10
        self.output_maps =None
        self.gradient =None
    def forword(self,x):
        #128 * 250
        self.input_maps= x
        self.output_maps = np.dot(x,self.weights)

    def update(self,y):
        # input 128 *250 y 128 * 10
        self.weights-=0.3*np.dot(self.input_maps.T,y)

    def backword(self,y):
        # y 128 * 30
        self.update(y)
        self.gradient = np.dot(y,self.weights.T)

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

image = np.tile(np.arange(0,1,0.1),(128,3,10,1))




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
fc1= FC(250,10)
fc1.forword(f1.output_maps)
print("fc1",fc1.output_maps.shape)
sm=softMax()
sm.forword(fc1.output_maps)
print(sm.output_maps.shape)
r = sm.output_maps - 1
print(r.shape)
sm.backword(r)
print("sm.gradient",sm.gradient.shape)
fc1.backword(sm.gradient)
f1.backword(fc1.gradient)
pool1.backword(f1.gradient)
print(pool1.gradient.shape)
model.backword(pool1.gradient)


