
import numpy as np

# 池化层的输入是feature maps，正向传播时记录下最大点的位置
class max_pooling(object):
    def __init__(self,kernel,stride):
        self.stride = stride
        self.kernel = kernel
        self.input_maps = None
        self.record_maps =[]
        self.output_maps = []
        self.gradient = None
    def forword(self,x):
        # 输入大小为 128 * 10 *10 *10
        self.output_maps = []
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
