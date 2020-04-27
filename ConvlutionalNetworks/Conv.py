
import numpy as np

class Conv(object):
    def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
        self.input_channel=input_channel
        self.output_channel = output_channel
        self.kernel=kernel
        self.stride=stride
        self.padding=padding
        self.output_maps=[]
        self.core = np.random.randn(output_channel,input_channel,kernel,kernel)*np.sqrt(2/(input_channel))
        self.gradient =None
        self.input_maps = None
    def forword(self,x):
        # 输入为 128 * 3 * 10 *10
        self.output_maps =[]
        self.input_maps = x
        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        mini_batch,c,w,h = x.shape
        neww,newh = int((w-self.kernel)/self.stride+1),int((h-self.kernel)/self.stride+1)

        self.output_maps = np.zeros((mini_batch,self.output_channel,neww,newh))

        x = x.reshape(mini_batch,1,self.input_channel,w,h)
        x = np.repeat(x,self.output_channel,axis=1)

        for i in range(neww):
            for j in range(newh):
                self.output_maps[:,:,i,j] = np.sum(x[:,:,:,i:i+self.kernel,j:j+self.kernel] * self.core,axis=(2,3,4)).reshape((mini_batch,self.output_channel))
        #输出为128 * output_channel * 10 *10

    def update(self,y):
        # 更新权值要用到所有参与卷积的点，需要使用padding后的图像
        mini_batch,output_channel,w,h = y.shape
        x = np.pad(self.input_maps, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        mini_batch,input_channel,iw,ih  = x.shape
        # 16 * 9 * 9
        x=x.reshape(mini_batch,1,input_channel,iw,ih).repeat(self.output_channel,axis=1)
        y=y.reshape(mini_batch,output_channel,1,w,h).repeat(self.input_channel,axis=2)

        deltacore=np.zeros(self.core.shape)

        for i in range(self.kernel):
            for j in range(self.kernel):
                zz= x[:,:,:,i:i + w, j:j + h] * y
                deltacore[:,:,i,j] += np.sum(zz,axis=(0,3,4)).reshape((output_channel,input_channel))/mini_batch
        print(deltacore.mean())
        self.core-=deltacore*0.005

    def backword(self,m):
        # 输入为 output_maps，更新core的
        # 输入为 20 * 8 * 10 *10
        # 计算输入点产生的误差
        y=m
        mini_batch, ic, iw, ih = self.input_maps.shape
        y = np.pad(y, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        self.gradient = np.zeros(self.input_maps.shape)

        newcore = np.zeros((self.output_channel,self.input_channel,self.kernel,self.kernel))
        for i in range(self.output_channel):
            for j in range(self.input_channel):
                newcore[i,j,:,:] = self.core[i,j,:,:].T[::-1].T[::-1]

        mini_batch,output_channel,w,h = y.shape
        y = y.reshape((mini_batch,self.output_channel,1,w,h)).repeat(self.input_channel,axis=2)

        for i in range(iw):
            for j in range(ih):
                newfeaturemap = np.sum(y[:, :,:, i:i + self.kernel, j:j + self.kernel] * newcore,axis=(1, 3, 4)).reshape((mini_batch, self.input_channel))
                self.gradient[:,:,i,j] = newfeaturemap

        self.update(m)
        #输出大小为input_maps,去掉padding


