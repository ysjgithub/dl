
import numpy as np

class full_connect(object):
    def __init__(self,input_nums,output_nums,bias=False):
        self.weights = np.random.randn(input_nums, output_nums)*np.sqrt(2/input_nums) # 250 * 10
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
