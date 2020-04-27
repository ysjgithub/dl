import numpy as np


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

