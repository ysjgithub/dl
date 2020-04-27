
import numpy as np

class relu(object):
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
