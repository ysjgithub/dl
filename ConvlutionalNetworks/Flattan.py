
import numpy as np

class Flattan(object):
    def __init__(self):
        self.input_maps = []
        self.output_maps = []
        self.gradient= None
    def forword(self,x):
        #input 是 128 *10* 5 *5
        self.output_maps = []
        self.input_maps = x
        for maps in x:
            self.output_maps.append(maps.flatten())
        self.output_maps = np.array(self.output_maps)
        #输出是 128 * 250
    def backword(self,y):
        # t = time.time()
        self.gradient = y.reshape(self.input_maps.shape)

