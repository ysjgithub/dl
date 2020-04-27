class Model(object):
    def __init__(self,seq):
        self.sequential = seq
    def forword(self,x):
        o = x
        for i in range(len(self.sequential)):
            l=self.sequential[i]
            l.forword(o)
            o=l.output_maps
        return o

    def backword(self,y):
        g = y
        for i in range(len(self.sequential)-1,-1,-1):
            l = self.sequential[i]
            l.backword(g)
            g = l.gradient
