import numpy as np
import pandas as pd



# 第dim维对应的所有概率
class attrP(object):
    def __init__(self,dim,labels):
        self.attrs = set(dataSet[dim].values)
        self.dim = dim
        # print(self.attrs)
        self.P = [(attr,[[_,0,0] for _ in labels]) for attr in self.attrs]
        # print(self.P)

    def update(self,e):
        for attr,labels in self.P:
            if attr == e[self.dim]:
                for label in labels:
                    if label[0] == e[2]:
                        label[1]+=1
                        return
    def calcP(self,yn,_lambda):

        for attr,labels in self.P:
            for i in range(len(labels)):
                labels[i][2] = (labels[i][1]+_lambda)/(yn[i][1]+_lambda*len(self.attrs))

    def predict(self,x):
        for attr,labels in self.P:
            if attr == x[self.dim]:
                return labels


class Beyes(object):
    def __init__(self,dataSet,_lambda=0):
        super(Beyes).__init__()
        self._lambda = _lambda
        self.P = []
        self.ycalcP = []
        self.constructModel(dataSet)


    def constructModel(self,dataSet):

        ylabels = set(dataSet[2].values)

        self.ycalcP = [(_,n,(n+self._lambda)/(len(dataSet)+len(ylabels)*self._lambda) ) for _,n \
                       in zip(ylabels,dataSet[2].value_counts())]
        columns = dataSet.drop(2,axis=1).columns.values.tolist()
        for column in columns:
            self.P.append(attrP(column,ylabels))

        for e in dataSet.values:

            for j in range(len(columns)):
                self.P[j].update(e)

        for c in self.P:
            c.calcP(self.ycalcP,self._lambda)

        print(self.ycalcP)

    def predict(self,x):
        print(self.ycalcP)
        r = [p for label,n,p in self.ycalcP]
        print(r)
        for p in self.P:
            # 得到每一维数据的概率
            dimI_P = p.predict(x)
            print(dimI_P)
            # 乘以每个标签的概率
            for i in range(len(r)):
                r[i] *= dimI_P[i][2]
        print(r)



s,m,l = 's','m','l'
dataSet = [
    [1,s,-1],
    [1,m,-1],
    [1,m,1],
    [1,s,1],
    [1,s,-1],
    [2,s,-1],
    [2,m,-1],
    [2,m,1],
    [2,l,1],
    [2,l,1],
    [3,l,1],
    [3,m,1],
    [3,m,1],
    [3,l,1],
    [3,l,-1]
]

dataSet = pd.DataFrame(dataSet)
model = Beyes(dataSet,1)
model.predict([2,s])


