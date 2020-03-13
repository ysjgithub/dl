import numpy as np
import pandas as pd

def gen_data():
    points,labels = [],[]
    tags = ['green','blue','red','yellow']
    x = [np.random.rand()*100,np.random.rand()*10]
    for i in  range(60):
        points.append([np.random.rand()*100,np.random.rand()*100])
        labels.append(tags[np.random.randint(0,4)])
    return x,np.array(points),np.array(labels)

def visualization(points,labels):
    import matplotlib.pyplot as plt

    for i in range(len(points)):
        plt.scatter(points.T[0, i], points.T[1, i], marker='^', c=labels[i])
    plt.axis([0, 100, 0, 100])
    plt.show()

def visualization1(points,labels,x,L):
    import matplotlib.pyplot as plt

    for i in range(len(points)):
        plt.scatter(points.T[0, i], points.T[1, i], marker='^', c=labels[i])
    for e in L:
        plt.plot([x[0],e[0]],[x[1],e[1]],label=distance(x,e))
    plt.axis([0, 100, 0, 100])
    plt.show()


class kdtree(object):
    def __init__(self,points,dim):
        self.left = None        #左节点
        self.right = None       #右节点
        self.points = points    #此节点分开的数据
        self.dim = dim      #记录此节点分开的维度
        self.feature = [0,0]
        if len(self.points)==1:
            self.feature = self.points.values[0]
        else:
            self.calcMid()
            self.departtree()

    #得到中位数
    def calcMid(self):
        self.points.sort_values(self.dim, inplace=True,ignore_index=True)
        l = self.points.shape[0]
        self.feature = self.points.values[l//2]
        self.points = self.points.drop(index=l//2,axis=0)

    # 构造下一级节点
    def departtree(self):
        # 还有多的节点
        left = []
        right = []
        for e in self.points.values:
            if e[self.dim] < self.feature[self.dim]:
                left.append(e)
            else:
                right.append(e)
        if len(left):
            self.left = kdtree(pd.DataFrame(left),(self.dim+1)%self.points.shape[1])
        if len(right):
            self.right = kdtree(pd.DataFrame(right),(self.dim+1)%self.points.shape[1])
        return

def distance(x,y):
    d = 0
    for i in range(len(x)):
        d+=(x[i]-y[i])**2
    return d


def argmaxdis(x,L):
    dis = []
    for e in L:
        dis.append(distance(x,e))
    return np.argmax(dis),np.max(dis)

def insertL(x,p,L,maxl):
    if len(L)<maxl:
        L.append(p)
    else:
        i,maxdis = argmaxdis(x,L)
        if distance(x,p)<maxdis:
            L[i] = p
    return


def findNearestK(x,root,L,maxl):
    if root:
        r = [root.left,root.right]
        tag = -1

        if x[root.dim] <= root.feature[root.dim]:
            tag = 0
            findNearestK(x,root.left,L,maxl)
        else:
            tag = 1
            findNearestK(x,root.right,L,maxl)

        #尝试将root.feature 插入L

        insertL(x,root.feature,L,maxl)

        if len(L)<maxl:
            #成功插入,没插满，找另一边
            findNearestK(x,r[1-tag],L,maxl)
        elif len(L) == maxl:
            #插满了,判断分界线的远近
            i ,maxdis = argmaxdis(x,L)
            if abs(x[root.dim] - root.feature[root.dim]) < maxdis:
                #分界线必最远的点近
                findNearestK(x,r[1-tag],L,maxl)

    return

import time

t = time.time()

x,points,labels = gen_data()

print("data has been generated, used ",time.time()-t)
t = time.time()

points = pd.DataFrame(points)
# visualization(points.values,labels)
tree = kdtree(points,0)
print("kd tree has been built,used ",time.time()-t)
t = time.time()

L = []
findNearestK(x,tree,L,3)
print("find best N nearest,used ",time.time()-t)

visualization1(points.values,labels,x,L)

ds = []
for e in points.values:
    ds.append(distance(x,e))
r = []
for i in range(3):
    r.append(points.values[np.argmin(ds)])
    ds[np.argmin(ds)] = 1000000
print("enumerate result",pd.DataFrame(r).sort_values(0))
print("KNN result",pd.DataFrame(L).sort_values(0))
