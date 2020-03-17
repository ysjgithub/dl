import  numpy as np


def gen_data(nums,dim):
    dataSet = []
    for i in range(50):
        p = [0,0]
        while p[0]==p[1]:
            p = [np.random.randint(0,10) for _ in range(dim)]
        dataSet.append(( np.array(p),0 if p[0]-p[1]<0 else 1 ))
    return dataSet


class logistic(object):

    def __init__(self,dataSet):
        self.weight = np.random.randn(1,2)
        self.bias = 0
        self.lenrning_rate = 0.1
        self.dataSet = dataSet


    def SGD(self):
        dataSet = self.dataSet
        # 随即梯度下降选择一个
        i = np.random.randint(-1,len(dataSet))
        x,y = dataSet[i]
        x = x.reshape(2,1)
        z = np.exp(np.dot(self.weight,x)+self.bias)

        dw = -x*y + z*x/((1+z))
        db = -y + z/((1+z))

        print(dw,db,self.lenrning_rate)


        self.weight = self.weight-self.lenrning_rate * dw.T
        self.bias = self.bias-self.lenrning_rate * db

    def accuracy(self):
        correct_total = 0
        for x,y in self.dataSet:
            z = np.exp(np.dot(self.weight,x)+self.bias)
            y_pre = z/(1+z)
            if (y_pre-0.5)*(y-0.5)>=0:
                correct_total +=1
        print(correct_total/len(self.dataSet))
        return correct_total/len(self.dataSet)

    def train(self):
        ite = 2
        while True:
            self.lenrning_rate = 0.001
            self.SGD()
            if self.accuracy() >= 1:
                break
            ite+=1


dataSet = gen_data(6,2)
model = logistic(dataSet)
model.train()