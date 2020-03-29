import  numpy as np

def gen_data(dim=3):
    points,labels = [],[]
    for i in range(50):
        p = [0,0]
        while p[0]==p[1]:
            p = [np.random.randint(0,10) for _ in range(dim)]
        points.append(p)
        labels.append(np.sign(p[0]-p[1]))
    return np.array(points),np.array(labels)


def init_para(dim=2):
    return np.ones((1,dim))


class LDA(object):
    def __init__(self,data,labels,dim=2):
        self.weights = np.random.randn(dim,1)
        self.X = data
        self.Y = labels

    def solve(self):
        mean = np.mean(self.X,axis=0)


        print(mean)

def vision(points,w,b=0):
    x = np.linspace(0.1,10.1,500)

    y = (-b-x*w[0,0])/w[0,1]
    for i in range(50):
        if labels[i]>0:
            plt.scatter(points.T[0,i], points.T[1,i], marker='^',c='green')
        else:
            plt.scatter(points.T[0,i], points.T[1,i], marker='o',c='red')
    plt.plot(x, y, '-r', label='wx+b=0')
    plt.axis([-1,10, -1, 10])
    plt.draw()
    plt.pause(0.2)
    plt.clf()


import matplotlib.pyplot as plt
plt.figure()
plt.ion()
points,labels=gen_data()
lda = LDA(points,labels,3)
lda.solve()
w_hat = init_para()
vision(points,w_hat,b=-10)




