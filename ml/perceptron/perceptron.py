import  numpy as np

def gen_data(dim=2):
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

def solve(points,w,labels,eta=1):
    b = 0
    while 1:
        predicty = np.dot(w_hat,points.T)+b
        err = predicty*labels
        print(np.sum(np.sign(err)))
        if np.sum(np.sign(err))==50:
            return w_hat,b
        # 判断是否全部判别成功
        if np.sum(np.sign(err))<50:
            i = np.random.randint(0,50)
            # 找到错误分类的点
            while np.sign(err[0,i])>0:
                i=np.random.randint(0,50)
            w+=eta*points[i]*labels[i]
            b += eta*labels[i]
        vision(points,w,b)


def vision(points,w,b=0):
    x = np.linspace(0.1,10.1,500)

    y = (-b-x*w[0,0])/w[0,1]
    for i in range(50):
        if labels[i]>0:
            plt.scatter(points.T[0,i], points.T[1,i], marker='^',c='green')
        else:
            plt.scatter(points.T[0,i], points.T[1,i], marker='o',c='red')
    plt.plot(x, y, '-r', label='wx+b=0')
    plt.axis([-1,-10, -1, 10])
    plt.show()

    import time
    time.sleep(0.8)

import matplotlib.pyplot as plt
points,labels=gen_data()
w_hat = init_para()
vision(points,w_hat,b=0)
solution,b = solve(points,w_hat,labels)




