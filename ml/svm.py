import numpy as np
import  matplotlib.pyplot as plt
import random

def clip(_lambda, L, H):
    ''' 修剪lambda的值到L和H之间.
    '''
    if _lambda < L:
        return L
    elif _lambda > H:
        return H
    else:
        return _lambda
def select_j(i, m):
    ''' 在m中随机选择除了i之外剩余的数
    '''
    j=i
    while i==j:
        j=int(random.random()*m)
    return j


def get_w(_lambdas, Xs, Ys):
    ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    '''
    _lambdas, Xs, Ys = np.array(_lambdas).reshape(10,1), np.array(Xs).T, np.array(Ys).reshape(1,10).T
    _lambdasy = _lambdas*Ys
    w = np.dot(Xs,_lambdasy).T
    return w

def SMO(Xs, Ys, C, max_iter):
    '''
    :param Xs: 特征向量的值
    :param Ys: 所有的特征向量的标签
    :param C: 软间隔常数, 0 <= alphlambda_i <= C
    :param max_iter: 外层循环最大迭代次数
    '''
    Xs = np.array(Xs)
    m,n = Xs.shape
    Ys = np.array(Ys)
    # 初始化参数
    lambda_s = np.zeros(m)
    b = 0
    it = 0
    def f(x):
        '''
        :param x 一个2x1的数组，代表二维坐标:
        :return: 一个预测值
        '''
        x= np.array(x).reshape(2,1)
        return np.dot(get_w(lambda_s,Xs,Ys),x)+b

    while it < max_iter:
        pair_changed = 0
        for i in range(m):
            lambda_i, x_i, y_i = lambda_s[i], Xs[i], Ys[i]
            fx_i = f(x_i)
            E_i = fx_i - y_i

            j = select_j(i, m)

            lambda_j, x_j, y_j = lambda_s[j], Xs[j], Ys[j]
            fx_j = f(x_j)
            E_j = fx_j - y_j

            K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
            eta = K_ii + K_jj - 2*K_ij

            if eta <= 0:
                print('WARNING  eta <= 0')
                continue
            # 获取更新的alpha对
            lambda_i_old, lambda_j_old = lambda_i, lambda_j
            lambda_j_new = lambda_j_old + y_j*(E_i - E_j)/eta
            # 对lambda_j进行修剪
            if y_i != y_j:
                L = max(0, lambda_j_old - lambda_i_old)
                H = min(C, C + lambda_j_old - lambda_i_old)
            else:
                L = max(0, lambda_i_old + lambda_j_old - C)
                H = min(C, lambda_j_old + lambda_i_old)

            lambda_j_new = clip(lambda_j_new, L, H)
            lambda_i_new = lambda_i_old + y_i*y_j*(lambda_j_old - lambda_j_new)

            if abs(lambda_j_new - lambda_j_old) < 0.00001:
                print('WARNING  lambda_j not moving enough')
                continue
            lambda_s[i], lambda_s[j] = lambda_i_new, lambda_j_new
            # 更新阈值b

            b_i = -E_i - y_i*K_ii*(lambda_i_new - lambda_i_old) - y_j*K_ij*(lambda_j_new - lambda_j_old) + b
            b_j = -E_j - y_i*K_ij*(lambda_i_new - lambda_i_old) - y_j*K_jj*(lambda_j_new - lambda_j_old) + b

            if 0 < lambda_i_new < C:
                b = b_i
            elif 0 < lambda_j_new < C:
                b = b_j
            else:
                b = (b_i + b_j)/2
            pair_changed += 1
            print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(it, i, pair_changed))
        if pair_changed == 0:
            it += 1
        else:
            it = 0
        print('iteration number: {}'.format(it))
    return lambda_s, b

Xs, Ys = [[1,2],[2,3],[3,4],[4,5],[5,6],[1,1],[2,2],[3,3],[4,4],[5,5]],[-1,-1,-1,-1,-1,1,1,1,1,1]
lambda_s=np.zeros((10,1))

lambda_s, b = SMO(Xs, Ys, 10, 40)

# 绘制数据点
plt.plot(np.array(Xs).T[0], np.array(Xs).T[1], 'ro')
w = get_w(lambda_s,Xs,Ys)
print(w,b)
ax,ay=w[0][0],w[0][1]
b=b[0][0]
t0 = (-b-ax)/ay
t1 = (-b-5*ax)/ay
plt.plot([1,5],[t0,t1])
plt.show()