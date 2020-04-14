import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist.load_data()
import matplotlib.pyplot as plt #plt 用于显示图片
import matplotlib.image as mpimg #mpimg 用于读取图片

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_pre(x):
    return sigmoid(x) * (1 - sigmoid(x))

def KL_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def train(trainData, weight, biases, beta, hat, eta):
    tn = trainData.shape[1]

    w1, w2 = weight[0], weight[1]
    b1, b2 = biases[0], biases[1]

    z1 = np.dot(w1, trainData)
    a1 = 0.99 * sigmoid(z1)
    z2 = np.dot(w2, a1)
    #     print(z1.shape)

    a2 = 0.99 * sigmoid(z2)

    rho_hat = np.sum(a1, axis=1) / tn
    rho = np.tile(hat, 196)

    print(rho_hat[0:10])
    KL = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (tn, 1)).transpose()

    d2 = (a2 - trainData) * 0.99 * sigmoid_pre(z2)
    db2 = np.sum(d2, axis=1) / tn
    dw2 = np.dot(d2, a1.transpose()) / tn + 0.003 * w2

    d1 = (np.dot(w2.transpose(), d2) + beta * KL) * 0.99 * sigmoid_pre(z1)
    db1 = np.sum(d1, axis=1) / tn
    dw1 = np.dot(d1, trainData.transpose()) / tn + 0.003 * w1

    weight[0] = w1 - eta * dw1
    weight[1] = w2 - eta * dw2
    biases[0] = b1 - eta * db1
    biases[1] = b2 - eta * db2
    #     print(weight,biases)
    return weight, biases


o,n  = 14,28
(trainImage, trainTarget), (testImage, testTarget) = mnist

initTrain = trainImage[0:1000].astype(np.float64).reshape(1000,784)*0.99/255

weight = [np.random.randn(x,y)/np.sqrt(y) for x,y in zip([196,784],[784,196])]
biases = [np.zeros(x) for x in [196,784]]

plt.figure(figsize=(10, 5))
for i in range(1000):
    weight,biases =  train(initTrain.transpose(), weight,biases, 3, 0.15,1)
a1 = sigmoid(np.dot(weight[0],initTrain[0]))
a2 = sigmoid(np.dot(weight[1],a1))
plt.subplot(1,2,1)
plt.imshow(initTrain[0].reshape(28,28))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(a2.reshape(28,28))
plt.axis('off')



def res(x):
    return x/ np.sqrt(np.sum(x * x))
plt.figure(figsize=(5, 5), dpi=150)
for m in range(o):
    for j in range(o):
        plt.subplot(o, o, o * m + j + 1)
        plt.imshow(res(weight[0][m * o + j].reshape(n, n)))
        plt.axis('off')
