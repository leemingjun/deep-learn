#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

# 这个机器学习的例子虽然简单，但是，麻雀虽小，五脏俱全。包含了参数调整、学习率设置等
# 之后的深度学习的图像识别、语言识别等基本上也是类似于这个程序的模板，只不过每个函数都
# 更加复杂而已
import numpy as np

# 生成样本数据。使之符合 y = weight * x + bias * x0 ，其中X0永远等于1
# numPoints : 样本数据的个数，默认是100个
# bias ： 偏置项
# weight : 权重


def generate_sample_data(numPoints=100, bias=26, weight=10):
    x = np.zeros(shape=(numPoints,  2))  # 矩阵 100 * 2
    y = np.zeros(shape=(numPoints))       # 矩阵 100 * 1，numpy也可以当做 1 * 100的矩阵
    # 基本的直线函数  y = x0 * b + x1 * w， 其中x0永远等于1
    for i in range(0, numPoints):
        x[i][0] = 1  # x0 用于等于 1
        x[i][1] = i  # x1 序列增长，1，2，3，4……
        # 根据直线函数，同时增加随机数，生成样本数据的目标标量，随机波动幅度为bias的一半
        y[i] = weight * x[i][1] + bias + np.random.randint(1, bias * 0.5)

    return x, y

# 通过梯度下降法，来对参数进行调整
# x : 样本数据中的（x0, x1）
# y ：样本数据中的目标标量
# m ：样本数据的个数，本例子中是100个
# theta ：参数θ ，是个 1 * 2 的矩阵，分别是参数b、w


def caculate_loss(x, y, m, theta):
    #  np.dot(x,  theta) 是矩阵乘法。x 是100 * 2矩阵， theta是一维的，可以看成 1 * 2 的矩阵
    #  np.dot(x,  theta) 的矩阵乘积是 100 * 1的矩阵，y 也是 100 * 1的矩阵，所以，直接相减
    loss = np.dot(x,  theta) - y

    # 代入损失函数，求出平均损失。这里开头的系数2无所谓，因为要乘以学习率，只要把学习率
    # 设置成原来的0.5倍，就相当于消除了这里的系数2
    return loss


# 通过梯度下降法，来对参数进行调整
# x : 样本数据中的（x0, x1）
# y ：样本数据中的目标标量
# theta ：参数θ ，是个 1 * 2 的矩阵，分别是参数b、w
# learn_rate ：学习率。学习率设置也很关键。为简单起见，这里依然采用常数
# m ：样本数据的个数，本例子中是100个
# num_Iterations : 最大迭代次数，一般地，我们判断模型是否可以输出，是根据误差函数足够小。
# 但是，为了防止因为误差函数无法收敛导致的死循环。所以，我们会设置最大迭代次数，
def gradient_descent(x, y, theta, learn_rate,  m,  num_Iterations):

    for i in range(0, num_Iterations):
        # 计算损失函数，
        loss = caculate_loss(x, y, m, theta)

        # loss 是一个 1 * 100 的矩阵， x是个 100 * 2的矩阵
        gradient = np.dot(loss,  x) / m

        # 更新参数
        theta = theta - learn_rate * gradient
        if i % 100 == 0:
            print("θ ： {0}， cost： {1} ".format(theta, np.sum(loss ** 2) / (2 * m)))
    return theta


# 线性回归函数，入口函数
def linear_regression():
    # 随机生成100个样本数据，总体上服从权重为10、偏执项为25
    x, y = generate_sample_data(100,  25,  10)
    m, n = np.shape(x)
    numIterations = 100000
    learn_rate = 0.0005
    theta = np.ones(n)
    theta = gradient_descent(x, y, theta, learn_rate, m, numIterations)

    print("y = {0} x + {1} ".format(round(theta[1], 2),  round(theta[0], 2)))


# 第一个机器学习的例子
linear_regression()
