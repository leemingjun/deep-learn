#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf


def generate_sample_data(numPoints=100, weight=10, bias=26):
    """ 生成样本数据。使之符合 y = weight * x + bias """

    # numPoints : 样本数据的个数，默认是100个
    # bias ： 偏置项，为了体现随机性，在0.8 bias到1.2 bias之间随机波动
    # weight :  权重
    x_data = np.random.rand(numPoints)
    y_data = x_data * weight + bias * \
        tf.random_uniform(shape=[100], minval=0.8, maxval=1.2)

    return x_data, y_data


def caculate_loss(y_data, y_prediction):
    """ 通过梯度下降法，来对参数进行调整 """

    # tf.square(y_data - y_prediction)，求期望值与实际值差的平方
    # tf.sqrt()，求开方, 可以对Tensor操作
    # tf.reduce_mean(), 求平均值，对Tensor操作
    loss = tf.reduce_mean(tf.sqrt(tf.square(y_data - y_prediction)))

    # 记录loss的数值变化，记录到日志中，可以通过TensorBoard来查看
    tf.summary.scalar('loss', loss)
    return loss


def gradient_descent(learn_rate, loss):
    """ 使用梯度下降法优化器，来对参数进行优化（调整） """

    # 优化的目标是最小化损失（loss）
    optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    return optimizer.minimize(loss)


def linear_regression():
    """ 线性回归函数，入口函数 """

    # 随机生成100个样本数据，总体上服从权重为10、偏执项为25
    x_data, y_data = generate_sample_data(100, 10, 25)

    # 生成一个权重变量，取[-1.0, 1.0）的一个随机值
    weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="weight")

    # 将权重也记录到日志中（直方图），可以通过TensorBoard来查看
    tf.summary.histogram('weight', weight)

    # 将偏置项也记录到日志中（直方图），可以通过TensorBoard来查看
    bias = tf.Variable(tf.zeros([1]), name="bias")
    tf.summary.histogram('bias', bias)

    y_prediction = x_data * weight + bias

    loss = caculate_loss(y_data, y_prediction)
    # 采用梯度下降法调整权重，学习率(learn_rate)设置为0.5
    train = gradient_descent(0.05, loss)

    # 初始化变量
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # 将之前所有的想要保存到日志中的summary合并起来
        merged = tf.summary.merge_all()
        # 创建一个summary文件写入对象
        writer = tf.summary.FileWriter("./logs/", sess.graph)
        sess.run(init)

        for step in range(2000):
            sess.run(train)

            # 计算合并后的所有变量，并且，将他们写到日志中，供TensorBoard展示
            merged_summary = sess.run(merged)
            writer.add_summary(merged_summary, step)
            if step % 10 == 0:
                print("y={:.2f}x+{:.2f}".format(sess.run(weight)
                                                [0], sess.run(bias)[0]))


# 借助TensorFlow实现线性回归的例子
linear_regression()
