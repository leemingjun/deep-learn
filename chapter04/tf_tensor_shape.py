#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf

t = tf.constant([[[1, 1, 1, 1], [2, 2, 2, 2]], [[3, 3, 3, 3], [4, 4, 4, 4]]])
# tf.shape(t) # [2 2 4]

# 打印二阶张量的阶和形状，分别是:
with tf.Session() as sess:
    print(sess.run(tf.shape(t)))
    print(sess.run(tf.size(t)))
    print(sess.run(tf.reshape(t, [-1])))
    print(sess.run(tf.reshape(t, [2, 8])))
    # 错误，张量的元素个数是必须保持不变，变形之前是4×4=16个元素，
    # 变形之后不能变成 2×4=8个元素
    # print(sess.run(tf.reshape(t, [2, 4])))
    
    # 第一维指定为-1，表示根据其他维度计算本维度的元素数量
    # 指定了第二维为4个元素，[[1 1 1 1]， [2 2 2 2]，[3 3 3 3]，[4 4 4 4]]
    print(sess.run(tf.reshape(t, [-1, 4])))
    # 第一维指定为2个元素，第二维指定为-1，表示根据其他维元素数量来计算
    # [[1 1 1 1 2 2 2 2],  [3 3 3 3 4 4 4 4]]
    print(sess.run(tf.reshape(t, [2, -1])))
