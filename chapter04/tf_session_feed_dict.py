#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf

tensor1 = tf.placeholder(tf.int32, shape=[2, 3], name="tensor1")
tensor2 = tf.placeholder(tf.int32, shape=[2, 3], name="tensor2")
tensor3 = tf.placeholder(tf.int32, shape=[3, 1], name="tensor3")


# 构建计算图，采用add 操作符
add = tf.add(tensor1, tensor2)

# 构建计算图，采用matmul作为操作符
mul = tf.matmul(tensor1, tensor3,)

# 演示通过数据注入的方式运行Session
with tf.Session() as sess:
    # 为tensor1、tensor2注入实际数据，然后，计算add张量
    print (sess.run(add, feed_dict={tensor1:[[1, 2, 3], [4, 5, 6]], tensor2:[[10, 20, 30], [40, 50, 60]]}))
    # 为tensor1、tensor3，然后，计算mul张量
    print (sess.run(mul, feed_dict={tensor1:[[1, 2, 3], [4, 5, 6]], tensor3:[[ 20], [ 50], [80]]}))
