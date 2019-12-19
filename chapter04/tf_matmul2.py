#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

# 2维 张量 `a_2d`
# [[1, 2, 3],
#  [4, 5, 6]]
a_2d = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name="a_2d")

# 2维 张量 `b_2d`
# [[ 7,  8],
#  [ 9, 10],
#  [11, 12]]
b_2d = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2], name="b_2d")

#  矩阵乘法，张量 `a_2d` 乘以张量 `b_2d`，输出张量  `c_2d`
#  矩阵乘法需要注意，第一个矩阵的列数要等于第二个矩阵的行数
# [[ 58,  64],
#  [139, 154]]
c_2d = tf.matmul(a_2d, b_2d, name="c_2d")


# 3维 张量 `a_3d`
# [[[ 1,  2,  3],
#   [ 4,  5,  6]],
#  [[ 7,  8,  9],
#   [10, 11, 12]]]
a_3d = tf.constant(np.arange(1, 13, dtype=np.int32),
                shape=[2, 2, 3], name="a_3d")

# 3维 张量 `b_3d`
# [[[13, 14],
#   [15, 16],
#   [17, 18]],
#  [[19, 20],
#   [21, 22],
#   [23, 24]]]
b_3d = tf.constant(np.arange(13, 25, dtype=np.int32),
                shape=[2, 3, 2], name="b_3d")

#  矩阵乘法，张量 `a_3d` 乘以张量 `b_3d`，输出张量  `c_3d`
# [[[ 94, 100],
#   [229, 244]],
#  [[508, 532],
#   [697, 730]]]
c_3d = tf.matmul(a_3d, b_3d, name="c_3d")

#  with的作用在于，确保在with语句之外，with语句内打开的对象关闭
#  本例中是确保sess对象在with语句之外关闭
with tf.Session() as sess:
    print ("Tensor c_2d is : \n")
    print (sess.run(c_2d))
    print ("Tensor c_3d is : \n")
    print (sess.run(c_3d))
    # 将数据图保存在到日志中，之后，可以通过Tensorboard来查看
    writer = tf.summary.FileWriter('./graph2', sess.graph)
    writer.flush()
    writer.close()
