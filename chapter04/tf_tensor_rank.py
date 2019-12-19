#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf

# Rank 0
# 创建零阶张量，也就是标量，用来表示用户姓名和年龄
# 数据类型分别是tf.string和tf.int32
my_scalar = tf.constant("张三", dtype=tf.string, shape=[], name="user_name")
age_scalar = tf.constant(36, dtype=tf.int32, shape=[], name="age")

# Rank 1
# 创建一阶张量，也就是以为数组，用来表示用户姓名和年龄列表
# 数据类型没有指定，TensorFlow会根据实际数值的类型来推断数据类型
use_name_list = tf.constant(["张三", "李四"], shape=[2], name="user_name_list")
age_list = tf.constant([36, 38], shape=[2], name="age_list")

# Rank 2
# 创建二阶张量，用户先分组，用来表示各个组内用户姓名和年龄列表
group_name_list = tf.constant([["张三", "李四"], ["王五", "赵六"]], shape=[2, 2])
group_age_list = tf.constant([[36, 38], [40, 50]], shape=[2, 2])

# 打印二阶张量的阶和形状，分别是:
# Rank:  2
# Shape:  [2 2]
with tf.Session() as sess:
    print ("Rank:  {}" .format(sess.run(tf.rank(group_name_list))))
    print ("Shape:  {}".format(sess.run(tf.shape(group_name_list))))
