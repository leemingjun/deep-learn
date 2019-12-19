#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf

# 访问一阶张量，只需要指定一个索引即可返回一个标量
my_vector = tf.constant([36, 38], shape=[2], name="age_list")
my_scalar = my_vector[1]

# 创建一个二阶张量，也就是二维数组
my_matrix = tf.constant([[36, 38], [40, 50]], shape=[2, 2])

# 指定两个维度，返回一个标量
my_scalar = my_matrix[0, 1]

# 指定零阶，返回一行
# 对[[36, 38], [40, 50]]，指定了零阶，去掉零阶的方括号，得到两个元素[36, 38], [40, 50]
# 再看指定的索引数值是 0 ，返回第一个元素[36,  38]
my_row_vector = my_matrix[0]

# 指定一阶，返回一列
# 对[[36, 38], [40, 50]]，指定了一阶，去掉一阶的方括号，得到[ {36, 38}, {40, 50}]
# 其中{}为了方便，展示去掉一阶之后，两组元素的区隔
# 再看指定的索引数值是 1 ，分别返回两个元素中的第二个数值，得到[38, 50]
my_column_vector = my_matrix[:, 1]

# 打印结果
# 38
# [36 38]
# [38 50]
with tf.Session() as sess: 
    print (sess.run(my_scalar))
    print (sess.run(my_row_vector))
    print (sess.run(my_column_vector))