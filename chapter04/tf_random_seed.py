#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf

# 如果指定了图的随机数种子，那么，每次生成的随机数序列都是" 一样的"
# tf.set_random_seed(1234)
# 如果需要每次生成的随机数序列都不会重复，那么，不要设置图随机数种子

a = tf.random_uniform([1])
b = tf.random_normal([1])

# 如果设置了操作的随机数种子，那么，该操作生成的随机数是" 重复的"
# 也就是说，张量a 对应的A1、A2 都相等
# 如果需要该操作每次生成的随机数序列都不重复，那么，不要设置操作的随机数种子
print("会话 1")
with tf.Session() as sess1:
    print(sess1.run(a))    # 生成 'A1'
    print(sess1.run(a))    # 生成 'A2'
    print(sess1.run(b))    # 生成 'B1'
    print(sess1.run(b))    # 生成 'B2'

print("会话 2")
with tf.Session() as sess2:
    print(sess2.run(a))    # 生成 'A1'
    print(sess2.run(a))    # 生成 'A2'
    print(sess2.run(b))    # 生成 'B1'
    print(sess2.run(b))    # 生成 'B2'
