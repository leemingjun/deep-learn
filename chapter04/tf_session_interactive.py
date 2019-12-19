#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf

v = tf.constant ([1.0, 2.0, 3.0])

sess = tf.InteractiveSession()

# 无需显示传递session，在交互式情况下使用更方便（代码量更小）
print(tf.log(v).eval())
sess.close()
