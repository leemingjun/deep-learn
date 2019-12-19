#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf

tensor_constant = tf.constant(
    [[1, 2, 3, 4], [5, 6, 7, 8]], name="tensor_constant")

tensor1 = tf.Variable([[1, 2, 3], [4, 5, 6]], name="tensor1")
tensor2 = tf.Variable([[10, 20, ], [30, 40], [50, 60]], name="tensor2")
tensor3 = tf.matmul(tensor1, tensor2)

# 必须执行变量初始化，否则，保存会出现错误
init = tf.global_variables_initializer()

# 在with 语句中打开sess，确保在with 语句之外sess 会关闭
with tf.Session() as sess:
    # 运行会话
    sess.run(init)

    # 如果不执行变量初始化操作，会出现错误:Attempting to use uninitialized value
    # sess.run(tensor3)

    # 保存所有的变量，系统自动挑选当前计算图中包含的" 变量"
    saver = tf.train.Saver(max_to_keep=5)

    # 只保存变量tensor1 和变量tensor2
    # saver = tf.train.Saver([tensor1, tensor2], max_to_keep=5)

    # 常量不能保存，否则会出现错误：TypeError: Variable to save is not a Variable
    # saver = tf.train.Saver(var_list=[tensor_constant, tensor3], max_to_keep=5)

    # tensor3 是常量，不能保存，否则会出现：TypeError: Variable to save is not a Variable
    # saver = tf.train.Saver(var_list=[tensor1, tensor2, tensor3], max_to_keep=5)

    # 还可以指定了global_step，会将global_step 作为保存的文件名一部分追加在"my-model" 后面
    # 这样，就构成了检查点文件名
    # 当step==0 时，保存的文件名是"my-model-0"
    # 当step==100 时，保存的文件名是"my-model-100"
    print(saver.save(sess, './checkpoint/my-model', global_step=100))
