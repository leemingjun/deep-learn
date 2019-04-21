#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

# 导入依赖模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from read_imagenet_data import input_fn

# 设置日志级别。
tf.logging.set_verbosity(tf.logging.INFO)


def trunc_normal(stddev):
    """
    生成截断正太分布的随机数。
    与随机正太分布随机数函数`random_normal_initializer`类似，
    区别之处在于，将落在两个标准差之外的随机数丢弃，并且，重新生成。

    @param mean: 正太分布的均值。
    @param stddev: 该正太分布的标准差。

    @Returns: 截断正太分布的随机数。
    """
    return tf.truncated_normal_initializer(0.0, stddev)


def conv2d(inputs, filters, filter_size=[3, 3], strides=(1, 1),
           stddev=0.01, padding='SAME',
           activation=tf.nn.relu, scope='conv2d'):
    """
    定义densenet中默认的卷积函数

    @param input_layer: 输入层。
    @param stride: 步长。
    @param padding: 填充方式。常用的有“SAME”和“VALID”两种模式。
    @param weights_initializer: 填充方式。常用的有“SAME”和“VALID”两种模式。

    @Returns: 图片和图片的标签。图片是以张量形式保存的。
    """
    with tf.variable_scope(scope):
        weights_initializer = trunc_normal(stddev)
        return tf.layers.conv2d(
            inputs, filters, kernel_size=filter_size, strides=strides,
            activation=activation, padding=padding,
            kernel_initializer=weights_initializer,
            name=scope)


def max_pool2d(inputs, pool_size=(3, 3), strides=(2, 2),
               padding='SAME', scope='max_pool2d'):
    """
    定义最大池化函数，将densenet模型中最常用的参数设置为默认值。

    @param inputs: 输入张量。
    @param pool_size: 池化过滤器的尺寸。
    @param stride: 步长。
    @param padding: 填充方式。常用的有“SAME”和“VALID”两种模式。
    @param weights_initializer: 填充方式。常用的有“SAME”和“VALID”两种模式。

    @Returns: 图片和图片的标签。图片是以张量形式保存的。
    """
    with tf.variable_scope(scope):
        return tf.layers.max_pooling2d(
            inputs, pool_size, strides, padding, name=scope)


def densenet_block(inputs, blocks, is_training=True,  scope='densenet_block'):
    """
    构建一个DenseNet模块，其中，blocks 代表模块中卷积层的个数。

    @param inputs: 输入的特征图谱。
    @param blocks: 本模块中包含的卷积层个数。
    @param is_training: 布尔值，指示是否处于训练过程。
    @param scope: 当前变量所处的范围。

    @Returns: 构建好的DenseNet网络模块。
    """
    net = inputs
    with tf.variable_scope(scope):
        for i in range(int(blocks)):
            net = add_layer(net, 'densenet_block_{}'.format(i),
                            is_training=is_training)

    return net


def add_layer(inputs, name, is_training=True, growth_rate=32):
    """
    向 DenseNet 模块中添加一层。

    @param inputs: 输入的特征图谱。
    @param name: 本层的名称。
    @param is_training: 布尔值，指示是否处于训练过程。
    @param growth_rate: 增长率，代表超参增长率。

    @Returns: 拼接输出特征图谱之后的输入特征图谱。
    """
    with tf.variable_scope(name):
        net = tf.layers.batch_normalization(
            inputs, training=is_training, name='batch_normalization')
        net = tf.nn.relu(net)
        net = conv2d(net, growth_rate)
        # 将本层的运算结果串联到特征图谱
        net = tf.concat([net, inputs], 3)
    return net


def transition_layer(inputs, is_training=True, scope='transition_layer'):
    """
    向 DenseNet 模型中添加一个转换层。

    @param inputs: 输入的特征图谱。
    @param name: 本层的名称。
    @param is_training: 布尔值，指示是否处于训练过程。

    @Returns: 拼接输出特征图谱之后的输入特征图谱。
    """
    # 获取输入张量的形状，
    shape = inputs.get_shape().as_list()
    in_channel = shape[3]

    with tf.variable_scope(scope):
        net = tf.layers.batch_normalization(
            inputs, training=is_training, name='batch_normalization')
        net = tf.nn.relu(net)
        # 确保深度保持不变
        net = conv2d(net, in_channel, filter_size=[1, 1])
        net = tf.layers.average_pooling2d(
            net, [2, 2], strides=(2, 2),
            padding='SAME', name='AvgPool_0a_3x3')
    return net


def densenet_model(features, labels, mode):
    """
    构建一个 DenseNet 网络模型。本例是 DenseNet-121, k = 32

    我们把 DenseNet 看成由8个构建层组成，每个构建层都是由一个或多个普通卷积层、池化层、或者 DenseNet 模块组成。

    @param feautres: 输入的ImageNet样本图片，形状为[-1， 224, 224, 3]。
    @param labels: 样本数据的标签。
    @param mode: 模型训练所处的模式。

    @Returns: 构建好的 DenseNet 网络模型。
    """
    # 第一构建层，包含一个卷积层、一个最大池化层
    # 输入张量的形状 [224, 224, 3]， 输出张量的形状 [112, 112, 64]
    net = conv2d(features, 64, [7, 7], strides=(2, 2), scope='Conv2d_1a_7x7/2')
    # 输入张量的形状 [112, 112, 64]， 输出张量的形状 [56, 56, 64]
    net = max_pool2d(net, [3, 3], strides=(2, 2), scope='MaxPool_1b_3x3/2')

    # 是否是训练模式
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # 第二构建层，由 6 个 DenseNet模块组成
    net = densenet_block(net, 6, is_training, scope='densenet_blockx6')

    # 第三构建层，是一个转换层
    net = transition_layer(net, is_training, scope='transition_layer_3')

    # 第四构建层，由 12 个 DenseNet模块组成
    net = densenet_block(net, 12, is_training, scope='densenet_block_x12')

    # 第五构建层，是一个转换层
    net = transition_layer(net, is_training, scope='transition_layer_5')

    # 第六构建层，由 32 个 DenseNet模块组成
    net = densenet_block(net, 32, is_training, scope='densenet_block_x32')

    # 第七构建层。全局平均池化、全连接层、softmax层
    with tf.variable_scope('Logits'):

        # 全局平均池化
        net = tf.reduce_mean(
            net, [1, 2], keep_dims=True, name='global_pool')

        # 1 x 1 x 1024
        net = tf.layers.dropout(net, rate=0.7,
                                name='Dropout_1b')
        net = tf.layers.flatten(net)
        # 全连接层，共有1000个类别的
        logits = tf.layers.dense(
            inputs=net, units=1000, activation=None)

    # 第八构建层，softmax分类层
    predictions = {
        # (为 PREDICT 和 EVAL 模式)生成预测值
        "classes": tf.argmax(input=logits, axis=1),
        # 将 `softmax_tensor` 添加至计算图。用于 PREDICT 模式下的 `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # 如果是预测模式，那么，执行预测分析
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # 如果是训练模式，执行模型训练
    elif mode == tf.estimator.ModeKeys.TRAIN:
        # 计算损失（可用于`训练`和`评价`中）
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)
    else:
        # 计算损失（可用于`训练`和`评价`中）
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        # 添加评价指标（用于评估）
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def densenet_train():
    """
    训练一个 DenseNet 模型。
    """
    densenet_classifier = tf.estimator.Estimator(
        model_fn=densenet_model, model_dir="./logs/model/resnet/")

    # 开始 DenseNet 模型的训练
    densenet_classifier.train(
        input_fn=lambda: input_fn(True, './data/imagenet/', 128),
        steps=2000)

    # 评估模型并输出结果
    eval_results = densenet_classifier.evaluate(
        input_fn=lambda: input_fn(False, './data/imagenet/', 12))
    print("\n识别准确率: {:.2f}%\n".format(eval_results['accuracy'] * 100.0))


# 模型训练
densenet_train()
