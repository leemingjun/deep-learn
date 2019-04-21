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
    定义ResNet中默认的卷积函数

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
    定义最大池化函数，将ResNet模型中最常用的参数设置为默认值。

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


def resnet_block(inputs, filters, scope='resnet_block'):
    """
     构建一个残差模块(ResNet模块)。

     @param inputs: 输入张量。
     @param filters: ResNet模块中的各个过滤器的个数。
        主干分支上有两个 3x3 过滤器，他们的输出通道数相同。
     @param scope: 代表该构建块的名称。

     @Returns: 该残差模块的网络模型。
     """
    with tf.variable_scope(scope):
        # 主干分支，两个 3x3 卷积堆叠
        with tf.variable_scope('Trunck'):
            # 默认采用 尺寸 3x3、步长 1x1 、ReLu激活函数
            trunck = conv2d(inputs, filters,  scope='trunck_1a_3x3')

            # 第二个卷积操作不需要使用ReLu激活函数，与旁路分支相加之后，再激活
            trunck = conv2d(trunck, filters,
                            activation=None, scope='trunck_1b_3x3')

        # 快捷连接（旁路分支）
        with tf.variable_scope('shortcut'):
            # 恒等变换，无需执行任何操作
            shortcut = inputs

        # 相加、并ReLu激活
        net = tf.nn.relu(tf.add(trunck, shortcut), name='add_relu')

    # 返回结果
    return net


def resnet_block_downsample(inputs, filters, strides=(2, 2),
                            scope='resnet_block_downsample'):
    """
     构建一个降采样的残差模块(ResNet模块)。

     将输入张量的宽度、高度缩减到原来的一半、将深度加倍。

     @param inputs: 输入张量。
     @param filters: ResNet模块中的各个过滤器的个数。
        filters[0],代表主干分支上第一个 3x3 卷积的输出通道数
        filters[2],代表主干分支上第一个 3x3 卷积的输出通道数

     @param scope: 代表该构建块的名称。

     @Returns: 该残差模块的网络模型。
     """
    with tf.variable_scope(scope):
        # 主干分支，两个 3x3 卷积堆叠
        with tf.variable_scope('Trunck'):
            # 采用 尺寸 3x3、步长为 2x2、ReLu激活函数
            # 宽度、高度缩减为原来一半、深度加倍
            trunck = conv2d(inputs, filters, strides=strides,
                            scope='trunck_1a_3x3')

            # 采用 尺寸 3x3、步长为 1x1
            # 第二个卷积操作不需要使用ReLu激活函数，与旁路分支相加之后，再激活
            trunck = conv2d(trunck, filters,
                            activation=None, scope='trunck_1b_3x3')

        # 快捷连接（旁路分支）
        with tf.variable_scope('shortcut'):
            # 恒等变换，为了实现降采样采用 尺寸 1x1、步长为 2x2 的卷积
            # 宽度、高度缩减一半、输出通道数加倍
            shortcut = conv2d(inputs, filters, filter_size=[1, 1],
                              strides=[2, 2], activation=None,
                              scope='shortcut_1x1')

    # 相加、并ReLu激活
    net = tf.nn.relu(tf.add(trunck, shortcut), name='add_relu')

    # 返回结果
    return net


def resnet_model(features, labels, mode):
    """
    构建一个ResNet网络模型。这里构建的是一个34层的ResNet网络。

    我们把ResNet看成由8个构建层组成，每个构建层都是由一个或多个普通卷积层、池化层、或者ResNet模块组成。

    @param feautres: 输入的ImageNet样本图片，形状为[-1， 224, 224, 3]。
    @param labels: 样本数据的标签。
    @param mode: 模型训练所处的模式。

    @Returns: 网络模型。
    """

    # 第一构建层，包含一个卷积层、一个最大池化层
    # 输入张量的形状 [224, 224, 3]， 输出张量的形状 [112, 112, 64]
    net = conv2d(features, 64, [7, 7], strides=(2, 2), scope='Conv2d_1a_7x7/2')
    # 输入张量的形状 [112, 112, 64]， 输出张量的形状 [56, 56, 64]
    net = max_pool2d(net, [3, 3], strides=(2, 2), scope='MaxPool_1b_3x3/2')

    # 第二构建层，由三个残差模块构成
    # 输入张量与输出张量形状一致，都是[56, 56, 64]
    net = resnet_block(net, 64, scope='ResNet_2a')
    net = resnet_block(net, 64, scope='ResNet_2b')
    net = resnet_block(net, 64, scope='ResNet_2c')

    # 第三构建层，由四个残差模块构成
    # 输入张量形状：[56, 56, 64]， 输出张量形状：[28, 28, 128]
    net = resnet_block_downsample(net, 128, scope='ResNet_3a')
    net = resnet_block(net, 128, scope='ResNet_3b')
    net = resnet_block(net, 128, scope='ResNet_3c')
    net = resnet_block(net, 128, scope='ResNet_3d')

    # 第四构建层，由六个残差模块构成
    # 输入张量形状：[28, 28, 128]， 输出张量形状：[14, 14, 256]
    net = resnet_block_downsample(net, 256, scope='ResNet_4a')
    net = resnet_block(net, 256, scope='ResNet_4b')
    net = resnet_block(net, 256, scope='ResNet_4c')
    net = resnet_block(net, 256, scope='ResNet_4d')
    net = resnet_block(net, 256, scope='ResNet_4e')
    net = resnet_block(net, 256, scope='ResNet_4f')

    # 第五构建层，由三个残差模块构成
    # 输入张量形状：[14, 14, 256]， 输出张量形状：[7, 7, 512]
    net = resnet_block_downsample(net, 512, scope='ResNet_5a')
    net = resnet_block(net, 512, scope='ResNet_5b')
    net = resnet_block(net, 512, scope='ResNet_5c')

    # 5c 之后的层。全局平均池化、全连接层、softmax层
    with tf.variable_scope('Logits'):

        # 第六构建层，全局平均池化
        net = tf.reduce_mean(
            net, [1, 2], keep_dims=True, name='global_pool')

        # 1 x 1 x 1024
        net = tf.layers.dropout(net, rate=0.7,
                                name='Dropout_1b')

        # 第七构建层，包括一个线性转换层
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


def resnet_train():
    """
    训练一个ResNet模型。
    """
    resnet_classifier = tf.estimator.Estimator(
        model_fn=resnet_model, model_dir="./logs/model/resnet/")

    # 开始 ResNet 模型的训练
    resnet_classifier.train(
        input_fn=lambda: input_fn(True, './data/imagenet/', 128),
        steps=2000)

    # 评估模型并输出结果
    eval_results = resnet_classifier.evaluate(
        input_fn=lambda: input_fn(False, './data/imagenet/', 12))
    print("\n识别准确率: {:.2f}%\n".format(eval_results['accuracy'] * 100.0))


# 模型训练
resnet_train()
