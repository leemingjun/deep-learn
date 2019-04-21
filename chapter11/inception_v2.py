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


def conv2d(inputs, filters, kernel_size=[3, 3], stride=(1, 1),
           stddev=0.01, padding='SAME', scope='conv2d'):
    """
    定义Inception中默认的卷积函数

    @param input_layer: 输入层。
    @param stride: 步长。
    @param padding: 填充方式。常用的有“SAME”和“VALID”两种模式。
    @param weights_initializer: 填充方式。常用的有“SAME”和“VALID”两种模式。

    @Returns: 图片和图片的标签。图片是以张量形式保存的。
    """
    with tf.variable_scope(scope):
        weights_initializer = trunc_normal(stddev)

        return tf.layers.conv2d(
            inputs, filters, kernel_size=kernel_size, strides=stride,
            padding=padding, kernel_initializer=weights_initializer,
            name=scope)


def max_pool2d(inputs, pool_size=(3, 3), stride=(2, 2),
               padding='SAME', scope='max_pool2d'):
    """
    定义最大池化函数，将Inception模型中最常用的参数设置为默认值。

    @param inputs: 输入张量。
    @param pool_size: 池化过滤器的尺寸。
    @param stride: 步长。
    @param padding: 填充方式。常用的有“SAME”和“VALID”两种模式。
    @param weights_initializer: 填充方式。常用的有“SAME”和“VALID”两种模式。

    @Returns: 图片和图片的标签。图片是以张量形式保存的。
    """
    with tf.variable_scope(scope):
        return tf.layers.max_pooling2d(
            inputs, pool_size, stride, padding, name=scope)


def depth(d, min_depth=16):
    """
    对输出通道数进行约束。
    计算输出通道数，避免设置的输出通道数过低
    如果，经过通道扩张因子计算之后，输出通道数小于最小的输出通道数，以最小的通道数为准

    @param d: 建议的输出通道数
    @param min_depth: 最小的输出通道数

    @Returns: 经过约束的输出通道数。
    """
    return max(d, min_depth)


def inception_v2_block(net, filters, strides=(1, 1), is_max_pool=False,
                       scope='inception_v2_block'):
    """
    定义Inception v2模块。

    @param net: 输入张量。
    @param filters: Inception模块中，各个分支中，过滤器的输出通道数。
        filters[0],代表分支(Branch_0) 的 1x1 卷积的输出通道数

        filters[1],代表分支(Branch_1) 的 1x1 卷积的输出通道数
        filters[2],代表分支(Branch_1) 的 3x3 卷积的输出通道数

        filters[3],代表分支(Branch_2) 的 1x1 卷积的输出通道数
        filters[4],代表分支(Branch_2) 的 3x3 卷积的输出通道数
        filters[4],代表分支(Branch_2) 的 3x3 卷积的输出通道数

        filters[5],代表分支(Branch_3) 的 1x1 卷积的输出通道数
    @param scope: 变量所属的作用域

    @Returns: 经过Inception模块卷积后的张量。
    """
    with tf.variable_scope(scope):

        # 1x1卷积分支
        with tf.variable_scope('Branch_0'):
            # 如果 filters[0] == 0，则不需要该分支
            if filters[0] > 0:
                branch_0 = conv2d(
                    net, depth(filters[0]), [1, 1],
                    stride=strides, scope='Conv2d_0a_1x1')

        # 1x1-->3x3 卷积分支
        with tf.variable_scope('Branch_1'):
            branch_1 = conv2d(
                net,
                depth(filters[1]), [1, 1],
                stddev=0.09,
                scope='Conv2d_0a_1x1')
            branch_1 = conv2d(
                branch_1, depth(filters[2]), [3, 3],
                stride=strides, scope='Conv2d_0b_3x3')

        # 1x1-->3x3-->3x3 卷积分支
        with tf.variable_scope('Branch_2'):
            branch_2 = conv2d(
                net,
                depth(filters[3]), [1, 1], stride=strides,
                stddev=0.09,
                scope='Conv2d_0a_1x1')
            branch_2 = conv2d(
                branch_2, depth(filters[4]), [3, 3],
                stride=(1, 1), scope='Conv2d_0b_3x3')
            branch_2 = conv2d(
                branch_2, depth(filters[4]), [3, 3],
                stride=(1, 1), scope='Conv2d_0c_3x3')

        # 3x3平均池化-->1x1卷积 分支
        with tf.variable_scope('Branch_3'):
            # Pool Projection分支，是执行平均池化、还是执行最大池化
            if is_max_pool is False:
                branch_3 = tf.layers.average_pooling2d(
                    net, [3, 3], strides=strides,
                    padding='SAME',  name='AvgPool_0a_3x3')
            else:
                branch_3 = max_pool2d(
                    net, [3, 3], stride=strides,
                    padding='SAME',  scope='MaxPool_0a_3x3')

            # 如果 filters[5] > 0 则执行 1x1卷积，否则，Pass through
            if filters[5] > 0:
                branch_3 = conv2d(
                    branch_3,
                    depth(filters[5]), [1, 1], stride=(1, 1),
                    stddev=0.1,
                    scope='Conv2d_0b_1x1')

        # 将所有的分支沿着维度3（深度）串联起来
        if filters[0] > 0:
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        else:
            net = tf.concat([branch_1, branch_2, branch_3], 3)

    return net


def inception_v2_base(inputs,
                      min_depth=16,
                      scope=None):
    """
    构建一个Inception v2 模型。详细情况可参考 http://arxiv.org/abs/1502.03167 中
    Inception(5b)。

    请注意，本模型虽然参考了原始的论文中的模型架构，本模型尽可能的与原始的模型架构一致，但是，
    二者并不完全是一一对应，包括Inception模块的各个分支的顺序（从左到右）、以及模型最后的
    全连接部分、分类预测部分等。

    @param inputs: 输入张量，代表ImageNet中的图片，形状为 [batch_size, 224, 224, 3]。
    @param min_depth: 最小的输出通道数，针对所有的卷及操作。当 depth_multiplier < 1 时，
        起到避免输出通道过少的作用； 当 depth_multiplier >= 1 时，不会激活。
    @param scope: 可选项。变量的作用域。

    @Returns: 构建好的Inception v2模型。
    """

    # 在 InceptionV2_Model 模型的变量范围
    with tf.variable_scope('InceptionV2_Model'):

        # 第一个构建层。 深度分层卷积。
        # 首先，对原始输入张量inputs（形状 [batch_size, height, width, depth]）逐层进行卷积操作，
        # 输出形状为 [batch_size, height, width, depth * depthwise_multiplier]。
        # 然后，最对上述的输出张量，执行 1x1 卷积(深度为depth * depthwise_multiplier)，
        # 输出张量[batch_size, height, width, output_channels]
        # 输入形状 [224, 224, 3], 输出形状 [112, 112, 64]
        net = tf.layers.separable_conv2d(
            inputs,
            64, [7, 7],
            strides=2,
            name='Conv2d_1a_7x7')

        # 第二个构建层。包括一个最大池化层、两个堆叠的3x3卷积层
        # 输入形状 [112, 112, 64], 输出形状 [56, 56, 64]
        net = max_pool2d(net, [3, 3], scope='MaxPool_2a_3x3', stride=2)
        # 56 x 56 x 64
        net = conv2d(net, depth(64), [1, 1],
                     scope='Conv2d_2b_1x1', stddev=0.1)
        # 56 x 56 x 64
        net = conv2d(net, depth(192), [3, 3], scope='Conv2d_2c_3x3')

        # 第三个构建层。
        # 输入形状 [56, 56, 64], 输出形状 [28, 28, 192]
        net = max_pool2d(net, [3, 3], scope='MaxPool_3a_3x3', stride=2)
        # 输入形状 [28, 28, 192], 输出形状 [28, 28, 256]
        net = inception_v2_block(
            net, filters=[64, 64, 64, 64, 96, 32], scope='Inception_v2_3a')
        # 输入形状 [28, 28, 256], 输出形状 [28, 28, 320]
        net = inception_v2_block(
            net, filters=[64, 64, 96, 64, 96,  64], scope='Inception_v2_3b')
        # 输入形状 [28, 28, 320], 输出形状 [28, 28, 576]
        net = inception_v2_block(
            net, filters=[0, 128, 160, 64, 96, 0],
            strides=(2, 2), is_max_pool=True, scope='Inception_v2_3c')

        # 第四个构建层。
        # 输入形状 [28, 28, 576], 输出形状 [14, 14, 576]
        net = inception_v2_block(
            net, filters=[224, 64, 96, 96, 128, 128],
            scope='Inception_v2_4a')
        # 输入形状 [14, 14, 576], 输出形状 [14, 14, 576]
        net = inception_v2_block(
            net, filters=[192, 96, 128, 96, 128, 128],
            scope='Inception_v2_4b')
        # 输入形状 [14, 14, 576], 输出形状 [14, 14, 576]
        net = inception_v2_block(
            net, filters=[160, 128, 160, 128, 160, 128],
            scope='Inception_v2_4c')
        # 输入形状 [14, 14, 576], 输出形状 [14, 14, 576]
        net = inception_v2_block(
            net, filters=[96, 128, 192, 160, 192, 128],
            scope='Inception_v2_4d')
        # 输入形状 [14, 14, 576], 输出形状 [14, 14, 1024]
        net = inception_v2_block(
            net, filters=[0, 128, 192, 192, 256, 0],
            strides=(2, 2), is_max_pool=True, scope='Inception_v2_4e')

        # 第五个构建层。
        # 输入形状 [7, 7, 1024], 输出形状 [7, 7, 1024]
        net = inception_v2_block(
            net, filters=[352, 192, 320, 160, 224, 128],
            scope='Inception_v2_5a')
        # 输入形状 [7, 7, 1024], 输出形状 [7, 7, 1024]
        net = inception_v2_block(
            net, filters=[352, 192, 320, 192, 224, 128],
            is_max_pool=True, scope='Inception_v2_5b')

        # Inception_v2_5b 之后的层。全局平均池化和全连接层
        with tf.variable_scope('Logits'):
            # 全局平均池化
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

    return logits


def inception_v2_model(features, labels, mode):
    net = inception_v2_base(features)

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
        # 在训练阶段，旁路分类器的结果乘以权重 0.3 增加到主干的分类结果中
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


def inception_v2_train():
    """
    训练一个 inception_v2 模型
    """
    inception_classifier = tf.estimator.Estimator(
        model_fn=inception_v2_model, model_dir="./tmp/inception_v2/")

    # 开始Inception模型的训练
    inception_classifier.train(
        input_fn=lambda: input_fn(True, './data/imagenet/', 128),
        steps=2000)

    # 评估模型并输出结果
    eval_results = inception_classifier.evaluate(
        input_fn=lambda: input_fn(False, './data/imagenet/', 12))
    print("\n识别准确率: {:.2f}%\n".format(eval_results['accuracy'] * 100.0))


# 模型训练
inception_v2_train()
