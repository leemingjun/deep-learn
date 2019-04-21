#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

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


def conv2d(inputs, filters, kernel_size=[7, 7], stride=(1, 1),
           stddev=0.01, padding='SAME', scope='conv2d'):
    """
    定义Inception中默认的卷积函数

    @param inputs: 输入层。
    @param filters: 过滤器的个数（输出通道数）
    @param stride: 步长。
    @param stddev: 生成权重正太分布随机数的标准差。
    @param padding: 填充方式。常用的有“SAME”和“VALID”两种模式。
    @param scope: 当前函数中变量的作用域。

    @Returns: 卷积之后的特征图谱。
    """
    with tf.variable_scope(scope):
        weights_initializer = trunc_normal(stddev)

        return tf.layers.conv2d(
            inputs, filters, kernel_size=kernel_size, strides=stride,
            padding=padding, kernel_initializer=weights_initializer,
            name=scope)


def max_pool2d(inputs, pool_size=(3, 3), strides=(2, 2),
               padding='SAME', scope='max_pool2d'):
    """
    定义最大池化函数，将Inception模型中最常用的参数设置为默认值。

    @param inputs: 输入张量。
    @param pool_size: 池化过滤器的尺寸。
    @param stride: 步长。
    @param padding: 填充方式。常用的有“SAME”和“VALID”两种模式。
    @param scope: 当前函数中变量的作用域。

    @Returns: 执行池化操作生成的特征图谱。
    """
    with tf.variable_scope(scope):
        return tf.layers.max_pooling2d(
            inputs, pool_size, strides, padding, name=scope)


def inception_block(net, filters, scope='inception_block'):
    """
    定义Inception模块。

    @param net: 输入张量。
    @param filters: Inception模块中，各个分支中，过滤器的输出通道数。
        filters[0],代表分支(Branch_0) 的 1x1 卷积的输出通道数

        filters[1],代表分支(Branch_1) 的 1x1 卷积的输出通道数
        filters[2],代表分支(Branch_1) 的 3x3 卷积的输出通道数

        filters[3],代表分支(Branch_2) 的 1x1 卷积的输出通道数
        filters[4],代表分支(Branch_2) 的 5x5 卷积的输出通道数

        filters[5],代表分支(Branch_3) 的 1x1 卷积的输出通道数
    @param scope: 变量所属的作用域

    @Returns: 经过Inception模块卷积后的张量。
    """
    with tf.variable_scope(scope):
        # 1x1卷积分支
        with tf.variable_scope('Branch_0'):
            branch_0 = conv2d(net, filters[0], [1, 1], scope='Conv2d_0a_1x1')

        # 1x1-->3x3 卷积分支
        with tf.variable_scope('Branch_1'):
            branch_1 = conv2d(net, filters[1], [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = conv2d(
                branch_1, filters[2], [3, 3], scope='Conv2d_0b_3x3')

        # 1x1-->5x5 卷积分支
        with tf.variable_scope('Branch_2'):
            branch_2 = conv2d(net, filters[3], [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = conv2d(
                branch_2, filters[4], [5, 5], scope='Conv2d_0b_3x3')

        # 3x3池化-->1x1 卷积分支
        with tf.variable_scope('Branch_3'):
            # 步长为1，保证最大池化操作之后的张量形状与其他分支保持一致
            branch_3 = max_pool2d(
                net, [3, 3], strides=(1, 1), scope='MaxPool_0a_3x3')
            branch_3 = conv2d(
                branch_3, filters[5], [1, 1], scope='Conv2d_0b_1x1')

    # 将所有的分支沿着维度3（深度）串联起来
    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
    return net


def auxiliary_classifier(net, scope='auxiliary_classifier'):
    """
    构建一个旁路分类器。
    注意：旁路分类器不能改变输入张量net，因为，主干上依然要访问该变量所指向的特征图谱。

    @param net: 从 4a 或者 4d 输出的特征图谱

    @Returns: 旁路分类器的网络模型。
    """
    # （4a）、（4d）输入张量形状不一样
    with tf.variable_scope(scope):
        # （4a）的分支池化结果张量的形状 [4, 4, 512]
        # （4d）的分支池化结果张量的形状 [4, 4, 528]
        # 对应的过滤器的深度不一样，也就是说，参数的个数不一样。
        aux_branch = tf.layers.average_pooling2d(
            inputs=net, pool_size=[5, 5], strides=(3, 3),
            name='average_pooling2d')
        aux_branch = conv2d(aux_branch, 128, [1, 1], scope='aux_conv2d_128')

        # 展平，准备与全连接连接
        aux_branch = tf.layers.flatten(aux_branch)
        # 全连接层，共有1024个类别的
        aux_branch = tf.layers.dense(
            inputs=aux_branch, units=1024, activation=None,
            name='auxiliary_dense_1024')

        # 1 x 1 x 1024
        aux_branch = tf.layers.dropout(aux_branch, rate=0.7,
                                       name='Dropout_1b')

        # 全连接层，共有1000个类别的
        logits = tf.layers.dense(
            inputs=aux_branch, units=1000, activation=None,
            name='auxiliary_dense_1000')

    return logits


def inception_v1_base(images):
    """
    构建一个Inception模型，包括第一构建层 到 第五构建层。

    我们把Inception看成由8个构建层组成，每个构建层都是由一个或多个普通卷积层、
    池化层、或者Inception模块组成。

    @param images: 输入的ImageNet图片，形状为[224, 224, 3]。

    @Returns: 网络模型。
    """

    # 第一构建层，包含一个卷积层、一个最大池化层
    # 输入张量的形状 [224, 224, 3]， 输出张量的形状 [112, 112, 64]
    net = conv2d(images, 64, [7, 7], stride=(2, 2), scope='Conv2d_1a_7x7')

    # 第二构建层，包含一个最大池化层、一个1x1卷积降维层、一个3x3卷积层
    # 输入张量的形状 [112, 112, 64]， 输出张量的形状 [56, 56, 64]
    net = max_pool2d(net, [3, 3], scope='MaxPool_2a_3x3')
    # 输入张量的形状 [56, 56, 64]， 输出张量的形状 [56, 56, 64]
    net = conv2d(net, 64, [1, 1], scope='Conv2d_2b_1x1')
    # 输入张量的形状 [56, 56, 64]， 输出张量的形状 [56, 56, 192]
    net = conv2d(net, 192, [3, 3], scope='Conv2d_2c_3x3')

    # 第三构建层，包含一个最大池化层、两个Inception模块(3a,3b)
    # 最大池化层。输入张量的形状 [56, 56, 192]， 输出张量的形状 [28, 28, 192]
    net = max_pool2d(net, [3, 3],  scope='MaxPool_3a_3x3')
    # 输入张量的形状 [28, 28, 192]， 输出张量的形状 [28, 28, 256]
    net = inception_block(
        net, filters=[64, 96, 128, 16, 32, 32], scope='inception_3a')
    # 输入张量的形状 [28, 28, 256]， 输出张量的形状 [28, 28, 480]
    net = inception_block(
        net, filters=[128, 128, 192, 32, 96, 64], scope='inception_3b')

    # 第四构建层，包含一个最大池化层、五个Inception模块 (4a, 4b, 4c, 4d, 4e)
    # 输入张量的形状 [28, 28, 480]， 输出张量的形状 [14, 14, 480]
    net = max_pool2d(net, [3, 3],  scope='MaxPool_4a_3x3')
    # 输入张量的形状 [14, 14, 480]， 输出张量的形状 [14, 14, 512]
    net = inception_block(
        net, filters=[192, 96, 208, 16, 48, 64], scope='inception_4a')
    # 从 （4a） 出发的旁路分类器
    logits_4a = auxiliary_classifier(net, scope='auxiliary_4a')
    # 输入张量的形状 [14, 14, 512]， 输出张量的形状 [14, 14, 512]
    net = inception_block(
        net, filters=[160, 112, 224, 24, 64, 64], scope='inception_4b')
    # 输入张量的形状 [14, 14, 512]， 输出张量的形状 [14, 14, 512]
    net = inception_block(
        net, filters=[128, 128, 256, 24, 64, 64], scope='inception_4c')
    # 输入张量的形状 [14, 14, 512]， 输出张量的形状 [14, 14, 528]
    net = inception_block(
        net, filters=[112, 144, 288, 32, 64, 644], scope='inception_4d')
    # 从 （4d） 出发的旁路分类器
    logits_4d = auxiliary_classifier(net, scope='auxiliary_4d')
    # 输入张量的形状 [14, 14, 528]， 输出张量的形状 [14, 14, 823]
    net = inception_block(
        net, filters=[256, 160, 320, 32, 128, 128], scope='inception_4e')

    # 第五构建层，包含一个最大池化层、两个Inception模块 (5a, 5b)
    # 输入张量的形状 [14, 14, 823]， 输出张量的形状 [7, 7, 823]
    net = max_pool2d(net, [3, 3],  scope='MaxPool_5a_2x2')
    # 输入张量的形状 [7, 7, 823]， 输出张量的形状 [7, 7, 823]
    net = inception_block(
        net, filters=[256, 160, 320, 32, 128, 128], scope='inception_5a')
    # 输入张量的形状 [7, 7, 823]， 输出张量的形状 [7, 7, 1024]
    net = inception_block(
        net, filters=[384, 192, 384, 48, 128, 128], scope='inception_5b')

    return net, logits_4a, logits_4d


def inception_v1_model(features, labels, mode):
    """
    构建一个Inception模型，包括第一构建层 到 第五构建层。

    我们把Inception看成由8个构建层组成，每个构建层都是由一个或多个普通卷积层、池化层、或者Inception模块组成。

    @param feautres: 输入的ImageNet样本图片，形状为[-1， 224, 224, 3]。
    @param labels: 样本数据的标签。
    @param mode: 模型训练所处的模式。

    @Returns: 网络模型。
    """

    net, logits_4a, logits_4d = inception_v1_base(features)

    # 5b 之后的层。全局平均池化和全连接层
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
        # 在训练阶段，旁路分类器的结果乘以权重 0.3 增加到主干的分类结果中
        logits += (logits_4a * 0.3 + logits_4d * 0.3)

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


def inception_train():
    """
    训练一个Inception模型。
    """
    inception_classifier = tf.estimator.Estimator(
        model_fn=inception_v1_model, model_dir="./tmp/inception_v1/")

    # 开始Inception模型的训练
    inception_classifier.train(
        input_fn=lambda: input_fn(True, './data/imagenet/', 128),
        steps=2000)

    # 评估模型并输出结果
    eval_results = inception_classifier.evaluate(
        input_fn=lambda: input_fn(False, './data/imagenet/', 12))
    print("\n识别准确率: {:.2f}%\n".format(eval_results['accuracy'] * 100.0))


# 模型训练
inception_train()
