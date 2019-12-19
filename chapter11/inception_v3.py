#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

# 导入依赖模块
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


def conv2d(inputs, filters, kernel_size=[3, 3], strides=(1, 1),
           stddev=0.01, padding='SAME', name='conv2d'):
    """
    定义Inception中默认的卷积函数

    @param input_layer: 输入层。
    @param strides: 步长。
    @param padding: 填充方式。常用的有“SAME”和“VALID”两种模式。
    @param weights_initializer: 填充方式。常用的有“SAME”和“VALID”两种模式。

    @Returns: 图片和图片的标签。图片是以张量形式保存的。
    """
    with tf.variable_scope(name):
        weights_initializer = trunc_normal(stddev)

        return tf.layers.conv2d(
            inputs, filters, kernel_size=kernel_size, strides=strides,
            padding=padding, kernel_initializer=weights_initializer,
            name=name)


def max_pool2d(inputs, pool_size=(3, 3), strides=(2, 2),
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
            inputs, pool_size, strides, padding, name=scope)


def depth(d, min_depth=16, depth_multiplier=1.0):
    """
    对输出通道数进行约束。
    计算输出通道数，避免设置的输出通道数，乘以通道数扩张因子（可能<1）之后，输出到通道数过低
    如果，经过通道扩张因子计算之后，输出通道数小于最小的输出通道数，以最小的通道数为准

    @param d: 建议的输出通道数
    @param min_depth: 最小的输出通道数
    @param depth_multiplier: 通道扩张因子。0 < depth_multiplier < 1时，限制输出通道数;
        depth_multiplier > 1时，不起作用。

    @Returns: 经过约束的输出通道数。
    """
    return max(int(d * depth_multiplier), min_depth)


def inception_v3_a(net, filters, strides=(1, 1),
                   scope='inception_v3_a'):
    """
    主要用于 35x35 的特征图谱。

    对应“图11-2（a）”所示架构。

    共有四个分支：
    Branch_0：1x1 卷积
    Branch_1：1x1 卷积 --> 5x5 卷积
    Branch_2：1x1 卷积 --> 3x3 卷积 --> 3x3 卷积
    Branch_3：3x3 池化 --> 1x1 卷积

    @param net: 输入张量。
    @param filters: Inception模块中，各个分支中，过滤器的输出通道数。
        filters[0],代表分支(Branch_0) 的 1x1 卷积的输出通道数

        filters[1],代表分支(Branch_1) 的 1x1 卷积的输出通道数
        filters[2],代表分支(Branch_1) 的 5x5 卷积的输出通道数

        filters[3],代表分支(Branch_2) 的 1x1 卷积的输出通道数
        filters[4],代表分支(Branch_2) 的 3x3 卷积的输出通道数
        filters[5],代表分支(Branch_2) 的 3x3 卷积的输出通道数

        filters[6],代表分支(Branch_3) 的 1x1 卷积的输出通道数
    @param strides: 本Inception模块中所有的卷积、池化操作的步长。
    @param scope: 变量所属的作用域

    @Returns: 经过Inception模块卷积后的张量。
    """
    with tf.variable_scope(scope):

        # Branch_0：1x1 卷积
        with tf.variable_scope('Branch_0'):
            branch_0 = conv2d(
                net, depth(filters[0]), [1, 1], name='Conv2d_0a_1x1')

        # Branch_1：1x1 卷积 --> 5x5 卷积
        with tf.variable_scope('Branch_1'):
            branch_1 = conv2d(
                net, depth(filters[1]), [1, 1], name='Conv2d_0b_1x1')
            branch_1 = conv2d(
                branch_1, depth(filters[2]), [5, 5],
                name='Conv_1_0c_5x5')

        # Branch_2：1x1 卷积 --> 3x3 卷积 --> 3x3 卷积
        with tf.variable_scope('Branch_2'):
            branch_2 = conv2d(
                net, depth(filters[3]), [1, 1],
                name='Conv2d_0a_1x1')
            branch_2 = conv2d(
                branch_2, depth(filters[4]), [3, 3],
                name='Conv2d_0b_3x3')
            branch_2 = conv2d(
                branch_2, depth(filters[5]), [3, 3],
                name='Conv2d_0c_3x3')

        # Branch_3：3x3 池化 --> 1x1 卷积
        with tf.variable_scope('Branch_3'):
            branch_3 = tf.layers.average_pooling2d(
                net, [3, 3], strides=(1, 1),
                padding='SAME', name='AvgPool_0a_3x3')
            branch_3 = conv2d(
                branch_3, depth(filters[6]), [1, 1],
                name='Conv2d_0b_1x1')

        # 将所有分支的输出，沿着深度串联起来起来
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    return net


def inception_v3_b_downsample(net, filters, strides=(2, 2),
                              scope='inception_v3_b'):
    """
    并行卷积，实现降采样，作用类似于 池化层。

    对应“图11-4（a）”所示架构。典型的特点是步长为（2, 2）。用在
    Inception v3的第六个构建层的开头，从第五个构建块接受输入，降采样之后，
    输出给第六个构建层。

    共有三个分支：
    Branch_0 ：1x1 卷积 --> 3x3 卷积 --> 3x3 卷积、步长为2
    Branch_1 ：1x1 卷积 --> 3x3 卷积、步长为2
    Branch_2 ：3x3 最大池化、步长为2

    @param net: 输入张量。
    @param filters: Inception模块中，各个分支中，过滤器的输出通道数。
        filters[0],代表分支(Branch_0) 的 1x1 卷积的输出通道数
        filters[1],代表分支(Branch_0) 的 3x3 卷积的输出通道数
        filters[2],代表分支(Branch_0) 的 3x3 卷积的输出通道数

        filters[3],代表分支(Branch_1) 的 3x3 卷积的输出通道数
    @param strides: 本Inception模块中卷积、池化操作的默认步长。
    @param scope: 变量所属的作用域

    @Returns: 经过Inception模块卷积后的张量。
    """
    with tf.variable_scope(scope):

        # Branch_0：1x1 卷积 --> 3x3 卷积 --> 3x3 卷积、步长为2
        with tf.variable_scope('Branch_0'):
            branch_0 = conv2d(
                net, depth(filters[0]), [1, 1], name='Conv2d_0a_1x1')
            branch_0 = conv2d(
                branch_0, depth(filters[1]), [3, 3],
                name='Conv2d_0b_3x3')
            branch_0 = conv2d(
                branch_0, depth(filters[2]), [3, 3], strides=strides,
                padding='VALID', name='Conv2d_1a_1x1')

        # Branch_1：1x1 卷积 --> 3x3 卷积、步长为2
        with tf.variable_scope('Branch_1'):
            branch_1 = conv2d(
                net, depth(filters[3]), [3, 3], strides=strides,
                padding='VALID', name='Conv2d_1a_1x1')

        # Branch_2：3x3 max_pool、步长为2
        with tf.variable_scope('Branch_2'):
            branch_2 = max_pool2d(
                net, [3, 3], strides=strides, padding='VALID',
                scope='MaxPool_1a_3x3')

        # 将所有的分支串联起来
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

    return net


def inception_v3_b(net, filters, strides=(1, 1), scope='inception_v3_b'):
    """
    主要用于 17x17 的特征图谱。

    非对称卷积，例如 1x7-->7x1，或者 7x1-->1x7-->7x1-->1x7
    对应“图11-2（b）”所示架构。

    共有四个分支：
    Branch_0 ：3x3 池化 --> 1x1 卷积
    Branch_1 ：1x1 卷积
    Branch_2 ：1x1 卷积 --> 1x7 卷积 --> 7x1 卷积
    Branch_3 ：1x1 卷积 --> 1x7 卷积 --> 7x1 卷积 --> 1x7 卷积 --> 7x1 卷积


    @param net: 输入张量。
    @param filters: Inception模块中，各个分支中，过滤器的输出通道数。
        filters[0],代表分支(Branch_0) 的 1x1 卷积的输出通道数

        filters[1],代表分支(Branch_1) 的 1x1 卷积的输出通道数

        filters[2],代表分支(Branch_2) 的 1x1 卷积的输出通道数
        filters[3],代表分支(Branch_2) 的 1x7 卷积的输出通道数
        filters[4],代表分支(Branch_2) 的 7x1 卷积的输出通道数

        filters[5],代表分支(Branch_3) 的 1x1 卷积的输出通道数
        filters[6],代表分支(Branch_3) 的 7x1 卷积的输出通道数
        filters[7],代表分支(Branch_3) 的 1x7 卷积的输出通道数
        filters[8],代表分支(Branch_3) 的 7x1 卷积的输出通道数
        filters[9],代表分支(Branch_3) 的 1x7 卷积的输出通道数

    @param strides: 本Inception模块中卷积、池化操作的默认步长。
    @param scope: 变量所属的作用域

    @Returns: 经过Inception模块卷积后的张量。
    """
    with tf.variable_scope(scope):

        # Branch_0 ：3x3 池化 --> 1x1 卷积
        with tf.variable_scope('Branch_0'):
            branch_0 = tf.layers.average_pooling2d(
                net, [3, 3], strides=strides,
                padding='SAME', name='AvgPool_0a_3x3')
            branch_0 = conv2d(branch_0, depth(filters[0]), [1, 1],
                              name='Conv2d_0b_1x1')

        # Branch_1 ：1x1 卷积
        with tf.variable_scope('Branch_1'):
            branch_1 = conv2d(
                net, depth(filters[1]), [1, 1], name='Conv2d_0a_1x1')

        # Branch_2 ：1x1 卷积 --> 1x7 卷积 --> 7x1 卷积
        with tf.variable_scope('Branch_2'):
            branch_2 = conv2d(
                net, depth(filters[2]), [1, 1], name='Conv2d_0a_1x1')
            branch_2 = conv2d(
                branch_2, depth(filters[3]), [1, 7], name='Conv2d_0b_1x7')
            branch_2 = conv2d(
                branch_2, depth(filters[4]), [7, 1], name='Conv2d_0c_7x1')

        # Branch_3 ：1x1 卷积 --> 1x7 卷积 --> 7x1 卷积 --> 1x7 卷积 --> 7x1 卷积
        with tf.variable_scope('Branch_3'):
            branch_3 = conv2d(
                net, depth(filters[5]), [1, 1], name='Conv2d_0a_1x1')
            branch_3 = conv2d(
                branch_3, depth(filters[6]), [7, 1], name='Conv2d_0b_7x1')
            branch_3 = conv2d(
                branch_3, depth(filters[7]), [1, 7], name='Conv2d_0c_1x7')
            branch_3 = conv2d(
                branch_3, depth(filters[8]), [7, 1], name='Conv2d_0d_7x1')
            branch_3 = conv2d(
                branch_3, depth(filters[9]), [1, 7], name='Conv2d_0e_1x7')

        # 将所有的分支串联起来,沿着深度维
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    return net


def inception_v3_c_downsample(net, filters, strides=(2, 2),
                              scope='inception_v3_b'):
    """
    作用于 17x17 的特征图谱, 降采样成为 8x8 的特征图谱


    共有三个分支：
    Branch_0 ：1x1 卷积 --> 3x3 卷积、步长为2
    Branch_1 ：1x1 卷积 --> 1x7 卷积 --> 7x1 卷积 --> 3x3 卷积、步长为2
    Branch_2 ：3x3 池化、步长为2

    @param net: 输入张量。
    @param filters: Inception模块中，各个分支中，过滤器的输出通道数。
        filters[0],代表分支(Branch_0) 的 1x1 卷积的输出通道数
        filters[1],代表分支(Branch_0) 的 1x1 卷积的输出通道数

        filters[2],代表分支(Branch_1) 的 1x1 卷积的输出通道数
        filters[3],代表分支(Branch_1) 的 1x7 卷积的输出通道数
        filters[4],代表分支(Branch_1) 的 7x1 卷积的输出通道数
        filters[5],代表分支(Branch_1) 的 3x3 卷积的输出通道数

    @param strides: 本Inception模块中卷积、池化操作的默认步长。
    @param scope: 变量所属的作用域

    @Returns: 经过Inception模块卷积后的张量。
    """
    with tf.variable_scope(scope):

        # Branch_0 ：1x1 卷积 - -> 3x3 卷积、步长为2
        with tf.variable_scope('Branch_0'):
            branch_0 = conv2d(
                net, depth(filters[0]), [1, 1], name='Conv2d_0a_1x1')
            branch_0 = conv2d(
                branch_0, depth(filters[1]), [3, 3], strides=strides,
                padding='VALID', name='Conv2d_1a_3x3')

        # Branch_1 ：1x1 卷积 - -> 1x7 卷积 - -> 7x1 卷积 - -> 3x3 卷积、步长为2
        with tf.variable_scope('Branch_1'):
            branch_1 = conv2d(
                net, depth(filters[2]), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = conv2d(
                branch_1, depth(filters[3]), [1, 7], name='Conv2d_0b_1x7')
            branch_1 = conv2d(
                branch_1, depth(filters[4]), [7, 1], name='Conv2d_0c_7x1')
            branch_1 = conv2d(
                branch_1, depth(filters[5]), [3, 3], strides=strides,
                padding='VALID', name='Conv2d_1a_3x3')

        # Branch_2 ：3x3 池化、步长为2
        with tf.variable_scope('Branch_2'):
            branch_2 = max_pool2d(
                net, [3, 3], strides=strides, padding='VALID',
                scope='MaxPool_1a_3x3')

        # 将所有的分支串联起来,沿着深度维
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

    return net


def inception_v3_c(net, filters, strides=(1, 1),
                   scope='inception_v3_c'):
    """
    主要用于 8 x 8 的特征图谱。

    对应“图11-8”所示架构。

    共有四个分支：
    Branch_0 ：3x3 池化 --> 1x1 卷积
    Branch_1 ：1x1 卷积
    Branch_2 ：1x1 卷积 --> 1x3 卷积  （分支卷积）
                       --> 3x1 卷积  （分支卷积）
    Branch_3 ：1x1 卷积 --> 3x3 卷积  --> 1x3 卷积（分支卷积）
                                    --> 3x1 卷积（分支卷积）


    @param net: 输入张量。
    @param filters: Inception模块中，各个分支中，过滤器的输出通道数。
        filters[0],代表分支(Branch_0) 的 1x1 卷积的输出通道数

        filters[1],代表分支(Branch_1) 的 1x1 卷积的输出通道数

        filters[2],代表分支(Branch_2) 的 1x1 卷积的输出通道数
        filters[3],代表分支(Branch_2) 的 1x3 卷积的输出通道数
        filters[4],代表分支(Branch_2) 的 3x1 卷积的输出通道数

        filters[5],代表分支(Branch_3) 的 1x1 卷积的输出通道数
        filters[6],代表分支(Branch_3) 的 3x3 卷积的输出通道数
        filters[7],代表分支(Branch_3) 的 1x3 卷积的输出通道数
        filters[8],代表分支(Branch_3) 的 3x1 卷积的输出通道数

    @param strides: 本Inception模块中卷积、池化操作的默认步长。
    @param scope: 变量所属的作用域

    @Returns: 经过Inception模块卷积后的张量。
    """
    with tf.variable_scope(scope):

        # Branch_0 ：3x3 池化 - -> 1x1 卷积
        with tf.variable_scope('Branch_0'):
            branch_0 = tf.layers.average_pooling2d(
                net, [3, 3], strides=strides,
                padding='SAME', name='AvgPool_0a_3x3')
            branch_0 = conv2d(
                branch_0, depth(filters[0]), [1, 1], name='Conv2d_0b_1x1')

        #  Branch_1 ：1x1 卷积
        with tf.variable_scope('Branch_1'):
            branch_1 = conv2d(
                net, depth(filters[1]), strides=strides, name='Conv2d_0a_1x1')

        # Branch_2 ：1x1 卷积 -> 1x3 卷积  （分支卷积）
        #                    -> 3x1 卷积  （分支卷积）
        with tf.variable_scope('Branch_2'):
            branch_2 = conv2d(
                net, depth(filters[2]), strides=strides, name='Conv2d_0a_1x1')

            # 将 1x3 卷积 和 3x1 卷积的结果串联起来
            branch_2 = tf.concat(axis=3, values=[
                conv2d(branch_2, depth(filters[3]), [1, 3],
                       name='Conv2d_0b_1x3'),
                conv2d(branch_2, depth(filters[4]), [3, 1],
                       name='Conv2d_0b_3x1')])

        # Branch_3 ：1x1 卷积 -> 3x3 卷积 -> 1x3 卷积（分支卷积）
        #                               -> 3x1 卷积（分支卷积）
        with tf.variable_scope('Branch_3'):
            branch_3 = conv2d(
                net, depth(filters[5]),
                strides=strides, name='Conv2d_0a_1x1')
            branch_3 = conv2d(
                branch_3, depth(filters[6]), [3, 3], name='Conv2d_0b_3x3')

            # 将 1x3 卷积 和 3x1 卷积的结果串联起来
            branch_3 = tf.concat(axis=3, values=[
                conv2d(branch_3, depth(filters[7]), [1, 3],
                       name='Conv2d_0c_1x3'),
                conv2d(branch_3, depth(filters[8]), [3, 1],
                       name='Conv2d_0d_3x1')])

        # 将所有的分支串联起来
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    return net


def inception_v3_base(inputs,
                      depth_multiplier=1.0,
                      scope=None):
    """
    本模型详细情况请参考：http://arxiv.org/abs/1512.00567.

    构建以Inception V3的模型，此函数包含模型的前六个构建层。

    请注意，本模型虽然参考了原始的论文中的模型架构，本模型尽可能的与原始的模型架构一致，但是，
    二者并不完全是一一对应，包括Inception模块的各个分支的顺序（从左到右）、以及模型最后的
    全连接部分、分类预测部分等。

    @param inputs: 输入张量，代表ImageNet中的图片，形状为 [batch_size, 299, 299, 3]。
    @param min_depth: 最小的输出通道数，针对所有的卷及操作。当 depth_multiplier < 1 时，
        起到避免输出通道过少的作用； 当 depth_multiplier >= 1 时，不会激活。
    @param depth_multiplier: 输出通道的扩张因子，针对所有的卷积操作。该值必须大于0。
        一般的，该参数的取值区间设置为(0, 1)，用于控制模型的参数数量。
    @param scope: 可选项。变量的作用域。

    @Returns: 构建好的Inception v3模型。
    """
    # 检查输出通道扩张因子.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier 必须大于0.')

    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        # 第一构建层，构建层中的第一层往往是池化层、或者步长为2的卷积层
        # 输入：299 x 299 x 3
        net = conv2d(inputs, depth(32), [3, 3],
                     strides=(2, 2), name='Conv2d_1a_3x3')

        # 第二构建层
        # 149 x 149 x 32
        net = conv2d(net, depth(32), [3, 3], name='Conv2d_2a_3x3')

        # 147 x 147 x 32
        net = conv2d(net, depth(64), [3, 3], name='Conv2d_2b_3x3')

        # 第三构建层，一般来说，池化层放在一个构建层的开头。
        # 147 x 147 x 64
        net = max_pool2d(net, [3, 3],  strides=(2, 2), scope='MaxPool_3a_3x3')

        # 73 x 73 x 64
        net = conv2d(net, depth(80), [1, 1], name='Conv2d_3b_1x1')

        # 第四构建层，一般来说，池化层放在一个构建层的开头。
        # 73 x 73 x 80.
        net = conv2d(net, depth(192), [3, 3], name='Conv2d_4a_3x3')

        # 第五构建层。包含一个池化层、三个 inception_v3_a 模块
        # 71 x 71 x 192.
        net = max_pool2d(net, [3, 3], strides=(2, 2), scope='MaxPool_5a_3x3')

        # 35 x 35 x 192.
        net = inception_v3_a(
            net, filters=[64, 48, 64, 64, 96, 96, 32], scope='mixed_5b')
        # 35 x 35 x 256.
        net = inception_v3_a(
            net, filters=[64, 48, 64, 64, 96, 96, 64], scope='mixed_5c')
        # 35 x 35 x 288.
        net = inception_v3_a(
            net, filters=[64, 48, 64, 64, 96, 96, 64], scope='mixed_5d')

        # 第六构建层，包含一个 inception_v3_b_downsample 层和4个inception_v3_b模块层
        #   17 x 17 x 768.
        net = inception_v3_b_downsample(
            net, filters=[64, 96, 96, 384], scope='Mixed_6a')

        #  17 x 17 x 768.
        net = inception_v3_b(
            net, filters=[192, 192, 128, 128, 192, 128, 128, 128, 128, 192],
            scope='Mixed_6b')

        #   17 x 17 x 768.
        net = inception_v3_a(
            net, filters=[192, 192, 160, 160, 192, 160, 160, 160, 160, 192],
            scope='Mixed_6c')

        #  17 x 17 x 768.
        net = inception_v3_b(
            net, filters=[192, 192, 160, 160, 192, 160, 160, 160, 160, 192],
            scope='Mixed_6d')

        #   17 x 17 x 768.
        net = inception_v3_a(
            net, filters=[192, 192, 192, 192, 192, 192, 192, 192, 192, 192],
            scope='Mixed_6e')

        # 第七构建层，包含一个 inception_v3_c_downsample 层和 2个 inception_v3_c 模块层
        # 8 x 8 x 1280.
        net = inception_v3_c_downsample(
            net, filters=[192, 320, 192, 192, 192, 192],
            scope='Mixed_7a')

        # 8 x 8 x 2048.
        net = inception_v3_c(
            net, filters=[192, 320, 384, 384, 384, 448, 384, 384, 384],
            scope='Mixed_7b')

        # 8 x 8 x 2048.
        net = inception_v3_c(
            net, filters=[192, 320, 384, 384, 384, 448, 384, 384, 384],
            scope='Mixed_7c')

    return net


def inception_v3_model(features, labels, mode):
    """
    本模型的详细情况，请参考：http://arxiv.org/abs/1512.00567。

    构建以Inception V3的模型，从原始图像输入到最终的模型输出。

    请注意，本模型虽然参考了原始的论文中的模型架构，本模型尽可能的与原始的模型架构一致，但是，
    二者并不完全是一一对应，包括Inception模块的各个分支的顺序（从左到右）、以及模型最后的
    全连接部分、分类预测部分等。

    阅读源代码时要注意以上问题。

    @param features: 输入张量，代表ImageNet中的图片，形状为 [batch_size, 299, 299, 3]。
    @param labels: 图片对应的类别，ImageNet默认有1000个类别。
    @param mode: 训练模式，对于训练过程、还是推理过程 或 评价过程。

    @Returns: 构建好的Inception v3模型。
    """

    # 构建 Inception v3 的基本网络结构
    net = inception_v3_base(features)

    # 网络尾部的全局池化
    with tf.variable_scope('Logits'):
        # 全局池化层
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='GlobalPool')

    # 1 x 1 x 2048
    dropout_rate = 0.7
    net = tf.layers.dropout(net, rate=dropout_rate,
                            name='Dropout_1b')
    # 第七构建层，包括一个线性转换层和softmax分类层
    # 2048
    net = tf.layers.flatten(net)
    # 全连接层，共有1000个类别的
    logits = tf.layers.dense(
        inputs=net, units=1000, activation=None, name='FC_1000')


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


def inception_v3_train():
    """
    训练一个 inception_v3 模型
    """
    inception_classifier = tf.estimator.Estimator(
        model_fn=inception_v3_model, model_dir="./tmp/inception_v3/")

    # 开始Inception模型的训练
    inception_classifier.train(
        input_fn=lambda: input_fn(True, './data/imagenet/', 128),
        steps=2000)

    # 评估模型并输出结果
    eval_results = inception_classifier.evaluate(
        input_fn=lambda: input_fn(False, './data/imagenet/', 12))
    print("\n识别准确率: {:.2f}%\n".format(eval_results['accuracy'] * 100.0))


# 模型训练
inception_v3_train()
