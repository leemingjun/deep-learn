#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

ds = tf.contrib.distributions
layers = tf.contrib.layers
tfgan = tf.contrib.gan


def generator(noise, mode):
    """
    tf.Estimator 需要一个参数"mode"，指示当前处于训练阶段，还是评估阶段

    @param noise: 代表输入的随机噪音。
    @param mode: 是否训练过程。训练过程中，批量正则化（BN）会更新 beta & gamma 系数；
                在测试过程中，会直接读取以上两个系数。
    @return:  构建好的反卷积层
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    return _generator(noise, weight_decay=2.5e-5, is_training=is_training)


def _generator(noise, weight_decay=2.5e-5, is_training=True):
    """
    构建生成模型(G)的 生成函数。
    输入一个噪音张量，输出与该噪音个数一致的图像。

    生成模型的操作过程与卷积过程正好相反，可以理解为卷积过程的逆过程。
    整个过程是输出尺寸逐渐增加，输出通道数逐渐降低的过程。


    @param noise: 代表输入的随机噪音。
    @param weight_decay: 权重的衰减系数。
    @param is_training: 释放是训练过程。训练过程中，批量正则化（BN）会更新 beta & gamma 系数；
                在测试过程中，会直接读取以上两个系数。
    @return:  构建好的反卷积层
    """
    with tf.contrib.framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)):
        with tf.contrib.framework.arg_scope(
                [layers.batch_norm], is_training=is_training):
            # 第一步，将噪音连接到一个全连接层，神经元个数 1024 个。
            full_conn = layers.fully_connected(noise, 1024)

            # 第二步，连接到第二个全连接层（6272个神经元=7 * 7 * 128）
            full_conn = layers.fully_connected(full_conn, 7 * 7 * 128)

            # 第三步，将第二层的全连接层神经元变形成 特征图谱（Feature Maps）
            # 以便于与后面的 反卷积层（也称转置卷积层） 连接
            deconv2d = tf.reshape(full_conn, [-1, 7, 7, 128])

            # 第四步，将上一步的变形后的特征图谱连接到反卷积层。
            # 与卷积过程相反：输出通道数减半、特征图谱的高度和宽度加倍
            # 输入形状:[7, 7, 128], 输出形状：[14, 14, 64],
            deconv2d = layers.conv2d_transpose(deconv2d, 64, [4, 4], stride=2)
            # 第五步，再增加一个反卷积层
            # 输入形状:[14, 14, 64],输出形状：[28, 28, 32],
            deconv2d = layers.conv2d_transpose(deconv2d, 32, [4, 4], stride=2)

            # 注意：此步骤不可缺少。
            # 随机噪音是采用均值为0、方差为1的随机数生成的，
            # 因此，我们需要将生成模型生成的数据映射回[-1, 1]的取值空间。
            conv2d = layers.conv2d(
                deconv2d, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

            return conv2d


def conv2d(input_layer, output_channels, filter_size=[4, 4], stride=2,
           activation_fn=tf.nn.leaky_relu, normalizer_fn=None,
           weight_decay=2.5e-5, name="conv2d"):
    """
    构建卷积层。

    @param input_layer: 代表输入张量
    @param output_channels: 输出通道数
    @param filter_size: 一个二维数组，过滤器（卷积核）的空间尺寸（高度、宽度）。
    @param stride: 整形，代表过滤器的步长。
    @param activation_fn: 激活函数，默认采用采用ReLU激活函数。
    @param weight_decay: 权重的衰减系数。
    @param is_training: 释放是训练过程。训练过程中，批量正则化（BN）会更新 beta & gamma 系数；
    在测试过程中，会直接读取以上两个系数。
    @param name: 本层的名称。
    @return:  构建好的卷积层
    """
    with tf.variable_scope(name):
        # 构建一个卷积层
        conv = layers.conv2d(
            input_layer, output_channels, filter_size=[4, 4], stride=2,
            activation_fn=tf.nn.leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay),
            name="conv2d")

        return conv


def discriminator(img, weight_decay=2.5e-5):
    """
    构建辨别模型（D）。

    输入一个图片数据，输出该图片属于目标类别（样本、生成）的可能性。

    只采用卷及操作，不采用池化操作，为此，将卷积的步长设置为2，以便于实现降采样。
    卷积层每增加一层，高度和宽度缩减到原来的1/2，输出通道数增加到原来的2倍。

    @param img: 输入的图片数据。形状为[-1, 28, 28, 1]

    @return:  该辨别模型的输出结果、以及构建好的辨别模型。
    """

    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=_leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        # 输入形状 [28, 28, 1], 输出形状： [14 ,14 , 64]
        conv2d = layers.conv2d(img, 64, [4, 4], stride=2)

        # 输入形状 [14 ,14 , 64], 输出形状： [7 ,7 , 128]
        conv2d = layers.conv2d(img, 128, [4, 4], stride=2)

        # 展平，以便于与后面的全连接层连接
        full_conn = layers.flatten(conv2d)

        # 与后面的一个包含 1024 神经元连接。
        full_conn = layers.fully_connected(
            full_conn, 1024, normalizer_fn=layers.layer_norm)

        # 与分类器连接
        return layers.linear(full_conn, 1)


def _leaky_relu(x):
    """
    带泄露的线性整流函数

    @param x: 输入的张量
    @return:  按照alpha为0.01线性整流之后的结果。
    """
    return tf.nn.leaky_relu(x, alpha=0.01)
