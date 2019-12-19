#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

# 导入依赖模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import tarfile
import urllib.request
import time
import sys

tf.logging.set_verbosity(tf.logging.INFO)


def read_cifar10_data(path=None):
    """
    读取Cifar-10的训练数据和测试数据。
    :param path: 保存Cifar-10的本地文件目录。
    :Returns: 训练集的图片、训练集标签、测试集图片、测试集标签。
    """
    # Cifar-10的官方下载网址，需要下载binary version文件
    url = 'https://www.cs.toronto.edu/~kriz/'
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']

    # 如果没有指定本地文件目录，那么，设置目录为"~/data/cifar10"
    if path is None:
        path = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')

    # 确保相关目录、及其子目录存在
    os.makedirs(path, exist_ok=True)

    # 如果本地文件不存在，那么，从网络上下载Cifar-10数据
    tar_file = os.path.join(path, tar)
    if not os.path.exists(tar_file):
        print("\n文件{}不存在，尝试从网络下载。".format(tar_file))
        # 从网上下载图片数据，并且，保存到本地文件
        img_url = os.path.join(url, tar)
        # 本地文件名称
        img_path = os.path.join(path, tar)
        print("开始下载: {}, 时间：{}".format(
            img_url, time.strftime('%Y-%m-%d %H:%M:%S')))
        # 文件下载进度条，

        def _progress(count, block_size, total_size):
            # 下载完成进度（百分比）
            percentage = float(count * block_size) / float(total_size) * 100.0
            # 下载进度条总共有50个方块组成（已完成的部分用'█'，未完成的用'.'）
            # 根据count的奇偶性，决定最后一个方块是否出现，实现闪烁的效果
            done = int(percentage / 2.0)
            done += (count & 1)
            # 显示进度条，其中'\r'表示在同一行显示（不换行）
            sys.stdout.write('\r[{}{}] 进度：{:.2f} count:{:2d}'.format
                             ('█' * done, '.' * (50 - done), percentage, count))
            sys.stdout.flush()
        # 从网络下载tar文件，并且，回调显示进度条的函数
        urllib.request.urlretrieve(img_url, img_path, _progress)
        print("保存到：{}".format(img_path))
        # 打印一个空行，将下载日志与数据读取日志分隔开
        print("")

    # 从tar.gz文件中读取训练数据和测试数据
    with tarfile.open(tar_file) as tar_object:
        # 每个文件包含10,000个彩色图像和10,000个标签
        # 每个图像的宽度、高度、深度（色彩通道），分别是32、32、3
        fsize = 10000 * (32 * 32 * 3) + 10000

        # 共有6个数据文件（5个训练数据文件、1个测试数据文件）
        buffer = np.zeros(fsize * 6, np.uint8)

        # 从tar.gz文件中读取数据文件的对象
        # -- tar.gz文件中还包含REDME和其他的非数据晚饭吗
        members = [file for file in tar_object if file.name in files]

        # 对数据文件按照名称排序
        # -- 确保按顺序装载数据文件
        # -- 确保测试数据最后加载
        members.sort(key=lambda member: member.name)

        # 从tar.gz文件中读取数据文件的的内容（解压）
        # 读取文件开始，增加空行隔开日志，更清晰
        print()
        for i, member in enumerate(members):
            # 得到tar.gz中的数据文件对象
            f = tar_object.extractfile(member)
            print("正在读取 {} 中的数据……".format(member.name))
            # 从数据文件对象中读取数据到缓冲区，按照字节读取
            buffer[i * fsize:(i + 1) *
                   fsize] = np.frombuffer(f.read(), np.ubyte)
        # 读取文件结束，增加空行隔开日志
        print()

    # 解析缓冲区数据
    # -- 样本数据是按数据块存储的，每个数据块有3073个字节长
    # -- 每个数据块的第一个字节是标签
    # -- 紧接着的3072个字节的图像数据（32 * 32 * 3 = 3,072）

    # 将每个数据块的第一个字节取出来，形成标签列表
    # 从第0个字节开始，将每隔3073个字节的数据取出来形成标签
    # 对应的字节索引为0×3073, 1×3073, 2×3073, 3×3073, 4×3073……
    labels = buffer[::3073]

    # 将标签数据删除，之后，剩下的全部是图像数据
    pixels = np.delete(buffer, np.arange(0, buffer.size, 3073))
    # 对图像数据进行归一化处理（除以255）
    images = pixels.reshape(-1, 3072).astype(np.float32)

    # 将样本数据切分成训练数据和测试数据
    # 第0个至第50,000个用作训练数据，从第50,000个开始的用作测试数据（共10,000个）
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    return train_images, train_labels.astype(np.int32), \
        test_images, test_labels.astype(np.int32)


def cifar10_model(features, labels, mode):
    """    创建CIFAR10图像识别模型
    :param features: 输入的特征列表，这里只有一个输入特征张量"x"，代表输入的图像
    :param labels: 输出的特征列表，这里是图像所述的类别
    :param mode: 模式，是训练状态还是评估状态
    """

    # （1） 定义输入张量
    # 输入层张量，[batch_size, height, weight, depth]
    # batch_size等于-1代表重整为实际输入的训练数据个数
    # CIFAR10的图像格式为[height, weight, depth] = [32, 32, 3]
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

    # （2） 构建模型（卷积神经网络）
    # 第一个卷积层,直接接受输入层（输入的原始图像数据）
    # 过滤器个数Filter_count = 32, 过滤器大小 Filter_size: 5×5
    # 请注意：过滤器的深度总是与输入张量的深度保持一致，本例中Filter_depth = 3
    # 填充方式"same", 表示按照卷积之后图像保持原状来填充。另外一种填充方式"valid"
    # 过滤器的激活函数采用tf.nn.relu的方式
    # 本层的输出是形状为32×32×64的数据长方体
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # 第一个池化层，接收conv1的输出作为本层的输入
    # 采用最大化池化方法， 池化过滤器尺寸 3×3， 步长为2，这样实现重叠池化
    # 在这种情况下，填充的层数必然是单层，因为输出的数据长方体的尺寸必须满足公式：
    # Output_size = ceil(input_size / stride)
    # 本层输出的数据长方体为16×16×64
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[
                                    3, 3], strides=2, padding='SAME')

    # 第二个卷积层和池化层，从第一个池化层接受输入
    # 过滤器个数64个，尺寸5×5, 填充方式为保持图像不变，激活函数relu
    # 本层输出的数据长方体是16×16×64
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # 第二个池化层，从第二个卷积层接收输入
    # 本层输出的数据长方体是8×8×64
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[
                                    3, 3], strides=2, padding='SAME')

    # 将第二个池化层的输出展平，以方便与后面的全连接层连接
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

    # 全连接层，接受第二个池化层展平后的结果作为输入
    # 共有1024个神经元、激活函数tf.nn.relu
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Dropout层，提高模型的健壮性
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.1, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # 输出层，
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # (为 PREDICT 和 EVAL 模式)生成预测值
        "classes": tf.argmax(input=logits, axis=1),
        # 将 `softmax_tensor` 添加至计算图。用于 PREDICT 模式下的 `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # 如果是评估（测试）模式，那么，执行预测分析
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 计算损失（可用于`训练`和`评价`中）
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # （3）完成模型训练
    # 配置训练操作（用于 TRAIN 模式）
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # 添加评价指标（用于评估）
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cifar10_train():
    """
    模型入口函数。读取训练数据完成模型训练和评估
    """
    # 创建一个卷积神经网络（CNN）的Estimator
    cifar10_classifier = tf.estimator.Estimator(
        model_fn=cifar10_model, model_dir="./tmp/cifar10_convnet_model")

    train_imgs, train_labels, test_imgs, test_labels = read_cifar10_data(
        "./data/")
    # 模型训练的数据输入函数
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_imgs},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    # 开始CIFAR10的模型训练
    cifar10_classifier.train(
        input_fn=train_input_fn,
        steps=20000)

    # 评估模型并输出结果
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_imgs},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn)
    print("\n识别准确率: {:.2f}%\n".format(eval_results['accuracy'] * 100.0))


# 执行测试文件
cifar10_train()
