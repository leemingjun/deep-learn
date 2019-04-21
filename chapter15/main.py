from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import struct
import numpy as np
import gzip
import time

import scipy.misc
from six.moves import xrange
import gan_networks as networks
import tensorflow as tf
tfgan = tf.contrib.gan

# 每训练批次中包含图像的个数.
batch_size = 64
#  最大的训练步数.
max_number_of_steps = 20000
# 生成噪音的个数
noise_dims = 64
# GAN网络生成的图片保存的路径.
eval_dir = './logs/mnist/estimator/'

# 全局变量。保存图片的静态常量
images = None

# 设置日志级别。
tf.logging.set_verbosity(tf.logging.INFO)

'''
读取MNIST数据文件。

@param path: 本地MNIST数据文件所在的路径。
@param data_type: 要读取的数据文件类型，包括"train"和"t10k"两种。
@Returns: 图片和图片的标签。图片是以张量形式保存的。
'''


def read_mnist_data(path='./data/mnist', data_type="train"):
    global images
    if images is None:
        img_path = os.path.join(path, ('%s-images-idx3-ubyte.gz' % data_type))

        # 使用gzip读取图片数据文件
        print("\n读取文件：%s" % img_path)
        with gzip.open(img_path, 'rb') as img_file:
            # 按照大端在前（big-endian）读取四个32位的整数，所以，总共读取16个字节
            magic, n_imgs, n_rows, n_cols = struct.unpack(
                ">IIII", img_file.read(16))
            # 分别是magic number、n_imgs(图片的个数)、图片的行列的像素个数
            # （n_rows, n_cols ）
            print("magic number：%d，期望图片个数：%d个" % (magic, n_imgs))
            print("图片长宽：%d × %d 个像素" % (n_rows, n_cols))

            # 读取剩下所有的数据，按照 labels * 784 重整形状
            # 其中 784 = 28 × 28 × 1（长×宽×深度）
            images = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(
                n_imgs, n_rows, n_cols, 1)
            print("实际读取到的图片：%d 个" % len(images))

        # 数据集中的数据保存的是uint8，取值范围是[-128, 127]，将其映射回[-1, 1)的取值范围
        images = (images.astype(np.float32) - 128.0) / 128.0
        # Labels的数据类型必须转换成为int32
        noise = tf.random_normal([len(images[0]), noise_dims])

    return images, noise


"""
读取训练数据。
@param noise_dims: 噪音的个数。
@param dataset_dir: 样本数据所在的本地路径。
@Returns: 所有的MNIST数据集中的图片、以及与图片个数相等的噪音。
"""


def get_train_data_fn(batch_size, noise_dims, dataset_dir='./data/mnist/',
                      num_threads=4):
    def train_input_fn():
        with tf.device('/cpu:0'):
            images_all, noise = read_mnist_data(dataset_dir)
        # 将图像数据包装成批次图片数据，用于训练
        images, noise = tf.train.shuffle_batch(
            [images_all[0], noise[0]],
            batch_size=batch_size,
            num_threads=4,
            capacity=10000,
            min_after_dequeue=1000
        )
        # 生成一个批次的噪音数据

        return noise, images
    return train_input_fn


"""
生成评价所需要的噪音数据。

@param batch_size: 一个批次的大小。
@param noise_dims: 噪音的个数。
@Returns: 生成的噪音数据。
"""


def get_eval_data_fn(batch_size, noise_dims):
    def predict_input_fn():
        noise = tf.random_normal([batch_size, noise_dims])
        return noise
    return predict_input_fn


"""
GAN模型的生成和训练。

@param batch_size: 一个批次的大小。
@param noise_dims: 噪音的个数。
@Returns: 生成的噪音数据。
"""


def main(need_trainning=True):
    # 确保 相关的文件路径 存在
    os.makedirs("./logs/model/mnist", exist_ok=True)
    # 创建GANEstimator
    gan_estimator = tfgan.estimator.GANEstimator(
        # 保存模型训练的过程
        model_dir="./logs/model/mnist/GANEstimator",
        # 生成模型函数
        generator_fn=networks.generator,
        # 辨别模型函数
        discriminator_fn=networks.discriminator,
        # 生成模型的损失函数
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        # 辨别模型的损失函数
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        # 生成模型的优化器
        generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
        # 辨别模型的优化器
        discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
        # 生成模型生成的数据与样本数据之间的加总方法
        add_summaries=tfgan.estimator.SummaryType.IMAGES,
        # 模型从上一次开始的地方继续训练，制定模型保存的路径
        # 指定该参数的意义在于，可以多次训练模型，每一次都是从上一次开始的地方继续
        warm_start_from="./logs/model/mnist/GANEstimator")

    # 是否需要执行模型训练过程
    if need_trainning:
        # 构建训练数据生成函数
        train_input_fn = get_train_data_fn(batch_size, noise_dims)
        # 训练GAN模型
        gan_estimator.train(train_input_fn, max_steps=max_number_of_steps)

    # 调用生成模型，生成图片。在这里共生成 36 个图片
    predict_input_fn = get_eval_data_fn(36, noise_dims)
    prediction_iterable = gan_estimator.predict(predict_input_fn)
    predictions = [prediction_iterable.__next__() for _ in xrange(36)]

    # 将这些图片并排排列在一起， 每行6个，共6行
    image_rows = [np.concatenate(predictions[i:i+6], axis=0) for i in
                  range(0, 36, 6)]
    tiled_image = np.concatenate(image_rows, axis=1)

    # 将生成好的图片保存起来
    if not tf.gfile.Exists(eval_dir):
        tf.gfile.MakeDirs(eval_dir)
    gan_file = os.path.join(
        eval_dir, time.strftime('%Y%m%d%H%M%S_gan.png'))
    print("\n\n将GAN模型生成的图片保存到： {}\n\n".format(gan_file))
    scipy.misc.imsave(gan_file,
                      np.squeeze(tiled_image, axis=2))


# 是否需要模型训练过程
main(need_trainning=True)
