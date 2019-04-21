#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

# 设置日志级别。
tf.logging.set_verbosity(tf.logging.INFO)

_RESIZE_MIN = 256
DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001

_num_train_files = 1024
_SHUFFLE_BUFFER = 10000


def crop_and_flip(image_buffer, bbox, num_channels):
    """
    随机剪切一个给定的图像的一部分，然后，随机翻转。

    @image_buffer: 一个字符串标量，代表原始的 JPG 图像缓存。
    @bbox: 一个三维的整型张量，形状如[1, num_boxes, coords]，其中，每一个坐标
        都是都是按照[ymin, xmin, ymax, xmax]排列的。
    @num_channels: 整型。要解码的图像通道数。

    @Returns: 剪切后图像的三维张量。
    """

    # 在人工标准对象边界框的基础上，对于边界框进行随机扭曲，扩充样本数据
    # 样本数据扩充方式包括 宽高比、大小
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100)

    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # 按照 剪切操作的要求，对于 对象边界框重新组合
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    # 对图像进行解码和剪切
    cropped = tf.image.decode_and_crop_jpeg(
        image_buffer, crop_window, channels=num_channels)

    # 翻转以便于增加一点点随机扰动
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped


def central_crop(image, crop_height, crop_width):
    """
    在图像的中央区域进行随机剪切。

    @image: 一个三维的图像张量。
    @crop_height: 剪切后图像的高度。
    @crop_width: 剪切后图像的宽度。

    @Returns: 剪切后图像的三维张量。
    """
    shape = tf.shape(input=image)
    # 图像的高度、宽度
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    # 计算除法，取整。
    crop_top = amount_to_be_cropped_h // 2

    amount_to_be_cropped_w = (width - crop_width)
    # 计算除法，取整。
    crop_left = amount_to_be_cropped_w // 2

    # 执行图像剪切
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def mean_subtraction(image,
                     means=[123.68, 116.779, 103.939],
                     num_channels=3):
    """
    从图像的所有通道中减去平均值。

    @image: 张量形状为 [height, width, channels].
    @means: 各个通道的平均值.
    @num_channels: 图像的通道数量。

    @Returns: 减去平均值的图像.

    """
    if image.get_shape().ndims != 3:
        raise ValueError('输入张量的形状必须是：[height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('均值的个数必须与输入通道数量一致。')

    # 将均值的维度扩充到与图像的维度数量一致
    # 图像的是三维的，原来的均值是一维的
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)

    return image - means


def calc_size(height, width, resize_min):
    """
    计算图像新的尺寸，将最小的边设置为最小尺寸，同时，保持图像原始的高宽比。

    @height: 32位的整型。当前的高度。
    @width: 32位的整型。当前的宽度。
    @resize_min: 32位的整型。图像最小的尺寸。

    @Returns:
      new_height: 32位的整型。计算后的高度。
      new_width: 32位的整型。计算后的宽度。
    """

    smaller_dim = tf.minimum(height, width)
    # Python 3中默认按照浮点数计算
    scale_ratio = resize_min / smaller_dim

    # 转换成整型，返回结果
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def resize_h2w_ratio(image, resize_min):
    """
    调整图像大小，同时，保持宽高比。

    @image: 一个三维的张量。
    @resize_min: 调整后图像的最小尺寸。

    @Returns:
      resized_image: 调整后图像的尺寸。
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    new_height, new_width = calc_size(height, width, resize_min)

    # 按照计算好的尺寸对图像进行调整
    resized_image = tf.image.resize(
        image, [new_height, new_width],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False)
    return resized_image


def preprocess_image(image_buffer, bbox, output_height, output_width,
                     num_channels, is_training=False):
    """
    对给定的图像进行预处理。

    预处理过程包括解码、剪切、调整大小等。在训练过程中，除了上述的操作之外，还有一些对图像的
    随机扰动操作，用于提高模型的精度。

    @image_buffer: 一个字符串标量，代表原始的 JPG 图像缓存。
    @bbox: 一个三维的整型张量，形状如[1, num_boxes, coords]，其中，每一个坐标
        都是都是按照[ymin, xmin, ymax, xmax]排列的。
    @output_height: 预处理之后的图像高度。
    @output_width: 预处理之后的图像宽度。
    @num_channels: 整型。要解码的图像通道数。
    @is_training: 布尔值。指示否是训练数据过程，训练过程中，对于图像的处理操作包含随机扰动。

    @Returns: 一个预处理之后的图像.
    """
    if is_training:
        # 训练过程，包含一些随机扰动
        image = crop_and_flip(image_buffer, bbox, num_channels)
        image = tf.image.resize(
            image, [output_height, output_width],
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)
    else:
        # 验证过程，执行解码、大小调整、只包含中间剪切
        image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
        image = resize_h2w_ratio(image, _RESIZE_MIN)
        image = central_crop(image, output_height, output_width)

    image.set_shape([output_height, output_width, num_channels])

    return mean_subtraction(image, num_channels=num_channels)


def parse_record(raw_record, is_training, dtype):
    """
    解析一条包含训练样本图片的记录。输入的原始记录被解析成（image, label）对，然后，图像
    数据被进一步处理（剪切、抖动等等）

    @raw_record: 字符串，原始的 TFRecord 文件名称。
    @is_training: 布尔值。指示否是训练数据集。
    @dtype: 图像或者特征的数据类型。

    @Returns: 元组，包含一个处理后的图像张量、一个 on-hot-encoded 的标签张量。
    """

    # 解析原始图像格式, 返回图像的张量、标签、边界框列表
    image_buffer, label, bbox = parse_example_tfrecord(raw_record)

    image = preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=DEFAULT_IMAGE_SIZE,
        output_width=DEFAULT_IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
        is_training=is_training)

    # 将图像数据转换成指定的数据类型
    image = tf.cast(image, dtype)

    return image, label


def parse_example_tfrecord(tfrecord_serialized):
    """
    解析一条 TFRecord 记录，该记录包含一个训练样本的图片。
    此函数读取 process_imagenet_data.py 生产并保存的 TFRecord 结果集。

    @example_serialized: 一个标量，包含按照 protocol buffer 协议序列化的字符串。
    @Returns:
      image_buffer:  张量，一个内容 JPG 图像的序列化字符串。
      label: 一个整型，包含类别的标签。
      bbox: 一个三维的整型张量，形状如[1, num_boxes, coords]，其中，每一个坐标
        都是都是按照[ymin, xmin, ymax, xmax]排列的。
    """
    sparse_int64 = tf.io.VarLenFeature(dtype=tf.int64)
    feature_map = {
        'image/height': tf.io.FixedLenFeature([], dtype=tf.int64),
        'image/width': tf.io.FixedLenFeature([], dtype=tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                   default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                  default_value=''),
        'image/object/bbox/xmin': sparse_int64,
        'image/object/bbox/ymin': sparse_int64,
        'image/object/bbox/xmax': sparse_int64,
        'image/object/bbox/ymax': sparse_int64,
        'image/object/bbox/label': sparse_int64
    }

    features = tf.io.parse_single_example(serialized=tfrecord_serialized,
                                          features=feature_map)

    # 读取图像的 高度 和 宽度，并且，转换成浮点数
    height = features['image/height']
    width = features['image/width']

    # 读取对象边界框列表 [xmin, ymin, xmax, ymax]
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # 各个对象边界框中对象所属的类别
    labels = features['image/object/bbox/label']
    labels = tf.sparse_tensor_to_dense(labels)

    # 将 [ymin, xmin, ymax, xmax] 沿着 维度0 串联起来
    # 将 对象的边界框的取值 映射到 [0, 1) 范围内
    bbox = tf.concat([
        # 如果对象边界框与实际图片不匹配，可能存在对象边界框超出图片可能
        tf.cast(tf.minimum(ymin/height, 1.0), tf.float32),
        tf.cast(tf.minimum(xmin/width, 1.0), tf.float32),
        tf.cast(tf.minimum(ymax/height, 1.0), tf.float32),
        tf.cast(tf.minimum(xmax/width, 1.0), tf.float32)], 0)

    # 将多个对象边界框整合到一个张量中
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

    return features['image/encoded'], labels, bbox


def get_filenames(data_dir='./data/imagenet/', is_training=True):
    """
    从指定的文件夹下，读取 TFRecord 文件名称列表

    @param data_dir: 字符串。TFRecord 格式的训练样本所在的文件路径
        训练集(train)文件名称的格式如下所示：
            ${data_dir}/train/train-00000-of-01024
            ${data_dir}/train/train-00001-of-01024

        验证集(validation)文件名称的格式如下所示：
            ${data_dir}/validation/validation-00000-of-00128
            ${data_dir}/validation/validation-00001-of-00128

    @param is_training: 布尔值。是否是训练过程，训练过程读取训练集；
        验证过程读取验证集（train、validation）

    @Returns:
      fnames: 字符串列表; 每个字符串代表一个 TFRecord 文件的名称.
    """
    fnames = []
    fnames_pattern = ''
    # 用于匹配训练集文件的匹配模式
    fnames_pattern = os.path.join(data_dir, 'train', 'train-*-of-01024')
    if not is_training:
        # 用于匹配验证集文件的匹配模式
        fnames_pattern = os.path.join(
            data_dir, 'validation', 'validation-*-of-00128')

    # 匹配到的 TFRecord 文件名称列表
    fnames = tf.gfile.Glob(fnames_pattern)

    return fnames


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             dtype=tf.float32, num_parallel_batches=1,
             parse_record_fn=parse_record):
    """
    从指定的文件夹下面，读取预先生成好的 TFRecord 记录。用作训练样本。

    @param is_training: 布尔值。是否是训练过程，训练过程读取训练集；
        验证过程读取验证集（train、validation）
    @param data_dir: 字符串。TFRecord 格式的训练样本所在的文件路径
        训练集(train)文件名称的格式如下所示：
            ${data_dir}/train/train-00000-of-01024
            ${data_dir}/train/train-00001-of-01024

        验证集(validation)文件名称的格式如下所示：
            ${data_dir}/validation/validation-00000-of-00128
            ${data_dir}/validation/validation-00001-of-00128
    @param dataset: 字符串。数据集名称。如train、validation
    @param batch_size: 整型。每个批次的样本个数。
    @param num_epochs: 整型。训练的轮数。在验证时，只用训练一轮。
    @param dtype: 字符串。数据集名称。如train、validation
    @param datasets_num_private_threads: 整型。每个批次的样本个数。
    @param num_parallel_batches: 整型。训练的轮数。在验证时，只用训练一轮。
    @param parse_record_fn: 整型。训练的轮数。在验证时，只用训练一轮。

    @Returns:
      images: 字符串列表; 每个字符串代表一个图片文件的名称.
      labels: 整数列表; 代表图片所属的类别.
    """

    # 训练数据的文件名模式
    fnames = get_filenames(data_dir, is_training)

    # 从指定的文件名称列表中，读取对应的数据集
    dataset = tf.data.TFRecordDataset(fnames)

    # 读取一个批次的训练数据，
    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training:
        # 对输入的文件列表进行乱序排列
        dataset = dataset.shuffle(buffer_size=_num_train_files)

    dataset.batch(batch_size).repeat(num_epochs)

    # 将原始记录解析成 images 和 labels
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=False))

    return dataset.make_one_shot_iterator().get_next()
