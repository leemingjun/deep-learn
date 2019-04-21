#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import random
from datetime import datetime

import numpy as np
from six.moves import xrange
import xml.etree.ElementTree as xml_parser

import tensorflow as tf

# 设置日志级别。
tf.logging.set_verbosity(tf.logging.INFO)


def convert2_int64_feature(value):
    """ 将 int64 类型值，包装成训练样本的格式 """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def convert2_float_feature(value):
    """ 将 float 类型值，包装成训练样本的格式 """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert2_bytes_feature(value):
    """ 将 bytes 类型值，包装成训练样本的格式 """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert2_train_example(filename, image_buffer, label, synset, human, bbox,
                           height, width):
    """
    将一个图像转换成一个训练样本。

    @param filenames: 字符串列表。每个字符串代表一个图片文件。
    @param image_buffer: 字符串。JPEG编码的RGB图像。
    @param labels: 整数列表。每个整数代表一个图片的类别。
    @param synsets: 字符串列表。每个字符串代表一个同义词标识（WordNet ID）。
    @param humans: 字符串列表。每个字符串代表一个可读的类别名称。
    @param bbox: 边界框列表，每个图像有0个或者多个边界框，每个边界框都由
    [xmin, ymin, xmax, ymax]组成
    @param height: 整型, 图像的高度（像素）
    @param width: 整型, 图像的宽度（像素）

    @Returns: proto格式的样本数据
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    # 将对象边界框的各个坐标拆分开，放在四个变量中
    for b in bbox:
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]

    # 如果没有读取到对象边界框，那么，返回None
    if len(xmin) == 0:
        return None

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': convert2_int64_feature(height),
        'image/width': convert2_int64_feature(width),
        'image/colorspace': convert2_bytes_feature(colorspace.encode()),
        'image/channels': convert2_int64_feature(channels),
        'image/class/label': convert2_int64_feature(label),
        'image/class/synset': convert2_bytes_feature(synset.encode()),
        'image/class/text': convert2_bytes_feature(human.encode()),

        # 一个图片中，包含多个对象边界框时，是将所有的 xmin 放在一个列表中，同理，
        # 将所有的 xmax、ymin、ymax 也放在一个列表内
        'image/object/bbox/xmin': convert2_int64_feature(xmin),
        'image/object/bbox/xmax': convert2_int64_feature(xmax),
        'image/object/bbox/ymin': convert2_int64_feature(ymin),
        'image/object/bbox/ymax': convert2_int64_feature(ymax),

        # 每个对象边界框，都对应一个 label，
        'image/object/bbox/label': convert2_int64_feature([label] * len(xmin)),
        'image/format': convert2_bytes_feature(image_format.encode()),
        'image/filename': convert2_bytes_feature(
            os.path.basename(filename).encode()),
        'image/encoded': convert2_bytes_feature(image_buffer)}
    ))
    return example


class JpegImageCoder(object):
    """ ImageNet数据读取时解码用工具类 """

    def __init__(self):
        # 为图像编码单独创建一个会话
        self._sess = tf.Session()

        # 将PNG图像转换成JPEG格式图片的功能函数
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # 将 CMYK 图像转换成 RGB JPEG 格式图片的功能函数
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # 初始化解码 RGB JPEG 图像函数
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data,
            # 有时候，JPEG图片没有完全下载，无法完美解析，所以，
            # 让 decode_jpeg 容忍一定程度的异常和错误，提高适应性
            # 代价是，会降低最终的模型识别准确率降低。
            try_recover_truncated=True,
            acceptable_fraction=0.75,
            channels=3)

    # 将 PNG 格式转换成 JPEG 格式
    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    # 将 CMYK 格式转换成 JPEG 格式
    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    # 将 JPEG 图像文件编码成二进制格式
    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        return image


def is_png(filename):
    """
    检测一个图像文件是否是 PNG 格式。

    @param filenames: 字符串列表。每个字符串代表一个图片文件。

    @Returns: 布尔值，表明该图标是否是 PNG 格式的图像。
    """

    # 文件列表来源自:
    # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
    # 请注意：文件名后缀的大小写。'jpeg' vs 'jpg'
    return 'n02105855_2933.jpg' in filename


def is_cmyk(filename):
    """
    检测图像文件是否使用一个CMYK 色彩看空间的 JPEG 图像。

    @param filenames: 字符串列表。每个字符串代表一个图片文件。

    @Returns: 布尔值，表明该图标是否是 采用 CMYK 色彩看空间的 JPEG 图像编码。
    """

    # 文件列表来源自:
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    # 请注意：文件名后缀的大小写。
    blacklist = ['n01739381_1309.jpg', 'n02077923_14822.jpg',
                 'n02447366_23489.jpg', 'n02492035_15739.jpg',
                 'n02747177_10752.jpg', 'n03018349_4028.jpg',
                 'n03062245_4620.jpg', 'n03347037_9675.jpg',
                 'n03467068_12171.jpg', 'n03529860_11437.jpg',
                 'n03544143_17228.jpg', 'n03633091_5218.jpg',
                 'n03710637_5125.jpg', 'n03961711_5286.jpg',
                 'n04033995_2932.jpg', 'n04258138_17003.jpg',
                 'n04264628_27969.jpg', 'n04336792_7448.jpg',
                 'n04371774_5854.jpg', 'n04596742_4225.jpg',
                 'n07583066_647.jpg', 'n13037406_4650.jpg']
    return os.path.basename(filename) in blacklist


def process_single_image(filename, coder):
    """
    读取一个单一的图像文件，转换成 JPG 格式。

    @param filenames: 字符串列表。每个字符串代表一个图片文件。
    @param coder: ImageCoder的实例，用于读取图片、或者对图片进行解码。

    @Returns:
      image_buffer: 字符串。JPEG编码的RGB图像。
      height: 整型, 图像的高度（像素）。
      width: 整型, 图像的宽度（像素）。
    """
    # 读取图像文件
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # 判断图像是否是 PNG 格式
    if is_png(filename):
        # 1 将图像从 PNG 格式转换成 JPEG 格式
        print('将图像从 PNG 转换成 JPEG 格式 %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # 判断图像是否是 CMYK 格式
    elif is_cmyk(filename):
        # 22 将图像从 CMYK 格式转换成 JPEG 格式
        print('将图像从 CMYK 转换成 RGB 格式 %s' % filename)
        image_data = coder.cmyk_to_rgb(image_data)

    # 按照 RGB 格式，对图像数据进行解码
    image = coder.decode_jpeg(image_data)

    height = image.shape[0]
    width = image.shape[1]

    return image_data, height, width


def process_batch_images(coder, thread_index, ranges, name, filenames,
                         synsets, labels, humans, bboxes, num_shards):
    """
    处理图片并且将图片转换成为 TFRecord 的单个线程。

    @param coder: ImageCoder的实例，用于读取图片、或者对图片进行解码。
    @param thread_index: 整型，一个批次中的索引值，取值范围[0, len(ranges)).
    @param ranges: 两个整型数值，表示并行处理过程中一个批次的范围。
    @param name: 字符串，唯一标识数据集。
    @param filenames: 字符串列表。每个字符串代表一个图片文件。

    @param synsets: 字符串列表。每个字符串代表一个同义词标识（WordNet ID）。
    @param labels: 整数列表。每个整数代表一个图片的类别。
    @param humans: 字符串列表。每个字符串代表一个可读的类别名称。
    @param bboxes: 边界框列表，每个图像有0个或者多个边界框。
    @param num_shards: 数据分片，并行处理时每个批次处理的图片个数。
    """

    # 每个线程处理 N 份数据， 其中，N = int(num_shards / num_threads).
    # 例如，如果每个分片含128个图片（num_shards = 128），采用两个线程并行处理
    # （um_threads = 2），那么，第一个线程处理的分片范围 [0, 64)
    num_threads = len(ranges)
    num_shards_per_batch = int(num_shards / num_threads)

    # 每个分片的范围，在每个线程分得的任务内，再次分片
    # 对于 ImageNet 来说，训练集包含大约100万个样本图片，如果采用 10 个线程并行处理
    # 那么，每个线程大约分得10多万个图片数据，在线程内，需要将本线程获得的任务再次分片
    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    # 每个线程处理的图片数量
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0

    # s 是图片文件的索引
    for idx in xrange(num_shards_per_batch):
        # 生成一个数据分片，将一个数据分片的中包含的图片数据、标签数据、对象边界框转换成TFRecord
        # 然后，保存到一个样本文件中，例如，'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + idx
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)

        # 根据不同的数据集名称，将结果文件保存到对应的文件夹
        output_file = os.path.join(data_dir, name, output_filename)

        # 将样本数据写入到对应的样本数据文件的句柄
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(
            shard_ranges[idx], shard_ranges[idx + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            synset = synsets[i]
            human = humans[i]
            bbox = bboxes[i]

            # 读取图片的内容，一律转换成为 JPG 格式
            image_buffer, height, width = process_single_image(filename, coder)

            # 将图片转换成为训练样本，TFRecord
            example = convert2_train_example(
                filename, image_buffer, label, synset, human,
                bbox, height, width)

            if example is not None:
                # 写入 TFRecord 格式的训练样本中
                # 每个训练样本中包含 num_shards 个图片文件
                writer.write(example.SerializeToString())
                shard_counter += 1
                counter += 1

            # 每处理 1000 个图片文件，
            if not counter % 1000:
                msg = "{} [{}]: 正在处理第 {}/{} 个图像".format(
                    datetime.now(), thread_index, counter,
                    num_files_in_thread)
                print(msg)
                sys.stdout.flush()

            writer.close()

        print('%s [线程 %d]: 已经将 %d 个图片写入 %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0

    print('\n%s [线程 %d]: 已经将 %d 个图片写入 %d 数据分片中.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def build_human_labels(synsets, human_readable_labels):
    """
    构建可读标签的列表。

    @param synsets: 字符串列表。每个字符串代表一个同义词标识（WordNet ID）。
    @param synset_to_human: 字典。从图片所属的类别到可读的类别映射，例如：
      'n02119022' --> 'red fox, Vulpes vulpes'

    @Returns: 与同义词一一对应的刻度标签列表。
    """
    humans = []
    for s in synsets:
        humans.append(human_readable_labels[s])
    return humans


def find_image_files(data_dir, labels_file):
    """
    读取 ImageNet 图片数据集。

    @param data_dir: 字符串. ImageNet图片集所在的本地目录。
    图片文件名称的格式如下所示：
        ${data_dir}/images/n02084071/n02084071_7.JPG
        ${data_dir}/images/n02084071/n02084071_60.JPG
    其中，n02084071 代表图片所属的类别（同义词 ID）

    一个图片中会包含多个对象，对象所在的位置大小、类别数据，保存在如下XML文件中：
        ${data_dir}/annotation/n02084071/n02084071_7.xml
        ${data_dir}/annotation/n02084071/n02084071_60.xml
    其中，n02084071 代表图片所属的类别（同义词 ID）

    @param labels_file: 字符串。包含类别名称的文本文件，每一行代表一个类别，如：
        n02084071
        n04067472
        n04540053

    @Returns:
      filenames: 字符串列表; 每个字符串代表一个图片文件的名称.
      synsets: 字符串列表; 每个字符串是一个同义词唯一标识（WordNet ID）.
      labels: 整数列表; 代表图片所属的类别.
    """

    labels_file = os.path.join(data_dir, labels_file)
    challenge_synsets = [l.strip() for l in
                         tf.gfile.FastGFile(labels_file, 'r').readlines()]

    labels = []
    filenames = []
    synsets = []

    # Label索引，索引0保留
    label_index = 1

    # 构建图片文件名称列表和对应的labels列表
    for synset in challenge_synsets:
        jpeg_file_path = os.path.join(data_dir, 'images', synset)
        # 如果，${data_dir}/images/${synset} 文件夹不存在
        # 说明该同义词对应的图片都不存在，略过
        if not os.path.exists(jpeg_file_path):
            print("路径：{} 不存在".format(jpeg_file_path))
            continue

        # 找到指定路径下的所有“*.jpg”图片
        jpeg_file_path = os.path.join(jpeg_file_path, '*.jpg')
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        synsets.extend([synset] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('共找到 {} 个类别 {} 个文件.'.format(
                label_index, len(challenge_synsets)))
        label_index += 1

    # 对图片进行乱序操作，让模型训练保证足够的多样性
    shuffled_index = list(range(len(filenames)))
    random.seed(datetime.now)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    synsets = [synsets[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('在 {} 文件夹下，查找 {} 类别图片，共找到 {} 个图片文件。'.format
          (data_dir, len(challenge_synsets), len(filenames)))
    return filenames, synsets, labels


def read_annotation_file(image_files):
    """
    读取各个图片中包含所有对象的框，返回根据文件名称到框的映射。

    @param image_files: 字符串列表。图片文件名称列表。
    每个文件名称形如：n02084071_79.jpg，每个图片对应的有一个annotation文件
    位于： ${data_dir}/annotation/${synset}/${synset}_*.xml

    该XML文件中包含0个或者多个对象，所包含的对象大小为：

    <object>
      <name>n02084071</name>
      <pose>Unspecified</pose>
      <truncated>0</truncated>
      <difficult>0</difficult>
      <bndbox>
        <xmin>0</xmin>
        <ymin>102</ymin>
        <xmax>182</xmax>
        <ymax>387</ymax>
      </bndbox>
    </object>

    @Returns:
      图片到该图片所包含的对象框列表的字典
    """
    images_to_bboxes = {}
    num_bbox = 0
    num_image = 0
    for image_filename in image_files:
        image_basename = os.path.basename(image_filename)
        tmp = os.path.splitext(image_basename)[0]
        synset = tmp[0:tmp.index('_')]
        xml_file = "./data/imagenet/annotation/{}/{}.xml".format(
            synset, tmp)
        if not tf.gfile.Exists(xml_file):
            print("图片 {} 对应的 annotation： {} 文件不存在".format(
                image_basename, xml_file))
            continue

        if xml_file:
            # 解析XML文件
            xmldoc = xml_parser.parse(xml_file)

            # 读取所有的对象列表
            object_list = xmldoc.findall('object')

            # 读取该对象坐标的对标位置（对应的像素）
            for object_element in object_list:
                xmin = int(object_element.find('bndbox/xmin').text)
                ymin = int(object_element.find('bndbox/ymin').text)
                xmax = int(object_element.find('bndbox/xmax').text)
                ymax = int(object_element.find('bndbox/ymax').text)

            # 对象所在的边界框
            box = [xmin, ymin, xmax, ymax]

            # 以图片文件的basename为主键，是包含'.jpg'的后缀的
            if image_basename not in images_to_bboxes:
                images_to_bboxes[image_basename] = []
                num_image += 1

            # 图片到 边界框的映射（Map）
            images_to_bboxes[image_basename].append(box)
            num_bbox += 1

        print('针对图片文件{}, 读取到 {} 个对象'.format(
            image_basename, len(images_to_bboxes[image_basename])))

    return images_to_bboxes


def build_bounding_boxes(filenames, image_to_bboxes):
    """
    给定一个图片，读取它所包含的对象边界框列表。

    @param filenames: 字符串列表。每个字符串代表一个图像的文件名称。
    @param image_to_bboxes: 字典，从图像文件名到对象边界框列表的映射，列表中
    每个图像可以包含0个或者多个对象边界框。理论上，本字典包含所有的对象边界框。
    @Returns:
      针对每一个图像的对象边界框列表，每一个图片都可以包含多个对象边界框。
    """
    num_image_bbox = 0
    bboxes = []
    for f in filenames:
        # 字典的 key， 本例中采用文件的 basename
        basename = os.path.basename(f)

        if basename in image_to_bboxes:
            bboxes.append(image_to_bboxes[basename])
            num_image_bbox += 1
        else:
            # 如果 image_to_bboxes 中没有该文件的对象边界框，那么，
            # 创建一个空的对象边界框列表（缺少该图像对象边界框信息）
            bboxes.append([])

    print('找到 %d 个图片的共计 %d 对象边界框' % (
        len(filenames), num_image_bbox))

    return bboxes


def build_synset_map(synset_to_human_file):
    """
    生成图片 类别ID 与 所属类别名称的映射。

    @synset_to_human_file: 字符串。包含图片类别ID 与 类别映射数据的文件.
    文件内容如下所示:
          n02119247    black fox
          n02119359    silver fox
          n02119477    red fox, Vulpes fulva

    每行包含一个映射关系，格式如：
    <同义词标识>\t<可读的类别名称>.

    @Returns:
      类别ID 到 类别名称的字典
    """

    # 读取文件内容
    synset_to_human_file = os.path.join(data_dir, synset_to_human_file)
    lines = tf.gfile.FastGFile(synset_to_human_file, 'r').readlines()
    # 要返回的 { 类别标识 -> 类别名称的 } 字典
    synset_to_human = {}
    for line in lines:
        if line:
            parts = line.strip().split('\t')
            synset = parts[0]
            human = parts[1]
            synset_to_human[synset] = human

    return synset_to_human


def process_imagenet_images(name, filenames, synsets, labels, humans,
                            bboxes, num_shards):
    """
    处理所有的图像，并且，将图像转换成为 TFRecord 格式，保存起来。

    @param name: 字符串。数据集的唯一标识。
    @param filenames: 字符串列表。每个字符串代表一个图像的文件名称。
    @param synsets: 字符串列表。每个字符串代表一个同义词标识（WordNet ID）。
    @param labels: 整数列表。每个整数代表一个图片的类别。
    @param humans: 字符串列表。每个字符串代表一个可读的类别名称。
    @param bboxes: 边界框列表，每个图像有0个或者多个边界框。
    @param num_shards: 并行处理时每个形成处理的图片个数.
    """
    # 按照并行线程的数量，将图像数据分成多个分片
    spacing = np.linspace(
        0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    threads = []

    # 计算每个分片的起止范围
    # 每个批次的起止范围 [ranges[i][0], ranges[i][1]].
    # 例如，总样本数量是130万个，线程数量是10个，那么，每个线程处13万个图片
    # 因为, 线程数量是10个，所以，len(ranges) = 10。
    # 对应的，ranges[0][0] = 0, ranges[0][1] = 130000。
    # 对应的，ranges[1][0] = 130001, ranges[1][1] = 160000。
    #           ……
    #        ranges[9][0] = 1170001, ranges[9][1] = 1300000
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # 加载一个线程，处理一个批次数据
    print('启动第 {}个线程，处理 {} 批次数据'.format(num_threads, ranges))
    sys.stdout.flush()

    # 启动线程同步机制，监控等到所有的线程执行完毕
    coord = tf.train.Coordinator()

    # 创建一个图像编码器
    coder = JpegImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        # 每个线程的参数
        args = (coder, thread_index, ranges, name, filenames,
                synsets, labels, humans, bboxes, num_shards)
        # 创建线程，指定参数
        t = threading.Thread(target=process_batch_images, args=args)
        # 启动线程
        t.start()
        threads.append(t)

    # 等待所有的线程执行完毕（线程同步）
    coord.join(threads)

    print('%s: 执行完毕，已经将所有的，共 %d 个图片保存到数据集中。' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def process_imagenet(name, num_shards, synset_to_human):
    """
    处理 ImageNet 数据集。

    @param name: 字符串。数据集名称，如 训练集train、验证集validation。
    @param num_shards: 数据集分成几个分片来处理，每个分片处理多少个图片.
    @param synset_to_human: 字典。从图片所属的类别到可读的类别映射，例如：
      'n02119022' --> 'red fox, Vulpes vulpes'
    """
    # 所有图像文件名、同义词、同义词标签列表
    filenames, synsets, labels = find_image_files(
        data_dir, labels_file)
    # 读取图像文件所包含的所有对象框列表
    image_to_bboxes = read_annotation_file(filenames)

    if len(image_to_bboxes) > 0:
        humans = build_human_labels(synsets, synset_to_human)
        bboxes = build_bounding_boxes(filenames, image_to_bboxes)
        # 处理所有的 ImageNet 图像文件
        process_imagenet_images(name, filenames, synsets, labels,
                                humans, bboxes, num_shards)
    else:
        print("没有找到对象边界框！")


def process_imagenet_data():
    """ 将原始图片转换成为TFRecord格式的样本数据 """

    # 读取可读分类标签名称,包含所有的分类标签名称
    synset_to_human = build_synset_map(synset_to_human_file)

    # 数据分片大小，测试集每个分片包括1024个样本
    train_shards = 1024
    # 验证集中，每个数据分片的大小为128个图片
    validation_shards = 128
    # 将训练集数据转换成 TFRecord 格式的样本数据
    process_imagenet('train', train_shards,
                     synset_to_human)
    # 将训练集数据转换成 TFRecord 格式的验证数据
    process_imagenet('validation',
                     validation_shards, synset_to_human)


# 相关参数
data_dir = './data/imagenet/'
train_dir = './data/imagenet/train/'
validation_dir = './data/imagenet/validation/'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

labels_file = 'synset_list.txt'
synset_to_human_file = 'synset_to_human.txt'

# ImageNet 包含图片大约1.3万个，需要并行处理，本例中采用4个线程并行处理
num_threads = 4

# ImageNet图像数据集文件，生成 TFRecord 格式的训练样本
process_imagenet_data()
