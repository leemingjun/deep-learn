#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import os
import struct
import numpy as np
import urllib.request
import gzip
import tensorflow as tf
import time
import matplotlib.pyplot as plt
# 以上部分是整个模型引用的软件包

'''
读取mnist数据文件。如果本地文件不存在，则从网络上下载并且保存到本地。
:param data_type: 要读取的数据文件类型，包括"train"和"t10k"两种，分别代表训练数据和测试
数据。
:Returns: 图片和图片的标签。图片是以张量形式保存的。
'''
def read_mnist_data(data_type="train"):
    img_path = ('./mnist/%s-images-idx3-ubyte.gz' % data_type)
    label_path = ('./mnist/%s-labels-idx1-ubyte.gz' % data_type)
    
    # 如果本地文件不存在，那么，从网络上下载mnist数据
    if not os.path.exists(img_path) or not os.path.exists(label_path) :
        # 确保./mnist/目录存在，如果不存在，就自动创建此目录
        if not os.path.isdir("./mnist/"):
            os.mkdir("./mnist/")
            
        # 从网上下载图片数据，并且，保存到本地文件
        img_url = ('http://yann.lecun.com/exdb/mnist/%s-images-idx3-ubyte.gz' % data_type)
        print("下载：%s" % img_url)
        urllib.request.urlretrieve(img_url, img_path)
        print("保存到：%s" % img_path)

        # 从网上下载标签数据，并且保存到本地
        label_url = ('http://yann.lecun.com/exdb/mnist/%s-labels-idx1-ubyte.gz' % data_type)
        print("下载：%s" % label_url)
        urllib.request.urlretrieve(label_url, label_path)
        print("保存到：%s" % label_path)
    
    # 使用gzip读取标签数据文件
    print("\n读取文件：%s" % label_path)
    with gzip.open(label_path, 'rb') as label_file:
        # 按照大端在前（big-endian）读取两个32位的整数，所以，总共读取8个字节
        # 分别是magic number、n_labels(标签的个数)
        magic, n_labels = struct.unpack('>II', label_file.read(8))
        print("magic number：%d，期望标签个数：%d 个" % (magic, n_labels))
        # 将剩下所有的数据按照byte的方式读取
        labels = np.frombuffer(label_file.read(), dtype=np.uint8)
        print ("实际读取到的标签：%d 个" % len(labels))

    # 使用gzip读取图片数据文件
    print("\n读取文件：%s" % img_path)
    with gzip.open(img_path, 'rb') as img_file:
        # 按照大端在前（big-endian）读取四个32位的整数，所以，总共读取16个字节
        magic, n_imgs, n_rows, n_cols = struct.unpack(">IIII", img_file.read(16))
        # 分别是magic number、n_imgs(图片的个数)、图片的行列的像素个数
        # （n_rows, n_cols ）
        print("magic number：%d，期望图片个数：%d个" % (magic, n_imgs))
        print("图片长宽：%d × %d 个像素" % (n_rows, n_cols))
        
        # 读取剩下所有的数据，按照 labels * 784 重整形状，其中 784 = 28 × 28 （长×宽）
        images = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(n_imgs, n_rows, n_cols)
        print ("实际读取到的图片：%d 个" % len(images))
 
    # Labels的数据类型必须转换成为int32
    return images, labels.astype(np.int32)

'''
定义输入特征列。输入的是手写数字的图片，图片的大小是固定的，每个图片都是长28个像素、
宽28个像素、单色。
:Returns: 输入特征列的集合。本例中输入特征只有一个，命名为x
'''
def define_feature_columns():
    # 返回值是输入特征列的集合（列表）
    # 请注意，返回对象是被方括号包含在内的
    return  [tf.feature_column.numeric_column(key="x", shape=[28, 28])]

# 构建深层神经网络模型
# model_name：模型的名称。如果需要训练多个模型，然后，通过验证数据来选择最好的
# 模型，那么，模型保存的路径需要区分开。这里使用model_name来区分多个模型。
# hidden_layers ：隐藏层神经元的数量，一个以为数组，其中，每个元素带个一个隐藏层的
# 神经元个数。例如[256, 32]，代表两个隐藏层，第一层有256个神经元，第二层有32个神经元
# [500, 500, 30]代表有三个隐藏层，神经元数量分别是500个、500个、30个
def mnist_dnn_classifier(name="mnist", hidden_layers=[256, 32]):
    # （1）定义输入特征列表，在这个例子中，输入特征只有一个就是“x”，
    # 该输入特征是一个28 × 28张量，其中，每个元素代表一个像素
    feature_columns = define_feature_columns()
    
    # （2）创建DNNClassifier实例
    # 构建优化函数，本例中采用AdamOptimizer优化器，初始学习率设置为1e-4
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    classifier = tf.estimator.DNNClassifier(
        # 指定输入特征列（输入变量），本例中只有一个“x”
        feature_columns=feature_columns,
        # 隐藏层，一个列表，其中的每个元素代表隐藏层神经元的个数
        hidden_units=hidden_layers,
        # 优化器，这里使用AdamOptimizer优化器
        optimizer=adam_optimizer,
        # 分类个数。手写数字的取值范围是0到9，总共有10个数字，所以，是十个类别。
        n_classes=10,
        # 将神经元dropout的概率。所谓的dropout，是指将神经元暂时地从神经网络中剔除，
        # 这样可以避免过拟合，提高模型的健壮性。 
        # 这里设置丢弃神经元的概率是10%（0.1）
        dropout=0.1,
        # 模型的保存的目录，如果是，多次训练，能够从上一次训练的基础上开始，能够提高模型的
        # 训练效果，一般来说，通过多次训练能够提高模型识别的精确度。
        model_dir=('./tmp/%s/mnist_model' % name),
        # 设置模型保存的频率。设置为每10次迭代，将训练结果保存一次。
        # 可以通过 tensorboard --logdir=model_dir，
        # 然后，通过http://localhost:6006/，来可视化的查看模型的训练过程
        config=tf.estimator.RunConfig().replace(save_summary_steps=10))
    
    # （3）定义数据输入函数
    # 读取训练数据。读取数据文件train-images-idx3-ubyte.gz、train-labels-idx1-ubyte.gz
    features, labels = read_mnist_data(data_type="train")
    # 样本数据输入函数
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        # 数据输入函数中的Dictionary对象，key=“x”，value是28 × 28的张量 
        x={"x": features},
        # 数据输入函数中的标签 
        y=labels,
        # 训练轮数 
         num_epochs=None,
        # 每批次的样本个数。对于模型训练来说，每一批数据就调整一次参数的方式能提高训练
        # 速度，实现更快的拟合。
        batch_size=100,
        # 是否需要乱序。乱序操作可以提高程序的健壮性。避免因为顺序数据中所包含的规律
        shuffle=True)
    
    # （4）模型训练
    print ("\n模型训练开始时间：%s" % time.strftime('%Y-%m-%d %H:%M:%S'))
    time_start = time.time()
    classifier.train(input_fn=train_input_fn, steps=20000)
    time_end = time.time()
    print ("模型训练结束时间：%s" % time.strftime('%Y-%m-%d %H:%M:%S'))
    print("模型训练共用时: %d 秒" % (time_end - time_start))

    # 读取测试数据集，用于评估模型的准确性
    # 读取测试数据文件t10kimages-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz
    test_features, test_labels = read_mnist_data(data_type="t10k")
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        # 数据输入函数中的Dictionary对象，key=“x”，value是28 × 28的张量 
        x={"x": test_features},
        # 数据输入函数中的标签 
        y=test_labels,
         # 轮数，对于测试来说，一轮就足够了。训练的过程才需要多轮 
        num_epochs=1,
        # 是否需要乱序。测试数据只需要结果，不需要乱序
        shuffle=False)
    
    # 评价模型的精确性
    print ("\n模型测试开始时间：%s" % (time.strftime('%Y-%m-%d %H:%M:%S')))
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print ("模型测试结束时间：%s" % (time.strftime('%Y-%m-%d %H:%M:%S')))
    
    print("\n模型识别的精确度:  {:.2f} % \n".format ((accuracy_score * 100)))

# 完成模型训练和识别准确率的评价
def mnist_model(estimator, train_input_fn, test_input_fn):
    estimator.train(input_fn=train_input_fn, steps=20000)
    accuracy_score = estimator.evaluate(input_fn=test_input_fn)["accuracy"]
    accuracy_score *= 100
    return accuracy_score

# 测试在dropout关闭的对模型准确率的影响
def search_best_model_ex():
    features, labels = read_mnist_data(data_type="train")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": features},
        y=labels,
         num_epochs=None,
        batch_size=100,
        shuffle=True)
    
    test_features, test_labels = read_mnist_data(data_type="t10k")
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_features},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    
    feature_columns = define_feature_columns()
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    
    # 分别计算神经元数量为[100, 50], [200, 100],……,[1200, 1000]的情况下，识别的准确率
    n_neurals = [50, 100, 200, 400, 500, 800, 1000, 1200]
    y = []
    for idx in range(0, len(n_neurals) - 1):
        hidden_layers = [n_neurals[idx + 1] , n_neurals[idx]]
        name = ("name{:d}".format(idx))
        classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=hidden_layers,
            optimizer=adam_optimizer,
            n_classes=10,
            # 设置dropout为None，不丢弃神经元，理论上来说会降低模型的健壮性
            # 识别准确率波动应该更大
            dropout=None,
            model_dir=('./tmp/%s/mnist_model' % name),
            config=tf.estimator.RunConfig().replace(save_summary_steps=10))
        
        score = mnist_model(classifier, train_input_fn, test_input_fn)
        y.append(score)
        
        print ("{}  scores: {:.2f}%".format(hidden_layers, score))
    
    x = n_neurals[1:]
    plt.plot(x, y, label="准确率", linestyle='-', linewidth=1, color='black') 
    
    plt.xlabel("神经元的数量")
    plt.ylabel("准确率")
    
    plt.legend(loc='upper center')  
    
    plt.show()

# 评估dropout对模型准确率的影响
search_best_model_ex()