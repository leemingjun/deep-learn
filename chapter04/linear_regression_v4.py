#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import os
import urllib.request
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.data import Dataset
import time
# 以上部分是整个模型引用的软件包

# 使用黑体来显示中文，如果不指定中文字体，中文会显示乱码
plt.rcParams['font.sans-serif'] = ['SimHei']


def read_sample_data(csv_file_name="./data/california_housing_train.csv"):
    """
    读取训练数据。首先尝试从本地读取，如果本地数据文件不存在，则从网络读取，
    然后，将样本数据整理成pandas的Dataframe对象。
    :param data_file_name: 本地训练数据文件名称
    :Returns: 包含样本数据的Dataframe对象
    """
    # 首先，检查本地文件是否存在，如果不存在，则从网络下载训练数据
    if not os.path.exists(csv_file_name):
        csv_url = "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv"
        print("开始下载：{}".format(csv_url))
        urllib.request.urlretrieve(csv_url, csv_file_name)
        print("保存到：{}".format(csv_file_name))

    # 从本地文件中读取
    print("从本地文件中读取数据：" + csv_file_name)
    california_housing_dataframe = pd.read_csv(csv_file_name)

    return california_housing_dataframe


def train_data_input_fn(features, targets, batch_size=128, shuffle=True,
                        num_epochs=None):
    """
    样本数据输入函数，用于将训练数据喂给训练模型（LinearEstimator）。

    @param features:  Dataframe对象，表示输入的特征变量列表。本例子中输入特征变量只有一个。
    @param targets: Dataframe对象，训练的目标变量，本例中目标变量是“median_house_value”，代表房价中位数
    @param batch_size:  批处理的大小。模型训练时，输入一批数据，计算预测值与实际样本目标变量
    的误差大小，然后，根据这一批数据的误差，调整响应的参数。根据每一批数据的预测误差调整结果可以提高模型的训练速度。
    @param shuffle: 是否需要乱序（随机输入样本）。目的是提高模型的健壮性。
    @param num_epochs: 训练轮数。如果，batch_size *  num_epochs大于样本数据的数量，那么，
    部分样本数据会被多次（重复）使用。
    @Returns: 返回值。下一个批次的输入特征、目标特征的元组。
    """

    # 将输入特征与目标特征构建的元组整合数据集（Dataset）
    dataset = Dataset.from_tensor_slices((features, targets))

    # 如果需要乱序，那么，执行乱序操作
    if shuffle:
        dataset.shuffle(batch_size)

    # 将样本数据集，按训练轮数、批次样本数据个数
    dataset = dataset.batch(batch_size).repeat(num_epochs)

    # 构造样本数据的迭代器，每次返回一个批次的数据
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels


def test_data_input_fn(features, targets):
    """
    返回测试数据。一般地，训练数据不能再用作测试数据。本例中采用抽样的方法，没有严格避免
    训练数据用作测试数据。
    @param features: 测试数据的输入特征列。
    @param targets: 测试数据的目标特征列表。
    @Returns: 返回测试数据的输入特征、目标特征数据。
    """
    # 将输入特征与目标特征构建的元组整合数据集（Dataset）
    dataset = Dataset.from_tensor_slices((features, targets))

    # 测试数据同样需要分批返回，每批次只包含一个数据
    dataset = dataset.batch(1).repeat(1)
    # 构造测试数据的迭代器，每次返回一个批次的数据
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels


def define_feature_columns(feature_col_name):
    """
    定义输入特征列。

    :param feature_col_name: 输入特征列的名称。
    :Returns: 输入特征列的集合。
    """
    # 输入特征列可以有多个输入特征（输入变量），所以，是一个列表
    # 请注意，返回对象是被方括号包含在内的
    return [tf.feature_column.numeric_column(feature_col_name)]


def make_train_and_test_data(dataframe, train_data_precent=0.8):
    """
    将样本数据切分成训练数据和测试数据，默认按照（8：2）的比例
    @param dataframe:  原始的样本数据
    @param train_data_precent:  训练数据站的百分比（默认是80%）
    @Returns: 返回训练数据和测试数据的Dataframe
    """
    train_dataframe = dataframe.sample(frac=train_data_precent)
    test_dataframe = dataframe.sample(frac=(1 - train_data_precent))

    return train_dataframe, test_dataframe


def train_mode(linear_regressor, train_dataframe, feature_col_name,
               target_col_name):
    """
    训练模型。
    @param linear_regressor:  线性回归模型
    @param train_dataframe:  训练数据
    @param feature_col_name:  输入的特征列名称
    @param target_col_name:  目标特征列名称
    @Returns: 返回训练数据和测试数据的Dataframe
    """

    # 从数据集中读取输入特征列的数据列表，构造成Dictionary的形式
    features_raw_data = train_dataframe[feature_col_name]
    # 所有的输入特征列表必须构造成Dictionary的形式
    train_features = {feature_col_name: np.array(features_raw_data)}
    # 构造目标特征列表
    train_targets = train_dataframe[target_col_name]

    # 模型训练开始，计时
    time_start = time.time()
    linear_regressor.train(input_fn=lambda:  train_data_input_fn(
        train_features, train_targets), steps=2000)
    time_end = time.time()
    seconds_used = np.round(time_end - time_start, 1)
    print("模型训练共用了: {} 秒".format(seconds_used))


def evolution_mode(linear_regressor, test_dataframe, feature_col_name,
                   target_col_name):
    """
    评估模型。计算模型在测试数据上的误差，用于评估模型。
    @param linear_regressor:  线性回归模型
    @param test_dataframe:  训练数据
    @param feature_col_name:  输入的特征列名称
    @param target_col_name:  目标特征列名称
    @Returns: 返回训练数据和测试数据的Dataframe
    """
    # (3) 用测试数据对模型进行测试，评价模型的误差
    # 从数据集中读取输入特征列的数据列表，构造成Dictionary的形式
    test_raw_data = test_dataframe[feature_col_name]
    # 所有的输入特征列表比如构造成Dictionary的形式
    test_features = {feature_col_name: np.array(test_raw_data)}
    # 构造目标特征列表
    test_targets = test_dataframe[target_col_name]

    # 调用预测函数
    predict_result = linear_regressor.predict(
        lambda: test_data_input_fn(test_features, test_targets))
    # 将预测结果转换成python的数组形式
    predictions = np.array([item['predictions'][0] for item in predict_result])

    # 计算预测结果与测试数据中输出特征的误差（差的平方）
    square_error = np.square(predictions - np.array(test_targets))
    # 求平均、开根号
    RMSE = np.sqrt(np.mean(square_error))
    return RMSE


def visualization_model(linear_regressor, dataframe, feature_col_name,
                        target_col_name):
    """
    抽样一部分数据，将模型用图形化的方式展现出来。
    @param linear_regressor:  线性回归模型
    @param dataframe:  样本数据集合
    @param feature_col_name:  输入的特征列名称
    @param target_col_name:  目标特征列名称
    """

    # 样本数据共有34000条，抽取2%的样本数据（约340条），用图形化的展现出来
    sample = dataframe.sample(frac=0.02)

    # 读取模型训练得到的结果，读取权重和偏置项
    weight = linear_regressor.get_variable_value('linear/linear_model/'
                                                 + feature_col_name + '/weights')[0]
    bias = linear_regressor.get_variable_value(
        'linear/linear_model/bias_weights')

    # 计算预测模型的直线的起点和终点。这是起点
    x_start = sample[feature_col_name].min()
    y_start = weight * x_start + bias

    # 预测模型执行的终点
    x_end = sample[feature_col_name].max()
    y_end = weight * x_end + bias

    # 横坐标、纵坐标的标签
    plt.xlabel(feature_col_name, fontsize=14)
    plt.ylabel(target_col_name, fontsize=14)

    # 画出样本数据的散点图
    plt.scatter(sample[feature_col_name], sample[target_col_name], c='black')

    # 画出预测曲线
    plt.plot([x_start, x_end], [y_start, y_end], c='black')

    plt.show()


def linear_regression_main(feature_col_name, target_col_name):
    """
    线性回归模型。
    @param feature_col_name：输入特征列的名称，本例中输入特征变量只有一个。
    @param target_col_name: 目标特征列的名称。
    """

    # （1）构建线性回归模型
    # 构建优化函数，本例中采用AdamOptimizer优化器，初始学习率设置为0.01
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    feature_columns = define_feature_columns(feature_col_name)
    # 构造线性回归模型，指定输入特征列和优化器
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=adam_optimizer
    )

    # (2) 读取样本数据，将样本数据切分成训练数据和测试数据
    california_housing_dataframe = read_sample_data()
    # 将房屋的价格的单位转换成“千元”
    california_housing_dataframe[target_col_name] /= 1000.0
    train_dataframe, test_dataframe = make_train_and_test_data(
        california_housing_dataframe)

    # (3) 训练模型。使用训练数据对模型进行训练。
    train_mode(linear_regressor, train_dataframe,
               feature_col_name, target_col_name)

    # (4) 评估模型。计算误差，用于评估模型的准确性
    SMSE = evolution_mode(linear_regressor, test_dataframe,
                          feature_col_name, target_col_name)

    # 输出均方误差与均方根误差
    print("模型的均方差是: {}".format(np.round(SMSE, 2)))

    # (5) 对模型训练的结果进行可视化操作
    visualization_model(linear_regressor, california_housing_dataframe,
                        feature_col_name, target_col_name)


# 线性回归的入口函数
linear_regression_main("total_rooms", "median_house_value")
# 当然也可以使用别的输入特征列来预测房价
# linear_regression_main("median_income", "median_house_value")
