#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签


def ReLu(z):
    return np.array([x if x > 0 else 0 for x in z])


def _leaky_relu(z):
    return np.array([x if x > 0 else 0.2 * x for x in z])


# 设置第一幅图像
plt.subplot(121)

# 获取当前坐标轴对象
ax = plt.gca()

# 将垂直坐标刻度置于左边框
ax.yaxis.set_ticks_position('left')
# 将水平坐标刻度置于底边框
ax.xaxis.set_ticks_position('bottom')
# 将左边框置于数据坐标原点
ax.spines['left'].set_position(('data', 0))
# 将底边框置于数据坐标原点
ax.spines['bottom'].set_position(('data', 0))

# 将右边框和顶边框设置成无色
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 设置x轴的刻度线，取值区间为[-10, 10]，共有10个间隔（对应11个刻度线）
plt.xticks(np.linspace(-10, 10, 11))

# x的取值，取值区间为[-10, 10],供取400个值
x = np.linspace(-10, 10, 400)
plt.plot(x, ReLu(x), label="ReLu",
         linestyle='-',  linewidth=1, color='black')
plt.legend(loc='upper left')

# 设置第二幅图像
plt.subplot(122)
plt.xticks(np.linspace(-10, 10, 11))

# 获取当前坐标轴对象
ax = plt.gca()

# 将垂直坐标刻度置于左边框
ax.yaxis.set_ticks_position('left')
# 将水平坐标刻度置于底边框
ax.xaxis.set_ticks_position('bottom')
# 将左边框置于数据坐标原点
ax.spines['left'].set_position(('data', 0))
# 将底边框置于数据坐标原点
ax.spines['bottom'].set_position(('data', 0))

# 将右边框和顶边框设置成无色
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

x = np.linspace(-10, 10, 400)
plt.plot(x, _leaky_relu(x), label="leaky_relu",
         linestyle='--',  linewidth=1, color='black')
plt.legend(loc='upper left')


plt.show()
