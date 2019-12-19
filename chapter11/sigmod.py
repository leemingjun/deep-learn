#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

# tanh 函数
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = 14  # 用来正常显示中文标签
plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 200 #分辨率

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))


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


x = np.linspace(-5, 5, 41)
plt.plot(x, sigmoid(x), label="sigmoid",
         linestyle='-',  linewidth=2, color='black')
plt.legend(loc='upper left')

print("{:.2f}".format(sigmoid(-2)))
print("{:.2f}".format(sigmoid(2)))
plt.show()
