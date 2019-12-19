#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

# 绘制正态分布概率密度函数
u = 0  # 均值μ=0
sig = math.sqrt(1)  # 标准差δ=1

x = np.linspace(u - 3 * sig, u + 3 * sig, 50)
y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)


u01 = -2  # 均值μ=6
sig01 = math.sqrt(2)  # 标准差 δ=2
x01 = np.linspace(u01 - 3 * sig, u01 + 3 * sig, 50)
y01_sig = np.exp(-(x01 - u01) ** 2 / (2 * sig01 ** 2)) / \
    (math.sqrt(2 * math.pi) * sig01)

y02 = np.random.random_sample(size=41) * 0.25 + 0.05


# 300 represents number of points to make between T.min and T.max
xnew = np.linspace(-5, 3, 41)
power_smooth = spline(T, power, xnew)


plt.plot(x, y_sig, color="black", linewidth=2)
plt.plot(x01, y01_sig, color="black", linewidth=1)
plt.plot(x02, y02, color="black", linewidth=1, linestyle=":", fmt="o")
plt.grid(True)
plt.show()
