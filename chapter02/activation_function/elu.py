#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

#tanh 函数
import numpy as np
import matplotlib.pyplot as plt

def elu(x, a):  
    y=[]  
    for i in x:  
        if i>=0:  
            y.append(i)  
        else:  
            y.append(a*np.exp(i)-1)  
    return y  


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
plt.plot(x, elu(x, 1), label="elu", linestyle='-',  linewidth=1, color='black') 
#plt.scatter(x,tanh_deriv(x),label="tanh_deriv")
#plt.title('tanh Function')
plt.legend(loc = 'upper left')  

plt.savefig("elu_20180622.png")  #保存图象

plt.show()