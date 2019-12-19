#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np

# 样本数据给出的分布概率，等于4的概率是100%，等于其他数字的概率是0%
p = [0,0,0,0,1,0,0,0,0,0]

# 第一种情况，识别结果是4的概率是80%，识别结果是9的概率是20%
q1 = [0,0,0,0,0.8,0,0,0,0,0.2]
# 第二种情况，识别结果是4的概率是60%，是9的概率是20%，是0的概率是20%
q2 = [0.2,0,0,0,0.6,0,0,0,0,0.2]

loss_q1 = 0.0
loss_q2 = 0.0

for i in range(1, 10):
    # 如果p(x)=0，那么，p(x)*log q1(x) 肯定等于0。如果q1(x)等于0，那么，log q1(x)不存在
    if p[i] != 0 and q1[i] != 0:
        loss_q1 += -p[i]*np.log(q1[i])
    
     # 如果p(x)=0，那么，p(x)*log q2(x) 肯定等于0。如果q2(x)等于0，那么，log q2(x)不存在    
    if p[i] != 0 and q2[i] != 0:
        loss_q2 += -p[i]*np.log(q2[i])
        
print ("第一种情况，误差是：{}".format( round(loss_q1, 4)))       
print ("第二种情况，误差是：{}".format(round(loss_q2, 4)))       