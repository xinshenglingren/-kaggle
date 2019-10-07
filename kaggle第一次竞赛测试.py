# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:00:01 2019

@author: 潘保恒
"""
#逻辑回归模型

#import numpy as np
#import re
#from pandas import DataFrame
#import time as time
#import matplotlib.pyplot as plt
#
#def get_data(filename):                 #读取数据
#    f = open(filename)
#    data = DataFrame(columns=['x1','x2','label']) #构造DataFrame存放数据，列名为x与y
#    line = f.readline()
#    line = line.strip()
#    p = re.compile(r'\s+')              #由于数据由若干个空格分隔，构造正则表达式分隔
#    while line:
#        line = line.strip()
#        linedata = p.split(line)
#        data.set_value(len(data),['x1','x2','label'],[1,float(linedata[1]),int(linedata[2])]) #数据存入DataFrame
#        line = f.readline()
#    return np.array(data.loc[:,['x1','x2']]),np.array(data['label'])
#def sigmoid(x):
#    return 1.0/(1+np.exp(-x))
#
##def stocGradAscent(dataMat,labelMat,alpha = 0.01):   #随机梯度上升
##    start_time = time.time()                         #记录程序开始时间
##    m,n = dataMat.shape
##    weights = np.ones((n,1))                         #分配权值为1
##    for i in range(m):
##        h = sigmoid(np.dot(dataMat[i],weights).astype('int64')) #注意：这里两个二维数组做内积后得到的dtype是object,需要转换成int64
##        error = labelMat[i]-h                        #误差
##        weights = weights + alpha*dataMat[i].reshape((3,1))*error #更新权重
##    duration = time.time()-start_time
##    print('time:',duration)
##    return weights
#
#def betterStoGradAscent(dataMat,labelMat,alpha = 0.001,maxstep = 3000):
#    start_time = time.time()
#    m,n = dataMat.shape
#    weights = np.ones((n,1))
#    for j in range(maxstep):
#        for i in range(m):
#            alpha = 4/(1+i+j) + 0.001                         #设置更新率随迭代而减小
#            h = sigmoid(np.dot(dataMat[i],weights).astype('int64'))
#            error = labelMat[i]-h
#            weights = weights + alpha*dataMat[i].reshape((3,1))*error
#    duration = time.time()-start_time
#    print('time:',duration)
#    return weights
#
#def show(dataMat, labelMat, weights):
#    #dataMat = np.mat(dataMat)
#    #labelMat = np.mat(labelMat)
#    m,n = dataMat.shape
#    min_x = min(dataMat[:, 1])
#    max_x = max(dataMat[:, 1])
#    xcoord1 = []; ycoord1 = []
#    xcoord2 = []; ycoord2 = []
#    for i in range(m):
#        if int(labelMat[i]) == 0:
#            xcoord1.append(dataMat[i, 1]); ycoord1.append(dataMat[i, 2])
#        elif int(labelMat[i]) == 1:
#            xcoord2.append(dataMat[i, 1]); ycoord2.append(dataMat[i, 2])
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.scatter(xcoord1, ycoord1, s=30, c="red", marker="s")
#    ax.scatter(xcoord2, ycoord2, s=30, c="green")
#    x = np.arange(min_x, max_x, 0.1)
#    y = (-float(weights[0]) - float(weights[1])*x) / float(weights[2])
#    ax.plot(x, y)
#    plt.xlabel("x1"); plt.ylabel("x2")
#    plt.show()
#
#if __name__=='__main__':
#    dataMat,labelMat = get_data('HTRU_2_train.csv')
#    weights = betterStoGradAscent(dataMat,labelMat)
#    show(dataMat,labelMat,weights)




import matplotlib.pyplot as plt
import numpy as np
# sigmoid函数和初始化数据
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def init_data():
    data = np.loadtxt('HTRU_2_train.csv')
    dataMatIn = data[:, 0:-1]
    classLabels = data[:, -1]
    dataMatIn = np.insert(dataMatIn, 0, 1, axis=1)  #特征数据集，添加1是构造常数项x0
    return dataMatIn, classLabels

# 梯度上升
def grad_descent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  #(m,n)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))  #初始化回归系数（n, 1)
    alpha = 0.001 #步长
    maxCycle = 500  #最大循环次数

    for i in range(maxCycle):
        h = sigmoid(dataMatrix * weights)  #sigmoid 函数
        weights = weights + alpha * dataMatrix.transpose() * (labelMat - h)  #梯度
    return weights
#计算结果
if __name__ == '__main__':
    dataMatIn, classLabels = init_data()
    r = grad_descent(dataMatIn, classLabels)
    print(r)
    
def plotBestFIt(weights):
    dataMatIn, classLabels = init_data()
    n = np.shape(dataMatIn)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if classLabels[i] == 1:
            xcord1.append(dataMatIn[i][1])
            ycord1.append(dataMatIn[i][2])
        else:
            xcord2.append(dataMatIn[i][1])
            ycord2.append(dataMatIn[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1,s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3, 3, 0.1)
    y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]  #matix
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()