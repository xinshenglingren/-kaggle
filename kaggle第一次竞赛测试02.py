# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:19:58 2019

@author: 潘保恒
"""

import numpy as np
#import matplotlib.pyplot as plt


#data = np.loadtxt('HTRU_2_train.csv', delimiter=',')
# 
#X = data[:, 0:2]
#y = data[:, 2]
# 
#pos = np.where(y == 1)
#neg = np.where(y == 0)
#plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
#plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
#plt.xlabel('Exam1')
#plt.ylabel('Exam2')
#plt.legend(['Fail', 'Pass'])
#plt.show()

def loadDataSet(data):
    for i in range(len(data[0])):
        sum = 0
        for d in data:
            sum += d[i]
        miu = sum / len(data)
        sum = 0
        for d in data:
            sum += (d[i] - miu)**2
            sum = sum/ len(data)
        sigma = sum**(0.5)
        for d in data:
            d = (d - miu)/sigma
    return data

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#批梯度上升
    
def gradAscent(dataMatIn, classLabels):
    # 输入训练数据
    dataMatrix = np.mat(dataMatIn)             #convert to NumPy matrix
    # 输入训练数据的标签（0 / 1）
    labelMat = np.mat(classLabels).transpose() #convert to NumPy matrix
    m,n = np.shape(dataMatrix)
    # 训练步长 （越大则收敛的速度）
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    # 训练函数的系数（为需要求解的结果）
    weights = np.ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        # 梯度上升算法的 迭代 算法
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

#随机梯度上升

#def stocGradAscent0(dataMatrix, classLabels):
#    m,n = np.shape(dataMatrix)
#    alpha = 0.01
#    weights = np.ones(n)   #initialize to all ones
#    for i in range(m):
#        #每次迭代只需要一个训练数据
#        h = sigmoid(sum(dataMatrix[i]*weights))
#        error = classLabels[i] - h
#        weights = weights + alpha * error * dataMatrix[i]
#    return weights

def judge(dataMatIn, weights, alpha):      #alpha是判断阈值
    dataMatIn = np.insert(dataMatIn, 0, 1, axis=0)
    dataMatrix = np.mat(dataMatIn)      #x矩阵
    m,n = np.shape(dataMatrix)
    h = sigmoid(dataMatIn*weights)
    if h>alpha:
        return 1
    else:
        return 0
    
data = np.loadtxt('HTRU_2_train.csv',delimiter=',')
np.random.shuffle(data)
traningLab = data[:,-1]      #训练集的标签集
traningData = data[:,:-1]      #训练集的数据集
traningData= loadDataSet(traningData)
testData = np.loadtxt('HTRU_2_test.csv',delimiter=',')#测试集的数据集
weights = gradAscent(traningData, traningLab)
print(weights)
p=[]
for i in range(len(testData)):
    p.append([i+1,judge(testData[i], weights, 0.5)])
print(p)

np.savetxt("nrwe.csv", p, delimiter=',',fmt="%d")
