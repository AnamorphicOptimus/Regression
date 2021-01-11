#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistics Regression
"""

import numpy as np
import pandas as pd
import time as time



filename = "./data.csv"
threshold = 0.4


def sigmoid(x):
    """Create logistic function/activation function"""
    return 1.0/(1+np.exp(-x))

def loadData(filename):
    # Load data, convert data into variables and label matrix
    data = pd.read_csv(filename)
    xMat = []
    yMat = []
    p = data.shape[1]-1
    n = data.shape[0]
    for i in range(0,n):
        tmpx = list(data.iloc[i,0:p])
        tmpx.insert(0,1.0)
        xMat.append(tmpx)

    yMat = list(data.iloc[:,p])

    return xMat, yMat
    
def GradientAscent(xMat,yMat):
    """Estimate regression coefficients using gradient ascent and maximum likelihood"""
    
    xMatrix = np.mat(xMat)
    yMatrix = np.mat(yMat)
    n, p = np.shape(xMatrix)
    
    # Set initial coefficients, learning rate (step size), number of iterations
    alpha = 0.0001
    cyclesCnt = 500
    weights = np.ones((p,1))
    
    for k in range(0,cyclesCnt):
        g = sigmoid(xMatrix * weights)
        error = (yMatrix.transpose() - g)
        # Gradient ascent to solve the maximum point/iterative update
        weights = weights + alpha * xMatrix.transpose() * error
    
    return weights

def labels(weights,xMat,yMat,threshold):

    xMatrix = np.mat(xMat)
    yMatrix = np.mat(yMat)
    yMatrix = yMatrix.tolist()
    probs = [x[0] for x in sigmoid(xMatrix*weights).tolist()]
    n = len(probs)
    y = [yMatrix[0][i] for i in range(0,n)]
    
    pre_label = [0]*n

    for i in range(0,n):
        if probs[i] >= threshold:
            pre_label[i] = 1
        else:
            pre_label[i] = 0
            
    return probs, pre_label, y

def score(predY,realY,probs,weights):
    n = 0
    for i in range(0,len(predY)):
        if predY[i]==realY[i]:
            n += 1
    AccurateRate = n/len(predY)
    print("Total Acc: ",(AccurateRate))
    
    # null deviance & residual deviance
    probs = np.array(probs)
    realY = np.array(realY)
    nullY = np.mean(realY)
    residual_deviance = sum(-2*(realY*np.log(probs)+(1-realY)*np.log(1-probs)))
    null_deviance = sum(-2*(nullY*np.log(probs)+(1-nullY)*np.log(1-probs)))
    print("null_deviance:",(null_deviance))
    print("residual_deviance:",(residual_deviance))

    # Define pseudo R^2
    R_squared = 1-residual_deviance/null_deviance
    print("R_squared:",(R_squared))
    
    # AIC:-2LogL+2(Number of independent variables + number of classification species -1)
    # SC:-2LogL+2(Number of independent variables + number of classification species -1)*ln(n_sample)
    xCnt = np.shape(xMat)[1]-1
    objCnt = np.shape(xMat)[0]
    AIC = residual_deviance+2*(xCnt+2-1)
    SC = residual_deviance+2*(xCnt+2-1)*np.log(objCnt)
    print("AIC:",(AIC))
    print("SC:",(SC))
    
if __name__=='__main__':
    start = time.time()
    xMat, yMat = loadData(filename)
    weights = GradientAscent(xMat,yMat)
    probs, pre_label, y = labels(weights,xMat,yMat,threshold)
    score(pre_label,y,probs,weights)
    end = time.time()
    print('time span:', end - start, 's')
    