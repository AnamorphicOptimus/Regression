#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridge and Lasso Regression
"""
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time as time

filename = "./Hitters.csv"

def standardization(data):
    '''
    input:numpy array
    output:numpy array
    
    '''
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def loadData(filename):
    '''
    input: filepath
    return: x_Matrix,y_Matrix(1*n)

    '''
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

    xMat = np.matrix(xMat)
    yMat = np.matrix(yMat)
    return xMat, yMat

def ridge(xMat,yMat,lambd = 2,Standardization = False):
	"""Ridge Regression Normal Equation Solving"""
    if Standardization:
        xMat = standardization(xMat)        
    ws = (xMat.T*xMat+np.eye(np.shape(xMat)[1])*lambd).I*(xMat.T*yMat.T)
    
    return ws

def lasso_regression(X, yMat, lambd=0.2, threshold=0.1):
    ''' 
    lasso optimization: coordinate axis descent method
        
    '''
    # Calculate the residual sum of squares
    rss = lambda X, y, w: (y - X*w).T*(y - X*w)
    y = yMat.T
    m, n = X.shape
    w = np.matrix(np.zeros((n,1)))
    r = rss(X, y, w)
    niter = itertools.count(1)
    for it in niter:
        for k in range(n):
            z_k = (X[:, k].T*X[:, k])[0, 0]
            p_k = 0
            for i in range(m):
                p_k += X[i, k]*(y[i, 0] - sum([X[i, j]*w[j, 0] for j in range(n) if j != k]))
            if p_k < -lambd/2:
                w_k = (p_k + lambd/2)/z_k
            elif p_k > lambd/2:
                w_k = (p_k - lambd/2)/z_k
            else:
                w_k = 0
            w[k, 0] = w_k
        r_prime = rss(X, y, w)
        delta = abs(r_prime - r)[0, 0]
        r = r_prime
        print('Iteration: {}, delta = {}'.format(it, delta))
        if delta < threshold:
            break
    return w

def tp_choose(X,yMat,lambd_list = None,modelName = "ridge"):
    '''Based on MSE + Cross validation to find the lambda in the optimal cost function'''
    y = yMat.T
    MSE = lambda X, y, w: (y - X*w).T*(y - X*w)/np.shape(X)[0]
    
    if lambd_list is None:
        lambd_list = list(range(1,100))
        for i in range(-3,11):
            lambd_list.append(10**i)
            
    if modelName =="ridge":
        ridgeMse = []
        for lam in lambd_list:
            Rweights = ridge(X,yMat,lambd=lam)
            ridgeMse.append(MSE(X,y,Rweights)[0,0])
           
        tar_ridgeTp = lambd_list[ridgeMse.index(min(ridgeMse))]      
        return tar_ridgeTp
    
    elif modelName =="lasso":
        lassoMse = []
        for lam in lambd_list:
            Lweights = lasso_regression(X,y.T,lambd=lam)       
            lassoMse.append(MSE(X,y,Lweights)[0,0])
            
        tar_lassoTp = lassoMse[lassoMse.index(min(lassoMse))]
        return tar_lassoTp
    else:
        print("The model doesn't exist")
        
def ModelScore(xMat,yMat,weights,modelname = "Model"):  
    print("="*50)
    print(modelname)
    yPred = xMat*weights
    yMat = yMat.T

    # RSE = np.sqrt(np.sum(np.square(yPred-yMat))/(n-p-1))
    n = np.shape(xMat)[0]#观测数量
    p = np.shape(xMat)[1]-1
    RSS = np.sum(np.square(yPred-yMat))
    TSS = np.sum(np.square(yMat-np.mean(yMat)))
   
    MSE = RSS/n
    R_squared = 1-RSS/TSS
    Adjusted_R_squared = 1-(RSS/(n-p-1))/(TSS/(n-1))
    corrcoef = np.corrcoef([yPred.tolist()[i][0] for i in range(n)],
                            [yMat.tolist()[i][0] for i in range(n)])[0,1]
    # AIC
    # errorVar = np.var(yPred-yMat)
    # spss16.0版本AIC、BIC
    AIC = n*np.log(RSS)+(2*p+2-n)*np.log(n)
    BIC = n*np.log(RSS)+(p+1-n)*np.log(n)

    print("MSE:",MSE)
    print("R_squared:",R_squared)
    print("Adjusted_R_squared:",Adjusted_R_squared)
    print("corrcoef:",corrcoef)
    print("AIC:",AIC)
    print("BIC:",BIC)


def ModelPlt(xMat,yMat,modelName = "ridge",weights = None,numTest = 30):
    '''
    weights: x of the plot
    numtest: the number of x
    xMat: n*p
    yMat: 1*n
    modelName: "ridge" / "lasso"
    return: the plot of parameter shrinkage 
    
    '''
    wMat = np.zeros((numTest+1,xMat.shape[1]))
    lambd_list = [0]
    for i in range(numTest):
        lambd_list.append(np.exp(i-10))
        
    if weights is None:    
        if modelName == "lasso":    
            for i in range(numTest+1):
                ws = lasso_regression(xMat,yMat,lambd = lambd_list[i])
                wMat[i,:]=ws.T  
        elif modelName == "ridge": 
            for i in range(numTest+1):
                ws = ridge(xMat,yMat,lambd = lambd_list[i])
                wMat[i,:]=ws.T
        else:
            print("The model doesn't exist!")
    else:
        if modelName == "lasso":    
            for w in weights:
                ws = lasso_regression(xMat,yMat,lambd = w)
                wMat[i,:]=ws.T  
        elif modelName == "ridge": 
            for w in weights:
                ws = ridge(xMat,yMat,lambd = w)
                wMat[i,:]=ws.T
        else:
            print("The model doesn't exist!")

    plt.plot(wMat)
    plt.xlabel('log(lambda)')
    plt.ylabel('weights')
        
    
if __name__=='__main__':
    
    ## ridge
    # Read data conversion format
    xMat, yMat = loadData(filename)
    # xMat = np.mat(preprocessing.scale(xMat))

    # choose best param
    ridge_lam = tp_choose(xMat,yMat,modelName="ridge")
    print("Best ridge lam:",ridge_lam)

    # Estimate coefficients
    start = time.time()
    weights = ridge(xMat,yMat,lambd = ridge_lam)
    end = time.time()
    print(' Ridge time span:', end - start, 's')

    # print the scores
    ModelScore(xMat,yMat,weights= weights,modelname = "ridge")
    ModelPlt(xMat,yMat,modelName="ridge")
    
    ## lasso
    lasso_lam = tp_choose(xMat,yMat,modelName="lasso")
    #lasso_lam = 4
    
    # Estimate coefficients
    start = time.time()
    lasso_weights = lasso_regression(xMat,yMat,lambd = lasso_lam)
    end = time.time()
    print('lasso time span:', end - start, 's')

    # print the scores
    ModelScore(xMat,yMat,lasso_weights,modelname = "lasso")
    ModelPlt(xMat,yMat,modelName="lasso")















