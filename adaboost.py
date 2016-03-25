# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:26:40 2015

@author: zhangm215
"""

import numpy as np

def loadSimpData():
    dataMat =  np.matrix([[1. , 2.1],
                          [2. , 1.1],
                          [1.3, 1. ],
                          [1. , 1. ],
                          [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels
    
def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMat)[0],1))
    if threshIneq == 'lt':
        retArray[dataMat[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMat[:,dimen] > threshVal] = -1.0
        
def buildStump(dataArr, calssLabels, D):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        