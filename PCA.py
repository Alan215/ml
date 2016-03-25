#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim = ' '):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [map(float, line) for line in stringArr]
    return np.mat(dataArr)

def PCA(dataMat, topMfeat = 9999999):
    #normalize the data
    #去均值，用标准差把数据放缩到同一规模
    meanVals = np.mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals
    #协方差
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVecs = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topMfeat+1):-1]
    redEigVecs =eigVecs[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVecs
    reconMat = (lowDDataMat * redEigVecs.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean(dataMat):
    numFeat = np.shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:,i].A))[0],i])
        dataMat[np.nonzero(np.isnan(dataMat[:,i].A))[0],i] = meanVal
    return dataMat

def pltPCA(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

dim = 2
data = [[0,0,0],[2,0,0],[2,0,1],[1,2,0],[0,0,1],[0,1,0],[0,-2,1],[1,1,-2]]
dataMat = np.mat(data)
# dataMat = loadDataSet("data/secom.data")
# dataMat = replaceNanWithMean(dataMat)
lowDMat, reconMat = PCA(dataMat, dim)
pltPCA(dataMat, reconMat)
print lowDMat