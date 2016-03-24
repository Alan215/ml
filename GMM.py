import kmeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

#super parameter
def GMM(K, data, stop_times):
    num = data.shape[0]
    dim = data.shape[1]

    clusters = kmeans.kmeans(K, data, 100)

    mul = np.zeros((K, dim))
    cov = np.zeros((K, dim, dim))
    for i in range(K):
        mul[i] = np.mean(clusters[i])
        cov[i] = np.cov(clusters[i])
    #latent variable
    z = np.zeros((num, K))

    times = 0
    while(times<stop_times):
        #compute
        p = np.zeros((num, K))
        #E step
        for i in range(num):
            nor = 0
            for j in range(K):
                p[i,j] = np.exp(-1/2*(data[i]-mul[j]).dot(inv(cov[j])).dot((data[i]-mul[j]).T))*det(cov[j])**(-1/2)
            for j in range(K):
                z[i,j] = p[i,j]/np.sum(p[i,:])
        #M step
        mul = z.T.dot(data),0/np.sum(z,axis=0)
        for j in range(K):
            tmp = np.zeros((dim,dim))
            for i in range(num):
                tmp += z[i,j]*np.outer(data[i],data[i])
            cov[j] = tmp/np.sum(z[:,j])
        times += 1


    z1 = np.diag(np.exp(-1/2*(data-mul1).dot(inv(cov1)).dot((data-mul1).T))/((2*np.pi)*det(cov1)**(1/2)))
    z2 = np.diag(np.exp(-1/2*(data-mul2).dot(inv(cov2)).dot((data-mul2).T))/((2*np.pi)*det(cov2)**(1/2)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(data[:,0], data[:,1], z1, color='blue')
    ax.plot_trisurf(data[:,0], data[:,1], z2, color='red')
    ax.set_xlabel('X')
    # ax.set_xlim(-40, 40)
    ax.set_ylabel('Y')
    # ax.set_ylim(-40, 40)
    ax.set_zlabel('Z')
    # ax.set_zlim(-100, 100)

    plt.show()
