import kmeans
import GMM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/kmeans_data.csv",header=None)
data = df.values
#test kmeans
# clusters = kmeans.kmeans(5, data, 100)
# for i in range(len(clusters)):
#     print np.mean(clusters[i], axis=0)
#     print np.cov(clusters[i].T)


#test GMM
GMM.GMM(5,data,1)