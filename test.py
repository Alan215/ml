import kmeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/kmeans_data.csv")
data = df.values
clusters = kmeans.kmeans(5, data, 100)

color=['b','r','y','g']

for i in range(len(clusters)):
    plt.scatter(clusters[i][:,0], clusters[i][:,1], color=color[np.random.randint(len(color))])
plt.show()