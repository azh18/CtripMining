import basicmining
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
import json

def userClustering(userFeature):
    km = KMeans(n_clusters=1000, init='k-means++', max_iter=300, n_init=1, verbose=False)
    X = np.array(userFeature)
    X_norm = (X-np.amin(X))/(np.amax(X)-np.amin(X))
    km.fit(X_norm)
    result = list(km.predict(X_norm))

if __name__ == '__main__':
    with open("userFeature.json") as json_file:
        userFeature = json.load(json_file)
    userClustering(userFeature)



