
import numpy as np
import pandas as pd
import random
import sys


def distance_sqrt(X,ctds):
    X_sum = np.sum(np.square(X),axis=1);
    Y_sum = np.sum(np.square(ctds),axis=1);
    multiply = np.dot(X, ctds.T);
    dist = np.sqrt(abs(X_sum[:, np.newaxis] + Y_sum-2*multiply))
    return dist


def dist(data, centers):
    distance = np.sum((np.array(centers) - data[:, None, :])**2, axis = 2)
    return distance

def kmeans_plus_plus(X, k):
    ctds = []
    X = np.array(X)

    first_idx = np.random.choice(range(X.shape[0]), )
    ctds.append(X[first_idx, :].tolist())
    
    for i in range(k - 1):
        distance = dist(X, np.array(ctds))
        min_dist = np.min(distance, axis = 1)
        max_idx = np.argmax(min_dist, axis = 0)
        new_ctds = X[max_idx, :]
        ctds.append(new_ctds.tolist())
        
    return np.array(ctds)



def kmeans(X,k,centroids=None, max_iter=30, tolerance=0.001):
    
    
    if centroids is None:
        random_index = np.random.choice(len(X), k, replace=False)
        ctds = X[random_index, :]
        ctds= np.array(ctds)

            
    elif centroids=='kmeans++':
        
        ctds=kmeans_plus_plus(X, k)
        ctds= np.array(ctds)
        

    for i in range(max_iter):
        
        distances=distance_sqrt(X,ctds)
        closest_ctd = np.argmin(distances, axis=1)         

        K, D = ctds.shape
        new_ctds = np.empty(ctds.shape)
        for j in range(ctds.shape[0]):
            
            new_ctds[j] = np.mean(X[closest_ctd == j], axis = 0)
 

        ctds=new_ctds

        distances=distance_sqrt(X,ctds)
        closest_ctd = np.argmin(distances, axis=1)

    return ctds,closest_ctd
                    




