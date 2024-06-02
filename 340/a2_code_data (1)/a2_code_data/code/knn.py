"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared
from collections import Counter

class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        """YOUR CODE HERE FOR Q1"""
        num_test = X_hat.shape[0]
        y_hat = np.zeros(num_test)

        for i in range(num_test):
            #distances = euclidean_dist_squared(X_hat[i:i+1], self.X)
            dist = np.sqrt(np.sum((self.X - X_hat[i])**2, axis=1))

            k_indices = np.argsort(dist)[:self.k] # sort dist array and grab indices of first k elements 
            k_labels = self.y[k_indices]
            
            y_hat[i] = np.bincount(k_labels).argmax() 

        return y_hat
    


        #raise NotImplementedError()

