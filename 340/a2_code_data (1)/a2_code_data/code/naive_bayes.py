import numpy as np


class NaiveBayes:
    """
    Naive Bayes implementation.
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...k-1
    """

    p_y = None
    p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        """YOUR CODE HERE FOR Q3.3"""

        # Initialize p_xy
        p_xy = np.zeros((d, k))
    
        for c in range(k):
            # Filter the samples that belong to class c
            X_c = X[y == c]
            
            p_xy[:, c] = X_c.sum(axis=0) / X_c.shape[0]
    
        self.p_y = p_y
        self.p_xy = p_xy


    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= 1 - p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred


class NaiveBayesLaplace(NaiveBayes):
    def __init__(self, num_classes, beta=0):
        super().__init__(num_classes)
        self.beta = beta

    def fit(self, X, y, beta=1): # default value bata = 1
    
        n, d = X.shape

        k = self.num_classes

        # Compute the probability of each class
        counts = np.bincount(y)
        p_y = counts / n

        """YOUR CODE FOR Q3.4"""

        # Initialize p_xy
        p_xy = np.zeros((d, k))
    
        for c in range(k):
            X_c = X[y == c]
            
            p_xy[:, c] = (X_c.sum(axis=0) + beta) / (X_c.shape[0] + 2 * beta)  

    
        self.p_y = p_y
        self.p_xy = p_xy


    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(n) 
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= 1 - p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred
    



        