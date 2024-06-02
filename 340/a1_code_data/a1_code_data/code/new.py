import numpy as np
import utils


class DecisionStumpEquality:
    """
    This is a decision stump that branches on whether the value of X is
    "almost equal to" some threshold.

    This probably isn't a thing you want to actually do, it's just an example.
    """

    y_hat_yes = 0
    y_hat_no = 0
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = np.round(X[i, j])

                # Find most likely class for each split
                is_almost_equal = np.round(X[:, j]) == t
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[np.round(X[:, j]) != t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode




    def predict(self, X):
        n, d = X.shape
        X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n) # creates an array of size n with all values set to self.y_hat_yes.

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] == self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat
    

   



################################################################################################################

# def fit(self, X, y):
#         """YOUR CODE HERE FOR Q6.2"""
#         #number of features and exmples. num of row and col
#         n, d = X.shape

#         #  Get an array with the number of each class. ex) array with the number of 0's, number of 1's, etc.
#         count = np.bincount(y)

#         # Get the index of the largest value in an array (count) by argmax.
#         # Thus, y_mode is the mode (most popular value) of y
#         y_mode = np.argmax(count)

#         self.y_hat_yes = y_mode
#         self.y_hat_no = 0
#         self.j_best = None
#         self.t_best = None

#         # If all the labels are the same, no need to split further
#         if np.unique(y).size <= 1:
#             return
        
#         #A minimum error is initialized to the number of samples that don't match the mode of y.
#         #  This error will be updated if a better split is found.
#         minError = np.sum(y != y_mode)



#         for j in range(d):  # For each feature
#             thresholds = np.unique(X[:, j])  # Potential thresholds

#             for t in thresholds:  # For each threshold
#                 # Split data based on threshold
#                 y_left = y[X[:, j] <= t]
#                 y_right = y[X[:, j] > t]
                
#                 # Determine majority class on each side

#                 if y_left.size != 0:
#                     y_left_mode = utils.mode(y_left)
#                 else:
#                     y_left_mode = utils.mode(y)
            
               
#                 if y_right.size != 0:
#                      y_right_mode = utils.mode(y_right)
#                 else:
#                      y_right_mode =  utils.mode(y)
                
#                 # Predictions
#                 y_pred = np.where(X[:, j] <= t, y_left_mode, y_right_mode)
                
#                 # Compute error
#                 errors = np.sum(y_pred != y)
                
                
#                 # Check if this stump gives lower error than previous best
#                 if errors < minError:
#                     minError = errors
#                     self.j_best = j
#                     self.t_best = t
#                     self.y_hat_yes = y_left_mode
#                     self.y_hat_no = y_right_mode


       # raise NotImplementedError()



class DecisionStumpErrorRate:
    y_hat_yes = 1
    y_hat_no = 1
    j_best = None
    t_best = None
    def fit(self, X, y):
        n, d = X.shape
        num_classes = len(np.unique(y)) 

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y, minlength=num_classes)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = np.round(X[i, j])

                # Find most likely class for each split
                is_el = np.round(X[:, j]) <= t
                y_yes_mode = utils.mode(y[is_el])
                y_no_mode = utils.mode(y[~is_el])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[np.round(X[:, j]) > t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

        
    def predict(self, X):
        """YOUR CODE HERE FOR Q6.2"""
        n, d = X.shape
        X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n) # creates an array of size n with all values set to self.y_hat_yes.

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] <= self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat




################################################################################################################



    


#X, which is the data on which we want to make predictions. 
# The shape will tell us the number of samples (n) and the number of features (d).
#X would be a matrix of size n x d
#X is rounded off because  the decision stump checks if the values are "almost equal"

def entropy(p):
    """
    A helper function that computes the entropy of the
    discrete distribution p (stored in a 1D numpy array).
    The elements of p should add up to 1.
    This function ensures lim p-->0 of p log(p) = 0
    which is mathematically true, but numerically results in NaN
    because log(0) returns -Inf.
    """
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)


class DecisionStumpInfoGain(DecisionStumpErrorRate):
    # This is not required, but one way to simplify the code is
    # to have this class inherit from DecisionStumpErrorRate.
    # Which methods (init, fit, predict) do you need to overwrite?
    y_hat_yes = 0
    y_hat_no = 0
    #index of the best feature (column in X) on which the decision stump will split the data.
    j_best = None
    # the best threshold value for splitting the data based on the feature j_best.
    t_best = None

    """YOUR CODE HERE FOR Q6.3"""


    def fit(self, X, y):
        n, d = X.shape   # how is the number of data points and number of cols are the same ??
        # get number of unique classes in y 
        num_classes = len(np.unique(y)) 

         # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        # Entropy before any split

        #if y has labels {0, 1, 2}, and y = [0, 1,1, 2, 2, 1], then np.bincount(y) would give [1, 3, 2],
        #  meaning class 0 appears once, class 1 appears three times, and class 2 appears twice.
        count_before_split = np.bincount(y, minlength=num_classes)
        # the probability distribution of the classes, you'd divide the count of each class 
        # by the total number of data points, n
        #[1/5, 3/5, 2/5]
        p_before_split = count_before_split / n
        entropy_before_split = entropy(p_before_split)




        # start with min infogain. update this if we find higher infogain
        max_info_gain = 0


        #  loop through all potential splits
        for j in range(d):
            #X[:, j] selects all rows (due to the :) of the j-th column of the matrix X
            # get unique elements of an array. ex) if a col has [2.5, 3.0, 2.5, 4.0, 4.0], then thresholds will be [2.5, 3.0, 4.0].
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                # Split data
                #y = [0, 1, 0, 1, 1] 
                y_left = y[X[:, j] <= t]  #[0, 1, 1]
                y_right = y[X[:, j] > t]  #[0, 1]


                
                # Calculate entropies after the split
                count_left = np.bincount(y_left, minlength=num_classes)
                p_left = count_left / len(y_left)
                entropy_left = entropy(p_left)
                
                count_right = np.bincount(y_right, minlength=num_classes)
                p_right = count_right / len(y_right)
                entropy_right = entropy(p_right)



                
                # Weighted average of the entropies of tbe subsets
                # (n1/n)*split entropy
                w_avg_entropy = ((len(y_left)/ n) * entropy_left) + ((len(y_right)/n) * entropy_right) 
                
                # Calculate information gain
                #total entropy - Weighted average of the entropies of tbe subsets
                info_gain = entropy_before_split - w_avg_entropy
                
                # If this split results in a higher infogain, save it since #
                # the current feature being evaluated is found to be the best one for splitting
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    
                    self.j_best = j
                    self.t_best = t
                    if len(y_left) > 0:
                        self.y_hat_yes = utils.mode(y_left)
                    
                    if len(y_right) > 0:
                        self.y_hat_no = utils.mode(y_right)
                   






        
    def hard_coded_predict(self, X):
        n, d = X.shape
        y_hat = np.zeros(n)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)
        
        for i in range(n):
            if X[i, 0] > -80.305106:
                y_hat[i] = self.y_hat_yes
                if X[i, 1] > 36.453576:
                   y_hat[i] = self.y_hat_yes
                else:
                   y_hat[i] = self.y_hat_no
            else:
                if X[i, 1] > 37.669007:
                   y_hat[i] = self.y_hat_yes
                else:
                   y_hat[i] = self.y_hat_no 
                   
        return y_hat


 