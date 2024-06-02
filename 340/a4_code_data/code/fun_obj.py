import numpy as np
from scipy.special import logsumexp
from scipy.optimize.optimize import approx_fprime

from utils import ensure_1d

"""
Implementation of function objects.
Function objects encapsulate the behaviour of an objective function that we optimize.
Simply put, implement evaluate(w, X, y) to get the numerical values corresponding to:
f, the function value (scalar) and
g, the gradient (vector).

Function objects are used with optimizers to navigate the parameter space and
to find the optimal parameters (vector). See optimizers.py.
"""


class FunObj:
    """
    Function object for encapsulating evaluations of functions and gradients
    """

    def evaluate(self, w, X, y):
        """
        Evaluates the function AND its gradient w.r.t. w.
        Returns the numerical values based on the input.
        IMPORTANT: w is assumed to be a 1d-array, hence shaping will have to be handled.
        """
        raise NotImplementedError("This is a base class, don't call this")

    def check_correctness(self, w, X, y):
        n, d = X.shape
        w = ensure_1d(w)
        y = ensure_1d(y)

        estimated_gradient = approx_fprime(
            w, lambda w: self.evaluate(w, X, y)[0], epsilon=1e-6
        )
        _, implemented_gradient = self.evaluate(w, X, y)
        difference = estimated_gradient - implemented_gradient
        if np.max(np.abs(difference) > 1e-4):
            print(
                "User and numerical derivatives differ: %s vs. %s"
                % (estimated_gradient, implemented_gradient)
            )
        else:
            print("User and numerical derivatives agree.")


class LeastSquaresLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of least squares objective.
        Least squares objective is the sum of squared residuals.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        y_hat = X @ w
        m_residuals = y_hat - y  # minus residuals, slightly more convenient here

        # Loss is sum of squared residuals
        f = 0.5 * np.sum(m_residuals ** 2)

        # The gradient, derived mathematically then implemented here
        g = X.T @ m_residuals  # X^T X w - X^T y

        return f, g


class RobustRegressionLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of ROBUST least squares objective.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        y_hat = X @ w
        residuals = y - y_hat
        exp_residuals = np.exp(residuals)
        exp_minuses = np.exp(-residuals)

        f = np.sum(np.log(exp_minuses + exp_residuals))

        # s is the negative of the "soft sign"
        s = (exp_minuses - exp_residuals) / (exp_minuses + exp_residuals)
        g = X.T @ s

        return f, g


class LogisticRegressionLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of logistics regression objective.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply; the y_i are in {-1, 1}

        # Calculate the function value
        f = np.sum(np.log(1 + np.exp(-yXw)))

        # Calculate the gradient value
        s = -y / (1 + np.exp(yXw))
        g = X.T @ s

        return f, g


class LogisticRegressionLossL2(LogisticRegressionLoss):
    def __init__(self, lammy):
        super().__init__()
        self.lammy = lammy

    def evaluate(self, w, X, y):
        w = ensure_1d(w)
        y = ensure_1d(y)

        """YOUR CODE HERE FOR Q2.1"""
        pass


class LogisticRegressionLossL0(FunObj):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function value of of L0-regularized logistics regression objective.
        """
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply

        # Calculate the function value
        f = np.sum(np.log(1.0 + np.exp(-yXw))) + self.lammy * np.sum(w != 0)

        # We cannot differentiate the "length" function
        g = None
        return f, g


# First, you exponentiate each score to make all values positive and increase the difference between the higher score and others (because the exponential function grows very fast):
# where exp_scores is the vector of the same length as z with the exponential of each score.
# Then, you sum all of the exponentiated scores to normalize them:
# Lastly, you divide each exponentiated score by the sum of all exponentiated scores:


#The loss associated with the softmax function 
# This loss quantifies how well the predicted probability distribution matches the true distribution
#loss= -∑ yi log(softmax(zi)) yi  true class label, and softmax(zi) is the predicted probability of class i.

class SoftmaxLoss(FunObj):
    def evaluate(self, w, X, y):
        w = ensure_1d(w)
        y = ensure_1d(y)

        n, d = X.shape
        k = len(np.unique(y))

        """YOUR CODE HERE FOR Q3.4"""
        # Hint: you may want to use NumPy's reshape() or flatten()
        # to be consistent with our matrix notation.
         # Get the number of data points (n) and features (d)
        n, d = X.shape  # Number of samples and features
        k = len(np.unique(y))  # Number of classes

        
        W = np.reshape(w, (k, d))  # Reshape w to W with shape k x d


        # Compute the scores for all classes
        scores = np.dot(X, W.T)  # n x k

     

        # softmax probabilities part
        exponentialized_scores = np.exp(scores)
        softmax_probs = exponentialized_scores / np.sum(exponentialized_scores, axis=1, keepdims=True)

     
        true_class_scores = scores[np.arange(n), y]


        #the loss function f(W)
        f = -np.sum(true_class_scores) + np.sum(np.log(np.sum(exponentialized_scores, axis=1)))



        # Compute the gradient
        dscores = softmax_probs
        dscores[np.arange(n), y] -= 1
        
        g = np.dot(dscores.T, X)

        # Flatten the gradient matrix into a vector
        g = g.reshape(-1)

        return f, g








        # W = np.reshape(w, (k, d))  # Reshape w to W with shape k x d

        # # Initialize the objective function value and gradient
        # f = 0
        # g = np.zeros((k, d))

        # for i in range(n):
        #     scores = np.dot(W, X[i])  # Compute scores for each class
        #     correct_class_score = scores[y[i]]
        #     exp_scores = np.exp(scores)
        #     sum_exp_scores = np.sum(exp_scores)

        #     # Update the objective function
        #     f += -correct_class_score + np.log(sum_exp_scores)

        #     # Update the gradient
        #     for j in range(k):
        #         softmax_prob = exp_scores[j] / sum_exp_scores
        #         if j == y[i]:
        #             g[j] += (-1 + softmax_prob) * X[i]
        #         else:
        #             g[j] += softmax_prob * X[i]

        # # Flatten the gradient matrix into a vector
        # g = g.reshape(-1)
        # return f, g















