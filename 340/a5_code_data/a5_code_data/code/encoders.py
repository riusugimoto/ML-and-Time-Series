import numpy as np

"""
Contains class definitions related to latent factor models, whose behaviours are
encapsulated by the "learned encoders", which are objects implementing encode() method.
"""


class LinearEncoder:
    """
    Latent factor models that can "encode" X into Z, and "decode" Z into X based on latent factors W.
    """

    mu = None
    W = None

    def encode(self, X):
        """
        Use the column-wise mean and principal components to
        compute the "component scores" to encode
        """
        X = X - self.mu
        return X @ self.W.T

    def decode(self, Z):
        """
        Transforms "component scores" back into the original data space.
        """
        return Z @ self.W + self.mu


class PCAEncoder(LinearEncoder):
    """
    Solves the PCA problem min_{Z,W} (Z*W - X)^2 using SVD
    """

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        """
        Learns the principal components by delegating to SVD solver.
        "Fitting" here is the matter of populating:
        self.mu: the column-wise mean
        self.W: the principal components
        """
        self.mu = np.mean(X, axis=0)
        X = X - self.mu

        U, s, Vh = np.linalg.svd(X)
        self.W = Vh[: self.k]





# Standardization of Data (X_train_standardized): PCA is sensitive to the scales of the variables, \
# so it's a common practice to standardize the data first. This involves subtracting the mean and dividing by the standard deviation for each feature.
#  This process ensures that each feature contributes equally to the analysis.

# Singular Value Decomposition (SVD): In the PCAEncoder class, the SVD of the standardized data X is computed. SVD decomposes a matrix into three other matrices: 
# U,s, and VH (hermitian or transposed). In the context of PCA, VH (denoted as W in your code) contains the principal components.

# Principal Components (W or Vh): The rows of VH (or W) are the principal components of the data. These components are the directions in the feature space that maximize the variance of the projected data. The first row of W is the first principal component, the second row is the second principal component, and so on.

# Projection onto Principal Components (Z): When you compute Z = X_train_standardized @ encoder.W.T, you're essentially projecting the standardized data onto the space defined by the principal components. Here, encoder.W.T is the transpose of W, which aligns the principal components correctly for the dot product with X_train_standardized. This operation transforms the data into a new space defined by the principal components, reducing its dimensionality while retaining the most significant variance in the data.

# In this case, since k=2 in your PCA encoder, W will have two rows (the first two principal components), and thus Z will be a 2-dimensional representation of your original data.





# Not exactly. The variance explained by the principal components in PCA doesn't directly measure how "different" the reduced dimensionality data is from the original data. 
# Rather, it measures how much of the original data's total variance is captured by those principal components.

# Here's a more detailed explanation:

# Total Variance: In the context of PCA, the total variance of the dataset is the sum of the variances of each feature in the dataset. 
# It represents the total "spread" of the data in all directions in the feature space.

# Explained Variance: When you perform PCA, each principal component captures some portion of the total variance.
#  The first principal component captures the most, the second captures the second most, and so on. The explained variance is the amount of total variance that is captured by the selected principal components. 
# This is often expressed as a percentage of the total variance.

# Unexplained Variance or Reconstruction Error: The variance that is not captured by the selected principal components is the unexplained variance or reconstruction error. 
# This is what is left over after projecting the data onto the lower-dimensional space defined by the principal components.

# When we say that the first two principal components explain, for example, 50% of the variance, it means that half of the total variability of the original data can be described using just these two dimensions.
#  The other half of the variability cannot be captured by these components and would require additional components to be more fully described.

# The goal in PCA is often to find the smallest number of principal components that still capture a large proportion of the total variance, thus simplifying the dataset while retaining most of the important information.