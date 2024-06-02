#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    LogisticRegressionLossL2,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
    InverseLR,
    InverseSqrtLR,
    InverseSquaredLR,
)
from utils import (
    load_dataset,
    load_trainval,
    load_and_split,
    plot_classifier,
    savefig,
    standardize_cols,
    handle,
    run,
    main,
)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Standard (regularized) logistic regression
    loss_fn = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    lr_model = LinearClassifier(loss_fn, optimizer)
    lr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(lr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(lr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(lr_model, X_train, y_train)
    savefig("logRegPlain.png", fig)

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegLinear.png", fig)


@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    """YOUR CODE HERE FOR Q1.1"""
    loss_fn = LogisticRegressionLossL2(0.01)
    optimizer = GradientDescentLineSearch()
    poly_kernel = PolynomialKernel(2)
    poly_klr_model = KernelClassifier(loss_fn, optimizer, poly_kernel)
    poly_klr_model.fit(X_train, y_train)

    print(f"Poly Kernel - Training error: {np.mean(poly_klr_model.predict(X_train) != y_train):.1%}")
    print(f"Poly Kernel - Validation error: {np.mean(poly_klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(poly_klr_model, X_train, y_train)
    savefig("logRegPolyKernel.png", fig)


    rbf_kernel = GaussianRBFKernel(sigma=0.5)
    rbf_klr_model = KernelClassifier(loss_fn, optimizer, rbf_kernel)
    rbf_klr_model.fit(X_train, y_train)

    print(f"RBF Kernel - Training error: {np.mean(rbf_klr_model.predict(X_train) != y_train):.1%}")
    print(f"RBF Kernel - Validation error: {np.mean(rbf_klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(rbf_klr_model, X_train, y_train)
    savefig("logRegRBFKernel.png", fig)










@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")
    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val

    best_train_error = float("inf")
    best_val_error = float("inf")
    best_train_params = (None, None)
    best_val_params = (None, None)

    """YOUR CODE HERE FOR Q1.2"""
    for i, sigma in enumerate(sigmas):
      for j, lammy in enumerate(lammys):
        loss_fn = KernelLogisticRegressionLossL2(lammy)
        optimizer = GradientDescentLineSearch()
        rbf_kernel = GaussianRBFKernel(sigma)
        model = KernelClassifier(loss_fn, optimizer, rbf_kernel)
        model.fit(X_train, y_train)

        train_error = np.mean(model.predict(X_train) != y_train)
        val_error = np.mean(model.predict(X_val) != y_val)
        train_errs[i, j] = train_error
        val_errs[i, j] = val_error

        if train_error < best_train_error:
            best_train_error = train_error
            best_train_params = (sigma, lammy)
        
        if val_error < best_val_error:
            best_val_error = val_error
            best_val_params = (sigma, lammy)
    
    # Plot decision boundaries for the best hyperparameters
     # Plot decision boundaries for the best validation hyperparameters
    sigma, lammy = best_val_params
    rbf_kernel = GaussianRBFKernel(sigma)
    loss_fn = KernelLogisticRegressionLossL2(lammy)
    model = KernelClassifier(loss_fn, optimizer, rbf_kernel)
    model.fit(X_train, y_train)
    fig = plot_classifier(model, X_train, y_train)
    savefig("logRegRBF_best_val.png", fig)
    print("Best val_sigma:", sigma)
    print("Best va_lammy:", lammy)
    

    sigma, lammy = best_train_params
    rbf_kernel = GaussianRBFKernel(sigma)
    loss_fn = KernelLogisticRegressionLossL2(lammy)
    model = KernelClassifier(loss_fn, optimizer, rbf_kernel)
    model.fit(X_train, y_train)
    fig = plot_classifier(model, X_train, y_train)
    savefig("logRegRBF_best_train.png", fig)
    print("Best train_sigma:", sigma)
    print("Best train_lammy:", lammy)

    print(f"RBF Kernel - Training error: {np.mean(model.predict(X_train) != y_train):.1%}")
    print(f"RBF Kernel - Validation error: {np.mean(model.predict(X_val) != y_val):.1%}")


    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    savefig("logRegRBF_grids.png", fig)








@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    savefig("animals_matrix.png", fig)
    plt.close(fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples


    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(animal_names[i], xy=xy)
    savefig("animals_random.png", fig)
    plt.close(fig)

    """YOUR CODE HERE FOR Q3"""

    encoder = PCAEncoder(k=2)
    encoder.fit(X_train)
    Z = X_train_standardized @ encoder.W.T 



    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1])
    for i in random_is:
        ax.annotate(animal_names[i], (Z[i, 0], Z[i, 1]))
    savefig("animals_pca.png", fig)
    plt.close(fig)



    # Identifying influential traits
    first_pc = encoder.W[0]
    second_pc = encoder.W[1]
    trait_1st_pc = trait_names[np.argmax(np.abs(first_pc))]
    trait_2nd_pc = trait_names[np.argmax(np.abs(second_pc))]

    # Assuming pca_encoder is your trained PCAEncoder instance
    # and trait_names is the list of trait names

    # Loadings for the principal components
    pc1_loadings = encoder.W[0]
    pc2_loadings = encoder.W[1]

    # Identifying the trait with maximum influence on each principal component
    pc1_max_trait_index = np.argmax(np.abs(pc1_loadings))
    pc2_max_trait_index = np.argmax(np.abs(pc2_loadings))

    pc1_max_trait = trait_names[pc1_max_trait_index]
    pc2_max_trait = trait_names[pc2_max_trait_index]

    print("Trait with largest influence on first principal component:", pc1_max_trait)
    print("Trait with largest influence on second principal component:", pc2_max_trait)


  #3.2 a
    # Compute the low-dimensional representation Z
    Z = X_train_standardized @ encoder.W.T

    # Reconstruct the high-dimensional data from Z and W
    X_reconstructed = Z @ encoder.W

    # Compute the Frobenius norm of the reconstruction error
    reconstruction_error = np.linalg.norm(X_reconstructed - X_train_standardized, 'fro')**2

    # Compute the Frobenius norm of the original data matrix
    total_variance = np.linalg.norm(X_train_standardized, 'fro')**2

    # Calculate the variance explained by PCA
    variance_explained = 1 - (reconstruction_error / total_variance)

    print("Variance Explained by the First Two Principal Components:", variance_explained)



  #3.3 b
    # Initialize the variance explained
    variance_explained = 0
    num_components = 0

    # Iterate over the principal components
    for i in range(d):
        # Reconstruct the data using the first i principal components
        encoder = PCAEncoder(k=i)
        encoder.fit(X_train_standardized)
        Z = X_train_standardized @ encoder.W.T 
        X_reconstructed = Z @ encoder.W
        reconstruction_error = np.linalg.norm(X_reconstructed - X_train_standardized, 'fro')**2
        
        # Calculate the variance explained by the first i principal components
        variance_explained = 1 - (reconstruction_error / total_variance)
        
        # Check if we have explained at least 50% of the variance
        if variance_explained >= 0.5:
            num_components = i
            break

    print("Number of PCs required to explain at least 50% of variance:", num_components)


    return trait_1st_pc, trait_2nd_pc












@handle("4")
def q4():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    # Train ordinary regularized least squares
    loss_fn = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch()
    model = LinearModel(loss_fn, optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs)  # ~700 seems to be the global minimum.

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    savefig("gd_line_search_curve.png", fig)


@handle("4.1")
def q4_1():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    """YOUR CODE HERE FOR Q4.1"""
    # Define loss function and learning rate
    loss_fn = LeastSquaresLoss()
    learning_rate = 0.0003

    # Batch sizes to try
    batch_sizes = [1, 10, 100]
    epochs = 10  # Set the number of epochs
    

    for batch_size in batch_sizes:
        print(f"Batch Size: {batch_size}")

        # Initialize Stochastic Gradient with GradientDescent as base optimizer
        base_optimizer = GradientDescent()
        lr_getter = ConstantLR(learning_rate)
        optimizer = StochasticGradient(base_optimizer, lr_getter, batch_size)

        # Initialize and train the model
        model = LinearModel(loss_fn, optimizer)
        model.fit(X_train, y_train)


        # Calculate and print errors
        train_mse = ((model.predict(X_train) - y_train) ** 2).mean()
        val_mse = ((model.predict(X_val) - y_val) ** 2).mean()
        print(f"Training MSE: {train_mse:.3f}")
        print(f"Validation MSE: {val_mse:.3f}\n")





@handle("4.3")
def q4_3():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    loss_fn = LeastSquaresLoss()
    batch_size = 10
    c = 0.1

    """YOUR CODE HERE FOR Q4.3"""
    learning_rate_strategies = [ConstantLR(c), InverseLR(c), InverseSquaredLR(c), InverseSqrtLR(c)]
    labels = ["ConstantLR", "InverseLR", "InverseSquaredLR", "InverseSqrtLR"]

    plt.figure(figsize=(10, 6))

    for lr_strategy, label in zip(learning_rate_strategies, labels):
        # Initialize base optimizer (assuming you have a GradientDescent class)
        base_optimizer = GradientDescent()

        # Initialize Stochastic Gradient optimizer
        stochastic_optimizer = StochasticGradient(base_optimizer, lr_strategy, batch_size)

        # Initialize the model
        model = LinearModel(loss_fn, stochastic_optimizer)

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Plot the learning curve
        plt.plot(model.fs, label=label)

    plt.xlabel("Number of SGD epochs")
    plt.ylabel("Objective function f value")
    plt.title("Learning Curves with Different Learning Rates")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
