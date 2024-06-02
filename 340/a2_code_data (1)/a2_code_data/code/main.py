#!/usr/bin/env python
import argparse
import os
import pickle
import utils
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    """YOUR CODE HERE FOR Q1. Also modify knn.py to implement KNN predict."""
    k_values = [1, 3, 10]
    for k in k_values:
        model = KNN(k)
        
        model.fit(X, y)

        # Predict using training data and compute training error
        y_train_pred = model.predict(X)
        train_error = np.mean(y_train_pred != y)
        
        # Predict using test data and compute test error
        y_test_pred = model.predict(X_test)
        test_error = np.mean(y_test_pred != y_test)
        
       
        
        print(f"For k = {k}:")
        print(f"Training error: {train_error}")
        print(f"Test error: {test_error}")
        print("--------------------------")


        # Plot using utils.plot_classifier
        utils.plot_classifier(model, X, y )
        plt.title(f"k-NN with k= {k} (Your Implementation)")
        plt.savefig("knn_your_implementation.png")  # Save the plot
        plt.show()




    #raise NotImplementedError()



@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    """YOUR CODE HERE FOR Q2"""
    ks = list(range(1, 30, 4))  #(1, 5, 9, 13,. . . , 29)
    n = X.shape[0]

    fold_size = n // 10
    cross_vali_accs = []

    for k in ks:
        knn = KNN(k)
        k_accs = []

        for i in range(10):
            # Create the mask for the validation fold
            mask = np.ones(n, dtype=bool)
            mask[i * fold_size: (i + 1) * fold_size] = False

            # Split the data into training and validation using the mask
            X_train = X[mask]
            y_train = y[mask]
            X_val   = X[~mask]
            y_val   = y[~mask]

          
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)

            accuracy = np.mean(y_pred == y_val)
            k_accs.append(accuracy)

        # Average accuracy across all 10 folds
        mean_acc = np.mean(k_accs)
        cross_vali_accs.append(mean_acc)

    # # cv_accs now contains the cross-validated accuracies for all ks
    # return cross_vali_accs



    test_accs = []  
    k_values = list(range(1, 30, 4))
    for k in k_values:
        model = KNN(k)
        model.fit(X, y)
        
        # Predict using test data and compute test error
        y_test_pred = model.predict(X_test)
        test_accuracy = np.mean(y_test_pred == y_test)

        test_accs.append(test_accuracy)

       

    plt.figure(figsize=(10, 6))
    #  cross-validation accuracies
    plt.plot(ks, cross_vali_accs, marker='o', linestyle='-', color='b', label="Cross Validation Accuracy")

    #  test accuracies
    plt.plot(ks, test_accs, marker='+', linestyle='--', color='r', label="Test Accuracy")

    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("kNN: Cross-validation and Test Accuracies vs. k")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
#    plt.show()
    fname = Path("..", "figs", "q2.png")
    plt.savefig(fname)



    train_errors = [] 
    test_errors  = []
    for k in k_values:
        model = KNN(k)
        model.fit(X, y)
        
        # Predict using test data and compute test error
        y_train_pred = model.predict(X)
        train_error = np.mean(y_train_pred != y)
        train_errors.append(train_error)


        y_test_pred = model.predict(X_test)
        test_error = np.mean(y_test_pred != y_test)
        test_errors.append(test_error)

    plt.figure(figsize=(10, 6))
    plt.plot(ks, train_errors, marker='o', linestyle='-', color='r', label="Train Error")
    plt.plot(ks, test_errors, marker='+', linestyle='--', color='b', label="Test Error")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.title("kNN: Train Error and Test Error vs. k")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    fname = Path("..", "figs", "q2.4.png")
    plt.savefig(fname)


    #raise NotImplementedError()















@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""
 # 1. Which word corresponds to column 73 of X? (This is index 72 in Python.)
    index_72 = wordlist[72]
    print("1. The word at column 73 (index 72 in Python) is:", index_72)

    # 2. Which words are present in training example 803 (Python index 802)?
    words_in_example_802 = []
    for i in range(len(X[802])):
        if X[802][i] == 1:
            words_in_example_802.append(wordlist[i])
    print("2. The words present in training example 803 (Python index 802) are:", words_in_example_802)

    # 3. Which newsgroup name does training example 803 come from?
    groupname_for_example_802 = groupnames[y[802]]
    print("3. The newsgroup name that training example 803 comes from is:", groupname_for_example_802)
    




@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")




    """CODE FOR Q3.4: Modify naive_bayes.py/NaiveBayesLaplace"""

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")

    print("Without Laplace Smoothing:", model.p_xy[:, 0])
    

    model2 = NaiveBayesLaplace(num_classes=4)
    model2.fit(X, y, beta=1)
    y_hat = model2.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes with Laplace training error: {err_train:.3f}")
    print("   ")   

    
    y_hat = model2.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes with Laplace validation error: {err_valid:.3f}")
    print("With Laplace Smoothing (β=1):", model2.p_xy[:, 0])
    print("   ")
    
    model3 = NaiveBayesLaplace(num_classes=4)
    model3.fit(X, y, beta=10000)
    print("With Laplace Smoothing (β=10000):", model3.p_xy[:, 0])





@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4. Also modify naive_bayes.py/NaiveBayesLaplace"""
   # raise NotImplementedError()



@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE FOR Q4. Also modify random_tree.py/RandomForest"""
    
    print("Random Forest")
    evaluate_model(RandomForest(num_trees=50, max_depth=np.inf))


   # raise NotImplementedError()



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]
    

    """YOUR CODE HERE FOR Q5.1. Also modify kmeans.py/Kmeans"""
    lowest_error = float('inf')
    best_model = None
    best_y = None
    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    error = model.error(X, y, model.means)
    if error < lowest_error:
            lowest_error = error
            best_model = model
            best_y = y
    
    print("Lowest Error obtained: ", lowest_error)

   

   
    lowest_error = float('inf')
    best_model = None
    best_y = None

    for _ in range(50):
        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        error = model.error(X, y, model.means)
        if error < lowest_error:
            lowest_error = error
            best_model = model
            best_y = y

    print("Lowest Error obtained: ", lowest_error)
    plt.scatter(X[:, 0], X[:, 1], c=best_y, cmap="jet")
    fname = Path("..", "figs", "kmeans_best_model.png")
    plt.savefig(fname)


 

    
    #raise NotImplementedError()



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.2"""
    errors = []
    
    for k in range(1, 11):
        lowest_error = float('inf')
        for _ in range(50):
            model = Kmeans(k)
            model.fit(X)
            y = model.predict(X)
            error = model.error(X, y, model.means)
            if error < lowest_error:
                lowest_error = error
        
        errors.append(lowest_error)

    plt.plot(range(1, 11), errors, marker='o')
    plt.title('Error as a function of k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Error')
    plt.xticks(range(1, 11))
    plt.grid(True)
    fname = Path("..", "figs", "5.2.png")
    plt.savefig(fname)




if __name__ == "__main__":
    main()
