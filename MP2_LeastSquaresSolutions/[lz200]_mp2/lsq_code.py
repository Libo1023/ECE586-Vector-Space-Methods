import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from scipy.sparse import coo_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

## Helper functions you don't need to modify

# Function to remove outliers before plotting histogram
def remove_outlier(x, thresh=3.5):
    """
    returns points that are not outliers to make histogram prettier
    reference: https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting/11886564
    Arguments:
        x {numpy.ndarray} -- 1d-array, points to be filtered
        thresh {float} -- the modified z-score to use as a threshold. Observations with
                          a modified z-score (based on the median absolute deviation) greater
                          than this value will be classified as outliers.
    Returns:
        x_filtered {numpy.ndarray} -- 1d-array, filtered points after dropping outlier
    """
    if len(x.shape) == 1: x = x[:,None]
    median = np.median(x, axis=0)
    diff = np.sqrt(((x - median)**2).sum(axis=-1))
    modified_z_score = 0.6745 * diff / np.median(diff)
    x_filtered = x[modified_z_score <= thresh]
    return x_filtered

## End of helper functions

## Coding Exercise Starts Here

# General function to compute Vandermonde matrix for Exercise 2.2
def create_vandermonde(x, m):
    """
    Arguments:
        x {numpy.ndarray} -- 1d-array of (x_1, x_2, ..., x_n)
        m {int} -- a non-negative integer, degree of polynomial fit
    Returns:
        A {numpy.ndarray} -- an n x (m+1) matrix where A_{ij} = x_i^{j-1}
    """
    # Add code to compute Vandermonde A
    #################################################################
    n = len(x)
    A = np.zeros((n, m+1))
    for i in range(0, n) :
        for j in range(0, m+1) :
            if j == 0 :
                A[i][j] = 1
            else :
                A[i][j] = x[i] ** j
    #################################################################
    return A

# General function to solve linear least-squares via normal equations for Exercise 2.2
def solve_linear_LS(A, y):
    """
    Arguments:
        A {numpy.ndarray} -- an m x n matrix
        y {numpy.ndarray} -- a length-m vector
    Returns:
        z_hat {numpy.ndarray} -- length-n vector, the optimal solution for the given linear least-square problem
    """
    # Add code to compute least squares solution z_hat via linear algebra
    #################################################################
    # The previous default of (-1) will use 
    # the machine precision times max(M, N).
    z_hat_tuple = np.linalg.lstsq(A, y, rcond = -1)
    z_hat = z_hat_tuple[0]
    #################################################################
    return z_hat

# General function to solve linear least-squares via via partial gradient descent for Exercise 2.2
def solve_linear_LS_gd(A, y, step, niter):
    """
    Arguments:
        A {numpy.ndarray} -- an m x n matrix
        y {numpy.ndarray} -- a length-m vector
        step -- a floating point number, step size
        niter -- a non-negative integer, number of updates
    Returns:
        z_hat {numpy.ndarray} -- length-n vector, the optimal solution for the given linear least-square problem
    """
    # Add code to approximate least squares solution z_hat via gradient descent
    #####################################################################################
    m = A.shape[0]
    n = A.shape[1]
    z_hat = np.zeros(n)
    for t in range(0, niter) :
        z_hat = z_hat + step * (y[t%m] - np.dot(A[t%m, :], z_hat)) * A[t%m, :]
    #####################################################################################
    return z_hat

# General function to extract samples with given labels and randomly split into test and training sets for Exercise 2.3
def extract_and_split(df, d, test_size=0.5):
    """
    extract the samples with given labels and randomly separate the samples into training and testing groups, 
    extend each vector to length 785 by appending a −1
    Arguments:
        df {dataframe} -- the dataframe of MNIST dataset
        d {int} -- digit needs to be extracted, can be 0, 1, ..., 9
        test_size {float} -- the fraction of testing set, default value is 0.5
    Returns:
        X_tr {numpy.ndarray} -- training set features, a matrix with 785 columns
                                each row corresponds the feature of a sample
        y_tr {numpy.ndarray} -- training set labels, 1d-array
                                each element corresponds the label of a sample
        X_te {numpy.ndarray} -- testing set features, a matrix with 785 columns 
                                each row corresponds the feature of a sample
        y_te {numpy.ndarray} -- testing set labels, 1d-array
                                each element corresponds the label of a sample
    """
    # Add code here extract data and randomize order
    ##########################################################################################
    d_count = 0
    for i in range(0, len(df)) :
        if df['label'][i] == d :
            d_count = d_count + 1
        else :
            continue
    d_features = np.zeros((d_count, 785))
    d_labels = np.zeros(d_count)

    d_index = 0
    for i in range(0, len(df)) :
        if df['label'][i] == d :
            d_features[d_index][0:784] = df['feature'][i]
            d_features[d_index][784] = -1
            d_labels[d_index] = d 
            d_index = d_index + 1
        else :
            continue

    X_tr, X_te, y_tr, y_te = train_test_split(d_features, d_labels, test_size = test_size)
    ##########################################################################################
    return X_tr, X_te, y_tr, y_te  

# General function to train and test pairwise classifier for MNIST digits for Exercise 3.2
def mnist_pairwise_LS(df, a, b, test_size=0.5, verbose=False, gd=False):
    """
    Pairwise experiment for applying least-square to classify digit a and digit b
    Arguments:
        df {dataframe} -- the dataframe of MNIST dataset
        a, b {int} -- digits to be classified
        test_size {float} -- the fraction of testing set, default value is 0.5
        verbose {bool} -- whether to print and plot results
        gd {bool} -- whether to use gradient descent to solve LS        
    Returns:
        res {numpy.ndarray} -- numpy.array([training error, testing error])
    """
    # Find all samples labeled with digit a and split into train/test sets
    ###################################################################################################################
    Xa_tr, Xa_te, ya_tr, ya_te = extract_and_split(df, a, test_size)

    # Find all samples labeled with digit b and split into train/test sets
    Xb_tr, Xb_te, yb_tr, yb_te = extract_and_split(df, b, test_size)

    # Construct the full training set
    X_tr = np.concatenate((Xa_tr, Xb_tr))
    y_tr = np.concatenate((ya_tr, yb_tr))

    for i in range(0, len(y_tr)) :
        if y_tr[i] == a :
            y_tr[i] = -1
        if y_tr[i] == b :
            y_tr[i] = 1
    
    # Construct the full testing set
    X_te = np.concatenate((Xa_te, Xb_te))
    y_te = np.concatenate((ya_te, yb_te))

    for i in range(0, len(y_te)) :
        if y_te[i] == a :
            y_te[i] = -1
        if y_te[i] == b :
            y_te[i] = 1
    
    # Run least-square on training set
    z_hat = solve_linear_LS(X_tr.astype(float), y_tr.astype(float))

    # Optional: Try also solving this problem with cyclic partial gradient descent
    # and compare the two methods.
    #z_hat = solve_linear_LS_gd(X_tr.astype(float), y_tr.astype(float), step = 0.0002, niter = 10000)
        
    # Compute estimate and classification error for training set
    y_hat_tr = np.dot(X_tr, z_hat)

    for i in range(0, len(y_hat_tr)) :
        if y_hat_tr[i] >= 0 :
            y_hat_tr[i] = 1 
        else :
            y_hat_tr[i] = -1

    err_tr = 1 - accuracy_score(y_true = y_tr, y_pred = y_hat_tr)
    
    # Compute estimate and classification error for training set
    y_hat_te = np.dot(X_te, z_hat)

    for i in range(0, len(y_hat_te)) :
        if y_hat_te[i] >= 0 :
            y_hat_te[i] = 1
        else :
            y_hat_te[i] = -1

    err_te = 1 - accuracy_score(y_true = y_te, y_pred = y_hat_te)
    
    if verbose:
        print('Pairwise experiment, mapping {0} to -1, mapping {1} to 1'.format(a, b))
        print('training error = {0:.2f}%, testing error = {1:.2f}%'.format(100 * err_tr, 100 * err_te))
        
        # Compute confusion matrix
        cm_train = np.zeros((2, 2), dtype=np.int64)
        cm_test = np.zeros((2, 2), dtype=np.int64)

        # cm[0, 0] = ((y_te == -1) & (y_hat_te == -1)).sum()
        # cm[0, 1] = ((y_te == -1) & (y_hat_te == 1)).sum()
        # cm[1, 0] = ((y_te == 1) & (y_hat_te == -1)).sum()
        # cm[1, 1] = ((y_te == 1) & (y_hat_te == 1)).sum()

        cm_train = confusion_matrix(y_true = y_tr, y_pred = y_hat_tr)
        cm_test = confusion_matrix(y_true = y_te, y_pred = y_hat_te)

        print('Confusion matrix for Training Set is:\n {0}'.format(cm_train))
        print('Confusion matrix for Test Set is:\n {0}'.format(cm_test))

        # Compute the histogram of the function output separately for each class 
        # Then plot the two histograms together
        ya_te_hat, yb_te_hat = Xa_te @ z_hat, Xb_te @ z_hat

        # ya_te_hat = np.dot(Xa_te, z_hat)
        # yb_te_hat = np.dot(Xb_te, z_hat)

        output = [remove_outlier(ya_te_hat).flatten(), remove_outlier(yb_te_hat).flatten()]

        plt.figure(figsize=(8, 4))
        plt.hist(output, bins=50)
    
    res = np.array([err_tr, err_te])
    ###################################################################################################################
    return res

