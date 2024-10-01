#!/usr/bin/env python3
"""
Shuffles data points in two matrices the same way
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices

    X: first numpy.ndarray (m, nx) to shuffle
        m: number of data points
        nx: number of features in X

    Y: second numpy.ndarray (m, ny) to shuffle
        m: same number of data points as in X
        ny: number of features in Y

    Returns: shuffles X and Y matrices
    """
    shuffle = np.random.permutation(X.shape[0])
    return X[shuffle], Y[shuffle]
