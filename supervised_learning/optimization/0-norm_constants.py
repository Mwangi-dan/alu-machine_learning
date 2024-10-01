#!/usr/bin/env python3
"""
Module that calculates normalization constants of a matrix
"""

import numpy as np


def normalization_constants(X):
    """
    Function that calculates normalization constants of a matrix
    
    X: numpy.ndarray of shape (nx, m)
        nx: number of input features
        m: number of data points

    Returns: mean and standard deviation of each feature, respectively
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std_dev = np.std(X, axis=1, keepdims=True)

    return mean, std_dev
