#!/usr/bin/env python3
"""
Normalizes (Standardizes) a matrix
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (Standardizes) a matrix

    X: numpy.ndarray of shape (d, nx)
        d: number of data points
        nx: number of features
    m: numpy.ndarray of shape (nx,)
    contains mean of all features of X
    s: numpy.ndarray of shape (nx,)
    Standard deviation of all features of X

    Returns: normalized X matrix
    """
    return (X - m) / s
