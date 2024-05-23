#!/usr/bin/env python3
'''
Module to determine definiteness of matrix
'''
import numpy as np


def definiteness(matrix):
    '''
    Returns:
    definite: (str) definiteness of matrix
    '''
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] or \
       np.array_equal(matrix, matrix.T) is False:
        return None

    eigenvalues, evec = np.linalg.eig(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
