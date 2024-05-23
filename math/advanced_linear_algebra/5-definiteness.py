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
    rows, cols = matrix.shape
    if shape != rows:
        return None

    evalue, evec = np.linalg.eig(matrix)

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
