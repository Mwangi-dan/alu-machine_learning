#!/usr/bin/env python3
"""
Function that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    n_matrix = matrix
    shape = []
    while isinstance(n_matrix, list):
        shape.append(len(n_matrix))
        if len(n_matrix) > 0:
            n_matrix = n_matrix[0]
        else:
            break

    return shape
