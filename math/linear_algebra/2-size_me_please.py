#!/usr/bin/env python3
"""
Function that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    r = len(matrix)
    c = len(matrix[0])
    return [r, c]
