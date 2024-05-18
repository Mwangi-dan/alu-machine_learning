#!/usr/bin/env python3
"""
Function that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    shape = []
    for i in matrix:
        shape.append(len(i))
    shape = list(set(shape))
    shape.append(len(matrix))
    return shape
