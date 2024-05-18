#!/usr/bin/env python3
"""
Function that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    shape = []
    shape.append(len(matrix))
    dim = []
    for i in matrix:
        dim.append(len(i))
    shape.append(set(dim))
    return shape
