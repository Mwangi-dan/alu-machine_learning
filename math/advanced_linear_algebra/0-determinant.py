#!/usr/bin/env python3
'''
Module to calculate the determinant of a matrix
'''


def determinant(matrix):
    '''
    Matrix: List of lists

    Returns:
    det: int determinant
    '''
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    height = len(matrix)
    if height is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) is 0 and height is 1:
            return 1
        if len(row) != height:
            raise ValueError("matrix must be a square matrix")
    if height == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    if height == 2:
        return det_2x2(matrix)
    a = [[matrix[1][1], matrix[1][2]], [matrix[2][1], matrix[2][2]]]
    b = [[matrix[1][0], matrix[1][2]], [matrix[2][0], matrix[2][2]]]
    c = [[matrix[1][0], matrix[1][1]], [matrix[2][0], matrix[2][1]]]

    a1 = matrix[0][0] * det_2x2(a)
    a2 = matrix[0][1] * det_2x2(b)
    a3 = matrix[0][2] * det_2x2(c)

    return a1 - a2 + a3


def det_2x2(mat):
    '''
    Determinant for 2 x 2 matrix

    Returns:
    det: (int) determinant
    '''
    return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]
