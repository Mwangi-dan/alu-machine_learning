#!/usr/bin/env python3
'''
Module to get minor matrix
'''


def minor(matrix):
    '''
    Returns:
    minor: (list) minor of matrix
    '''
    det = __import__('0-determinant').determinant

    def g_minor(matrix, row, col):
        return [r[:col] + r[col+1:] for r in (matrix[:row] + matrix[row+1:])]

    if type(matrix) is not list:
        raise TypeError('matrix must be a list of lists')
    height = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        elif len(row) != height or len(row) == height and len(row) == 0:
            raise ValueError('matrix must be a non-empty square matrix')
        elif len(row) == 1:
            return [[1]]

    minor_mat = []

    for i in range(height):
        minor_row = []
        for j in range(height):
            minor_matrix = g_minor(matrix, i, j)
            minor_row.append(det(minor_matrix))
        minor_mat.append(minor_row)

    return minor_mat
