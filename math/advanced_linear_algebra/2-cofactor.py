#!/usr/bin/env python3
'''
Cofactor matrix
'''


def cofactor(matrix):
    '''
    Returns:
    cofactor: (list) matrix
    '''
    minor = __import__('1-minor').minor

    if type(matrix) is not list:
        raise TypeError('matrix must be a list of lists')
    height = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        elif len(row) != height or len(row) == height and len(row) == 0:
            raise ValueError('matrix must be a non-empty square matrix')

    minor_mat = minor(matrix)
    co_mat = []

    for i in height:
        cof_row = []
        for j in height:
            sign = (-1) ** (i + j)
            cofactor = sign * minor_mat[i][j]
            cof_row.append(cofactor)
        co_mat.append(cof_row)

    return co_mat
