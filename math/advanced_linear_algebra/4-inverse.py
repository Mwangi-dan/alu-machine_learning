#!/usr/bin/env python3
'''
Module to get inverse of matrix
'''


def inverse(matrix):
    '''
    Returns:
    inv: (list) inverse of matrix
    '''
    if type(matrix) is not list:
        raise TypeError('matrix must be a list of lists')
    height = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        elif len(row) != height or len(row) == height and len(row) == 0:
            raise ValueError('matrix must be a non-empty square matrix')

    det = __import__('0-determinant').determinant
    adj = __import__('3-adjugate').adjugate

    det_matrix = det(matrix)
    adj_matrix = adj(matrix)

    if det_matrix == 0:
        return None

    inv = []
    for row in adj_matrix:
        new_row = []
        for j in row:
            new_row.append(j/det_matrix)
        inv.append(new_row)

    return inv
