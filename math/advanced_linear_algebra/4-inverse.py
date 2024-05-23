#!/usr/bin/env python3
'''
Module to get inverse of matrix
'''


def inverse(matrix):
    '''
    Returns:
    inv: (list) inverse of matrix
    '''
    adj = __import__('3-adjugate').adjugate
    det = __import__('0-determinant').determinant

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

    returnn inv
