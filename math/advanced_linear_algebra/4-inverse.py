#!/usr/bin/env python3
'''
Module to get inverse of matrix
'''


def inverse(matrix):
    '''
    Returns:
    inv: (list) inverse of matrix
    '''
    det = __import__('0-determinant').determinant
    adj = __import__('3-adjugate').adjugate

    det_matrix = det(matrix)
    adj_matrix = adj(matrix)

    return 1/det_matrix * adj_matrix
