#!/usr/bin/env python3
'''
Module with fucntion to get adjugate
'''


def adjugate(matrix):
    '''
    Returns:
    adj: (list) transposed cofactor matrix
    '''
    cofactor = __import__('2-cofactor').cofactor

    cof = cofactor(matrix)

    rows = len(cof)
    cols = len(cof[0])

    adj = []

    for col in range(cols):
        new_row = []
        for row in range(rows):
            new_row.append(cof[row][col])
        adj.append(new_row)

    return adj
