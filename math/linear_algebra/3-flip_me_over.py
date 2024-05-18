#!/usr/bin/env python3
"""
Transposing matrices
"""


def matrix_transpose(matrix):
    """"
    matrix: list of list(s)

    Returns:
    t_matrix: transposed matrix
    """
    rows = len(matrix)
    col = len(matrix[0])
    t_matrix = [[0 for _ in range(rows)] for _ in range(col)]
    for i in range(rows):
        for j in range(col):
            t_matrix[j][i] = matrix[i][j]

    return t_matrix
