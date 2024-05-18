#!/usr/bin/env python 3
"""
Concatenates two matrices
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Returns
    n_matrix: concatenate matrix along a specific axis
    """
    cat = __import__('6-howdy_partner').cat_arrays

    n_matrix = []
    if axis == 0:
        n_matrix.append(cat(mat1, mat2))

        return n_matrix
    elif axis > 0:
        for i in range(len(mat1)):
            n_matrix.append(mat1[i] + mat2[i])
        return n_matrix
    else:
        return None
