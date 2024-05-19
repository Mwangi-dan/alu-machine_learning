#!/usr/bin/env python3
"""
Adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    mat1, mat2: list of list(S)

    Returns:
    add_m: added matrices || None
    """
    shape = __import__('2-size_me_please').matrix_shape
    add_arrays = __import__('4-line_up').add_arrays

    if shape(mat1) == shape(mat2):
        add_m = []
        dim = len(mat1)
        for i in range(dim):
            add_m.append(add_arrays(mat1[i], mat2[i]))
        return add_m
    else:
        return None
