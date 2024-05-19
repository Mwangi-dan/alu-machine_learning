#!/usr/bin/env python3
'''
Concatenate two matrices along a specific axis
'''

import numpy as np


def np_cat(mat1, mat2, axis=0):
    ''''
    Returns
    n_matrix: (list) concateneated matrix
    '''
    return np.concatenate((mat1, mat2), axis=axis)
