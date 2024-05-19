#!/usr/bin/env python3
'''
Bracing the Elements
'''


def np_elementwise(mat1, mat2):
    '''
    Performs element-wise addition, substraction,
    multiplication, division

    Returns
    (add, sub, mul, div): (tuple)
    '''
    return (np.sum(mat1, mat2),
            np.subtract(mat1, mat2),
            np.multiply(mat1, mat2),
            np.divide(mat1, mat2))
