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
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
