#!/usr/bin/env python3
"""
Adding two arrays
"""


def add_arrays(arr1, arr2):
    """
    arr1, arr2: list of list(s) of the same shape

    Returns
    add_array: added matrix || None: if arrays are different shape
    """
    shape = __import__('2-size_me_please').matrix_shape
    if shape(arr1) == shape(arr2):
        rows = len(arr1)
        add_array = []

        for i in range(rows):
            add_array.append((arr1[i] + arr2[i]))

        return add_array
    else:
        return None
