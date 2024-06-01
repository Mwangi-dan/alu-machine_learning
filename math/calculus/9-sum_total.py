#!/usr/bin/env python3
"""
Summation function
"""


def summation_i_squared(n):
    """
    n: (int)
    Returns:
    sum: (int) summation
    """
    sum = 0
    for i in range(1, n):
        sum += i * i
    return sum
