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
    if n < 1 or type(n) != int:
        return None
    return int((n * (n + 1) * ((2 * n) + 1)/6
