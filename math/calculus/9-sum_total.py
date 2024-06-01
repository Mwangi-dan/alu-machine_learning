#!/usr/bin/env python3
"""
Summation function
"""
import math


def summation_i_squared(n):
    """
    n: (int)
    Returns:
    sum: (int) summation
    """
    if not n or type(n) != int:
        return None
    return math.factorial(n)
