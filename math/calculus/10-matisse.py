#!/usr/bin/env python3
"""
Calculating derivative of
a polynomial
"""


def poly_derivative(poly):
    """
    calculates the derivative of a polynomial
    """
    if not isinstance(poly, list) or len(poly) == 1:
        return None
    if len(poly) == 1:
        return [0]

    derivative = [poly[i] * i for i in range(1, len(poly))]
    if len(derivative) == 0:
        return None

    if all(x == 0 for x in derivative):
        return [0]

    return derivative
