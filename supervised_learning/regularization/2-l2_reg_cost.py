#!/usr/bin/env python3
"""
L2 Regularization Cost
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    cost: cost of the network without L2 regularization
    lambtha: regularization parameter
    weights: dictionary of the weights and biases
    L: number of layers in the neural network
    m: number of data points used

    Returns: the cost of the network accounting for L2 regularization
    """
    norm = 0
    for i in range(1, L + 1):
        norm += np.linalg.norm(weights['W' + str(i)])
    return cost + (lambtha / (2 * m)) * norm
