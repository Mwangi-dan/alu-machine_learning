#!/usr/bin/env python3
"""
Updates the weights of a neural network with Dropout rglz
using gradient descent
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a NN with dropout r using gd

    Y: np.ndarray (classes, m) - correct
        m: number of data points
    weights: dict - weights and biases of the NN
    cache: dict - outputs of each layer of the NN
    alpha: float - learning rate
    keep_prob: float - probability that a node will be kept
    L: int - number of layers in the NN

    *NN uses tanh activations on each layer except the last
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        dW = (1 / m) * np.matmul(dZ, A.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        if i - 1 > 0:
            dA = 1 - A ** 2
            dZ = np.matmul(W.T, dZ) * dA
            D = cache['D' + str(i - 1)]
            dZ = dZ * D
            dZ = dZ / keep_prob
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db

    return weights
