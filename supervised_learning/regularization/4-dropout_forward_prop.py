#!/usr/bin/env python3
"""
Forward propagation using Dropout
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Droput

    X: numpy.ndarray of shape (nx, m) containing the
    input data for the network
    weights: dictionary of the weights and biases of the NN
    L: number of layers in the network
    keep_prob: probability that a node will be kept

    Returns: dictionary containing the output of each layer
    """
    cache = {}
    cache['A0'] = X
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A = cache['A' + str(i - 1)]
        Z = np.matmul(W, A) + b
        if i == L:
            t = np.exp(Z)
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A = A * D
            A = A / keep_prob
            cache['D' + str(i)] = D
            cache['A' + str(i)] = A
    return cache
