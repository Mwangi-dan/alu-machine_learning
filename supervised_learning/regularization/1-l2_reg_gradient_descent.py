#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization

This script updates the weights and biases of a neural
network using gradient descent with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    Y: one-hot numpy.ndarray of shape (classes, m) that contains
        the correct labels for the data
    weights: dictionary of the weights and biases of the neural
    network
    cache: dictionary of the outputs of each layer of the neural
    network
    alpha: learning rate
    lambtha: regularization parameter
    L: number of layers in the neural network

    The neural network uses tanh activations on
    each layer except the last, which uses
    a softmax activation
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        dW = (1 / m) * np.matmul(dZ, A.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.matmul(W.T, dZ) * (1 - A ** 2)
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
