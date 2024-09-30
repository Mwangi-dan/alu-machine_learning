#!/usr/bin/env python3
"""
Forward propagation using Dropout
"""

import tensorflow as tf


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
        Z = tf.matmul(W, A) + b
        if i == L:
            t = tf.nn.softmax(Z)
        else:
            A = tf.nn.tanh(Z)
            D = tf.nn.dropout(A, keep_prob)
            cache['D' + str(i)] = D
            cache['A' + str(i)] = A
    return cache
