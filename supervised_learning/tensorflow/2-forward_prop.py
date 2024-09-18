#!/usr/bin/env python3
"""
Module with forward propoagation
"""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates forward propagation graph for NN
    x: placeholder of input data
    layer_sizes: list with number of nodes in each layer
    activations: list with activation functions for each layer of network

    Returns: prediction of network in tensor form
    """
    for i in range(len(layer_sizes)):
        if i == 0:
            prev = x
        prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev
