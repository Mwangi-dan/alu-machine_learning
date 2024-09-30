#!/usr/bin/env python3
"""
Creates a layer of NN using dropout
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: activation function that should be used on the layer
    keep_prob: probability that a node will be kept

    Returns: output of the new layer
    """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=kernel
    )
    drop = tf.layers.Dropout(rate=keep_prob)
    return drop(layer(prev))
