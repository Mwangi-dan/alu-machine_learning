#!/usr/bin/env python3
"""
Module to create a layer
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev: tensor output of the previous layer
    n: number of nodes in the layer to create
    activation: activation function that the layer should use

    Returns: tensor output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation, kernel_initializer=initializer, name="layer")
    return layer(prev)
