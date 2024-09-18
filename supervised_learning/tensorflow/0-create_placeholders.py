#!/usr/bin/env python3
"""
Module that holds the function create_placeholders
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """
    nx: Number of input features to the network
    classes: Number of classes in the classifier

    Returns: placeholders named x and y, respectively
    """
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")

    return x, y
