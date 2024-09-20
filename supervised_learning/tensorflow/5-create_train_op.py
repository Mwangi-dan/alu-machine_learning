#!/usr/bin/env python3
"""
Module that creates the training operation for the network
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network
    loss: tensor containing the loss of the network's prediction
    alpha: learning rate

    Returns: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op