#!/usr/bin/env python3
"""
Module that calculates the softmax cross-entropy loss of a prediction
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates softmax cross-entropy loss of a prediction
    y: placeholder for the labels of input data
    y_pred: tensor containing the network's predictions

    Returns: tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
