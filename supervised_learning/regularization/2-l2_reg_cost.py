#!/usr/bin/env python3
"""
L2 Regularization Cost
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Cost: tensor containing the cost of the network
    without L2 regularization

    Returns: tensor containing the cost of the network
    """
    return cost + tf.losses.get_regularization_losses()