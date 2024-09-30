#!/usr/bin/env python3
"""
Module that determines if you should stop GD early
"""

import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early

    cost: current validation cost of the neural network
    opt_cost: lowest recorded validation cost of the neural network
    threshold: threshold used for early stopping
    patience: patience count used for early stopping
    count: how long the threshold has not been met

    Returns: a boolean of whether the network should be stopped early,
    followed by the updated count
    """
    if opt_cost - cost > threshold:
        count = 0
        opt_cost = cost
    else:
        count += 1
    if count == patience:
        return True, count
    return False, count
