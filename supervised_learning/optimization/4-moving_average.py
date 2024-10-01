#!/usr/bin/env python3
"""
Calculate sthe weighted moving average of a dataset
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a dataset

    data: list of data
    beta: weight used for moving average

    Should use bias correction

    Returns: list containing the moving averages of data
    """
    V = 0
    moving_averages = []
    for i in range(len(data)):
        V = beta * V + (1 - beta) * data[i]
        moving_averages.append(V / (1 - beta**(i + 1)))
    return moving_averages
