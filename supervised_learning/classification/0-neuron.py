#!/usr/bin/env python3
"""
Module that holds the class of 'Neuron'
"""

import numpy as np

class Neuron:
    """
    class that defines a single Neuron performinh binary classification
    """
    def __init__(self, nx):
        """
        Constructor
        """
        try:
            if type(nx) is not int:
                raise TypeError
            if nx < 1:
                raise ValueError
        except TypeError:
            print("nx must be an integer")
        except ValueError:
            print("nx must be a positive integer")

        self.nx = nx
        self.__W = [0] * nx
        self.b = 0
        self.A = 0