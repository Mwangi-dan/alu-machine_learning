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
                raise TypeError("nx must be an integer")
            elif nx < 1:
                raise ValueError("nx must be a positive integer")
        except TypeError:
            raise TypeError("nx must be an integer")
        except ValueError:
            raise ValueError("nx must be a positive integer")
        
        self.nx = nx
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
