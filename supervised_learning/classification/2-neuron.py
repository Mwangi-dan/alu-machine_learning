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
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    # getter function - W
    @property
    def W(self):
        """
        Getter function for W
        """
        return self.__W

    # getter function - b
    @property
    def b(self):
        """
        Getter function for b
        """
        return self.__b

    # getter function - A
    @property
    def A(self):
        """
        Getter function for A
        """
        return self.__A


    def forward_prop(self, x):
        """
        Method that calculates the forward propagation of the neuron
        """
        self.__A = 1 / (1 + np.exp(-(np.matmul(self.__W, x) + self.__b)))
        return self.__A
