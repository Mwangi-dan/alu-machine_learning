#!/usr/bin/env python3
"""
Defining a Neural Network
"""

import numpy as np


class NeuralNetwork():
    """
    Neural Network with one hidden layer performing binary classification
    """
    def __init__(self, nx, nodes):
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
        try:
            if type(nodes) is not int:
                raise TypeError("nodes must be an integer")
            elif nodes < 1:
                raise ValueError("nodes must be a positive integer")
        except TypeError:
            raise TypeError("nodes must be an integer")
        except ValueError:
            raise ValueError("nodes must be a positive integer")
        self.nodes = nodes
        # privatize instance attributes
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    # Getter function - W1
    @property
    def W1(self):
        """
        Getter function for W1
        """
        return self.__W1

    # Getter function - b1
    @property
    def b1(self):
        """
        Getter function for b1
        """
        return self.__b1

    # Getter function - A1
    @property
    def A1(self):
        """
        Getter function for A1
        """
        return self.__A1

    # Getter function - W2
    @property
    def W2(self):
        """
        Getter function for W2
        """
        return self.__W2

    # Getter function - b2
    @property
    def b2(self):
        """
        Getter function for b2
        """
        return self.__b2

    # Getter function - A2
    @property
    def A2(self):
        """
        Getter function for A2
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Method that calculates the forward propagation of the neural network

        X: numpy.ndarray with shape (nx, m) that contains the input data
        nx: number of input features to the neuron
        m: number of examples

        Returns: the private attributes __A1 and __A2, respectively
        """
        self.__A1 = 1 / (1 + np.exp(-(np.matmul(self.__W1, X) + self.__b1)))
        self.__A2 = 1 / (
            1 + np.exp(-(np.matmul(self.__W2, self.__A1) + self.__b2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Method to calculate the cost of the Neural Network

        Y: numpy.ndarray with shape (1, m) that contains the correct labels
        A: numpy.ndarray with shape (1, m) containing the activated output

        Returns: the cost
        """
        m = Y.shape[1]
        return -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
