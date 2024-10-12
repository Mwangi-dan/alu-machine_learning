#!/usr/bin/env python3
"""
Module that defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork():
    """
    Class that defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """
        Class constructor

        nx: number of input features
        layers: list representing the number of nodes in each layer of network

        L: number of layers in the network
        cache: dictionary to hold all intermediary values of the network
        weights: dictionary to hold all weights and biases of the network
        """
        try:
            if nx is not type(int):
                raise TypeError("nx must be an integer")
            elif nx < 1:
                raise ValueError("nx must be a positive integer")
        except TypeError:
            raise TypeError("nx must be an integer")
        except ValueError:
            raise ValueError("nx must be a positive integer")

        try:
            if layers is not type(list) or len(layers) > 0:
                raise TypeError("layers must be a list of positive integers")
            if not isinstance(layers, 
                              list) or not all(
                                  isinstance(i,
                                  int) and i > 0 for i in layers):
                raise TypeError("layers must be a list of positive integers")
        except TypeError:
            raise TypeError("layers must be a list of positive integers")
        except ValueError:
            raise ValueError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers

        self.L = len(layers)
        self.cache = {}

        self.weights = {}
        for i in range(self.L):
            if i == 0:
                self.weights['W1'] = np.random.randn(
                    self.layers[i], self.nx) * np.sqrt(2 / self.nx)
                self.weights['b1'] = np.zeros((self.layers[i], 1))
            else:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    self.layers[i], self.layers[i - 1]) * np.sqrt(
                        2 / self.layers[i - 1])
                self.weights['b' + str(i + 1)] = np.zeros((self.layers[i], 1))
