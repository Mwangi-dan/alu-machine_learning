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
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
