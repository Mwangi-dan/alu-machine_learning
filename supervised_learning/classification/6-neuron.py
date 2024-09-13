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

    def cost(self, Y, A):
        """
        Method that calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """
        Method that evaluates the neuron's predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Method that calculates one pass of gradient descent on the neuron
        """
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - (alpha * dw).T
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Method that trains the neuron
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        elif iterations < 1:
            raise ValueError("iterations must be a positive integer")
        elif type(alpha) is not float:
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
