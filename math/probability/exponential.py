#!/usr/bin/env python3
"""
Exponential Distribution
"""
pi = 3.1415926536
e = 2.7182818285


class Exponential:
    """
    Exponential Distribution class
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor
        """
        if data is None:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError("lambtha must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) > 2:
                    self.data = data
                    self.lambtha = float(len(data) / sum(data))
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")

    def pdf(self, x):
        """
        Calculates the Probability Density Function (PDF)

        x: time period

        Returns:
        PDF Value of x
        """
        if x < 0:
            return 0
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        x: given time period

        Returns:
        CDF Value of x
        """
        return 1 - (e ** (-self.lambtha * x))
