#!/usr/bin/env python3
"""
Poisson Distribution

Probability: Math for ML
"""

pi = 3.1415926536
e = 2.7182818285


class Poisson:
    """
    Poisson Class
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Class constructore
        """
        if data is None:
            if lambtha >= 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError("lambtha must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) > 2:
                    self.data = data
                    self.lambtha = float(sum(data) / len(data))
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")

    def factorial(self, n):
        """
        Calculates the factorial of a number

        n: number to calculate factorial of

        Returns:
        Factorial of n
        """
        if n == 0:
            return 1
        return n * self.factorial(n - 1)

    def pmf(self, k):
        """
        Calculates Poisson PMF (Probability Mass Function)

        k: number of successes

        Returns:
        PMF Value of k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        return ((self.lamtha ** k) *
                (e ** (-self.lambtha))) / self.factorial(k)

    def cdf(self, k):
        """
        Calculates the Cumulative Distribution Function (CDF)

        k: number of successes

        Returns:
        CDF Value of k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
