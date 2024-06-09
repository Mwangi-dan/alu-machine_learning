#!/usr/bin/env python3
"""
Poisson Distribution

Probability: Math for ML
"""

pi = 3.1415926536
e = 2.7182818285


class Binomial:
    """
    Binomial Class
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor
        """
        if data is None:
            if n > 0:
                self.n = int(n)
            else:
                raise ValueError("n must be a positive value")
            if 0 < p < 1:
                self.p = float(p)
            else:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)
            p = 1 - (variance / mean)
            self.n = int(round(mean / p))
            self.p = float(mean / self.n)

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
        Calculates Binomial PMF (Probability Mass Function)

        k: number of successes

        Returns:
        PMF Value of k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        fact = self.factorial(self.n) / \
            (self.factorial(k) * self.factorial(self.n - k))
        return (fact) * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given k value

        k: given value

        Returns:
        CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        return sum([self.pmf(i) for i in range(k + 1)])
