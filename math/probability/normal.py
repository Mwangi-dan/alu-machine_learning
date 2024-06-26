#!/usr/bin/env python3
"""
Normal Distribution
"""
pi = 3.1415926536
e = 2.7182818285


class Normal:
    """
    Normal Distribution class
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor
        """
        if data is None:
            if stddev > 0:
                self.stddev = float(stddev)
                self.mean = float(mean)
            else:
                raise ValueError("stddev must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) > 1:
                    self.data = data
                    self.mean = sum(data) / len(data)
                    self.stddev = (sum([(x - self.mean) ** 2 for x in data]) /
                                   len(data)) ** 0.5
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")

    def z_score(self, x):
        """
        Calculates the z-score of a given x value

        x: given value

        Returns:
        Z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x value of a given z-score

        z: z-score

        Returns:
        x value of z
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x value

        x: given value

        Returns:
        PDF value for x
        """
        z = self.z_score(x)
        den = self.stddev * ((2 * pi) ** 0.5)
        power = -1 * (z ** 2) / 2
        num = e ** power
        return num / den

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x value

        x: given value

        Returns:
        CDF value for x
        """
        mean = self.mean
        stddev = self.stddev
        value = (x - mean) / (stddev * (2 ** (1 / 2)))
        val = value - ((value ** 3) / 3) + ((value ** 5) / 10)
        val = val - ((value ** 7) / 42) + ((value ** 9) / 216)
        val *= (2 / (pi ** (1 / 2)))
        cdf = (1 / 2) * (1 + val)
        return cdf
