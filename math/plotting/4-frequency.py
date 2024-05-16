#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=10, edgecolor="black", range=(0, 100))
plt.xlabel("Grades")
plt.xlim(left=0, right=100)
plt.ylim(0, 30)
plt.ylabel("Number of students")
plt.title("Project A")

plt.show()
