import numpy as np
from scipy.special import gamma


def student_distribution(x, mean=0.0, std=1.0):
    v = 2 / (std - 1)
    return gamma((v + 1) / 2) / (np.sqrt(v * np.pi) * gamma(v / 2)) * (1 + (x - mean)**2 / v)**(- (v + 1) / 2)


def gaussian_distribution(x, mean=0.0, std=1.0):
    return 1 / (std * np.sqrt(2.0 * np.pi)) * np.exp(- 0.5 * ((x - mean) / std)**2)


def cauchy_distribution(x, x0=0.0, gamma=1.0):
    return 1 / (np.pi * gamma) * (gamma**2 / ((x - x0)**2 + gamma**2))


def laplace_distribution(x, mean=0.0, b=1.0):
    return 1 / (2 * b) * np.exp(- np.abs(x - mean) / b)


def logistic_distribution(x, mean=0.0, s=1.0):
    return np.exp(- (x - mean) / s) / (s * (1 + np.exp(- (x - mean) / s)))


dist_dict = {
    'normal': gaussian_distribution,
    'median': gaussian_distribution,
    'laplace': laplace_distribution,
    'student': student_distribution,
    'correlation': gaussian_distribution,
}
