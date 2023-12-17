import numpy as np


def loss_function_1(f_theta_x, y):
    """Loss function: (f_theta(x) - min(max(y, -1), 1))**2"""
    return (f_theta_x - np.minimum(np.maximum(y, -1), 1)) ** 2


def loss_function_2(f_theta_x, y):
    """Loss function: |y * (sqrt(|y / sqrt(1 + f_theta_x**2)|))|"""
    return abs(y * np.sqrt(abs(y / np.sqrt(1 + f_theta_x**2))))


def loss_function_3(f_theta_x, y):
    """Loss function: |(y / (1 + log(|y - 1|)**2)) - f_theta_x|"""
    return abs((y / (1 + np.log(abs(y - 1)) ** 2)) - f_theta_x)


def loss_function_4(f_theta_x, y):
    """Loss function: sqrt(|y * (y - f_theta_x)|)"""
    return np.sqrt(abs(y * (y - f_theta_x)))


def loss_function_5(f_theta_x, y):
    """Loss function: sqrt(|y - f_theta_x|)"""
    return np.sqrt(abs(y - f_theta_x))
