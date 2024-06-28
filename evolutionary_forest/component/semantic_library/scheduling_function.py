import random

import matplotlib.pyplot as plt
import numpy as np


def scheduling_controller(mode, current_gen, total_gen):
    if mode == "Linear" and random.random() < linear_decreasing_function(
        x=current_gen, end_x=total_gen
    ):
        return True

    if mode == "Exponential" and random.random() < exponential_decay(
        x=current_gen, decay_rate=0.1, start_value=1
    ):
        return True

    if mode == "Step" and random.random() < step_decay(
        x=current_gen, drop_rate=0.5, epochs_drop=10, start_value=1
    ):
        return True

    if mode == "Polynomial" and random.random() < polynomial_decay(
        x=current_gen, max_x=total_gen, start_value=1, end_value=0.1, power=2
    ):
        return True

    if mode == "Cosine" and random.random() < cosine_annealing(
        x=current_gen, max_x=total_gen, start_value=1, end_value=0.1
    ):
        return True

    if mode == "CosineRestart" and random.random() < cosine_annealing_with_restart(
        x=current_gen, initial_lr=1, min_lr=0.1, T_max=20, T_mult=2
    ):
        return True

    return False


# Linear Decreasing Function
def linear_decreasing_function(x, end_x=100, start_value=1, end_value=0.1):
    start_x = 0
    slope = (end_value - start_value) / (end_x - start_x)
    return start_value + slope * (x - start_x)


# Exponential Decay Function
def exponential_decay(x, decay_rate=0.1, start_value=1):
    return start_value * np.exp(-decay_rate * x)


# Step Decay Function
def step_decay(x, drop_rate=0.5, epochs_drop=10, start_value=1):
    return start_value * (drop_rate ** (x // epochs_drop))


# Polynomial Decay Function
def polynomial_decay(x, max_x=100, start_value=1, end_value=0.1, power=2):
    return (start_value - end_value) * ((1 - (x / max_x)) ** power) + end_value


# Cosine Annealing Function
def cosine_annealing(x, max_x=100, start_value=1, end_value=0.1):
    cos_inner = (np.pi * x) / max_x
    cos_out = np.cos(cos_inner) + 1
    return end_value + 0.5 * (start_value - end_value) * cos_out


def cosine_annealing_with_restart(x, initial_lr=1, min_lr=0.1, T_max=50, T_mult=2):
    cycle = 0
    t_curr = 0
    T_curr = T_max

    while t_curr + T_curr <= x:
        t_curr += T_curr
        T_curr *= T_mult
        cycle += 1

    t_since_restart = x - t_curr
    lr = min_lr + 0.5 * (initial_lr - min_lr) * (
        1 + np.cos(np.pi * t_since_restart / T_curr)
    )
    return lr


def plot_scheduling_functions():
    x = np.linspace(0, 100, 1000)  # Generate 1000 points between 0 and 100

    # Generate values for each function
    y_linear = linear_decreasing_function(x)
    y_exponential = exponential_decay(x)
    y_step = step_decay(x)
    y_polynomial = polynomial_decay(x)
    y_cosine = cosine_annealing(x)
    y_cosine_restart = [
        cosine_annealing_with_restart(
            epoch, initial_lr=1, min_lr=0.1, T_max=20, T_mult=2
        )
        for epoch in x
    ]

    # Plot each function
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 3, 1)
    plt.plot(x, y_linear, label="Linear Decreasing")
    plt.title("Linear Decreasing")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(2, 3, 2)
    plt.plot(x, y_exponential, label="Exponential Decay", color="orange")
    plt.title("Exponential Decay")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(2, 3, 3)
    plt.plot(x, y_step, label="Step Decay", color="green")
    plt.title("Step Decay")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(2, 3, 4)
    plt.plot(x, y_polynomial, label="Polynomial Decay", color="red")
    plt.title("Polynomial Decay")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(2, 3, 5)
    plt.plot(x, y_cosine, label="Cosine Annealing", color="purple")
    plt.title("Cosine Annealing")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(2, 3, 6)
    plt.plot(x, y_cosine_restart, label="Cosine Annealing with Restart", color="blue")
    plt.title("Cosine Annealing with Restart")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_scheduling_functions()
