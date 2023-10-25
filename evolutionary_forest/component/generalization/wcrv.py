import numpy as np
from minepy import MINE


# Function to calculate the MIC using minepy
def calculate_mic(x, y):
    mine = MINE()
    mine.compute_score(x, y)
    mic = mine.mic()
    return mic


# Function to calculate the Weighted MIC between Residuals and Variables (WCRV)
def calculate_WCRV(inputs, residuals, weights, median_weight):
    wcrv = 0
    for vk, weight_vk in enumerate(weights):
        if weight_vk >= median_weight:
            mic_vk_R = calculate_mic(inputs[:, vk], residuals)
            wcrv += weight_vk * mic_vk_R
        else:
            wcrv += 1 - weight_vk  # Apply a penalty for unimportant variables
    return wcrv


if __name__ == "__main__":
    # Example data
    inputs = np.random.rand(100, 5)  # Assume 100 samples and 5 features
    residuals = np.random.rand(100)  # Assume 100 residuals
    weights = [
        calculate_mic(inputs[:, i], residuals) for i in range(inputs.shape[1])
    ]  # Calculate the weight for each input variable
    median_weight = np.median(weights)  # Calculate the median weight

    # Calculate WCRV
    wcrv = calculate_WCRV(inputs, residuals, weights, median_weight)
    print("WCRV:", wcrv)
