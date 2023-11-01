class OnlineGaussianWelford:
    def __init__(self, decay_factor=1.0):
        self.n = 0  # Number of data points seen
        self.mean = 0  # Running mean
        self.M2 = 0  # Running variance times n
        self.decay_factor = decay_factor

    def update(self, x):
        self.n = self.decay_factor * self.n + 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 = self.decay_factor * self.M2 + delta * delta2

    def get_parameters(self):
        if self.n < 2:
            return self.mean, 0
        else:
            variance_n = self.M2 / self.n
            return self.mean, variance_n


# Usage:
if __name__ == "__main__":
    import numpy as np

    data_stream = [1, 2, 3, 4, 5]  # Example data stream
    model = OnlineGaussianWelford(decay_factor=1)
    print("Expected", np.mean(data_stream), np.var(data_stream))

    for x in data_stream:
        model.update(x)

    mu, sigma2 = model.get_parameters()
    print("Result", f"Mean: {mu}, Variance: {sigma2}")
