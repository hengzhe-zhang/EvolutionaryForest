class MultipleValuesMovingAverage:
    def __init__(self, window_size, num_values):
        self.window_size = window_size
        self.num_values = num_values
        self.cumulative_sums = [0] * num_values
        self.counts = [0] * num_values
        self.values = [[] for _ in range(num_values)]

    def add_values(self, values):
        if len(values) != self.num_values:
            raise ValueError(f"Expected {self.num_values} values, got {len(values)}")

        for i in range(self.num_values):
            if self.counts[i] < self.window_size:
                self.cumulative_sums[i] += values[i]
                self.values[i].append(values[i])
                self.counts[i] += 1
            else:
                self.cumulative_sums[i] += values[i] - self.values[i].pop(0)
                self.values[i].append(values[i])

        return self.get_moving_averages()

    def get_moving_averages(self):
        moving_averages = []
        for i in range(self.num_values):
            if self.counts[i] == 0:
                moving_averages.append(0)
            else:
                moving_averages.append(
                    self.cumulative_sums[i] / min(self.counts[i], self.window_size)
                )
        return moving_averages


if __name__ == "__main__":
    multiple_ma = MultipleValuesMovingAverage(window_size=3, num_values=2)
    print(multiple_ma.add_values([10, 100]))
    print(multiple_ma.add_values([20, 200]))
    print(multiple_ma.add_values([30, 300]))
    print(multiple_ma.add_values([40, 400]))
