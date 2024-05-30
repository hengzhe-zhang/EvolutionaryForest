from typing import List, Union


class MovingAverage:
    def __init__(self, window_size, num_values=1):
        self.window_size = window_size
        self.num_values = num_values
        self.cumulative_sums = [0] * num_values
        self.counts = [0] * num_values
        self.values = [[] for _ in range(num_values)]

    def add_values(self, values: Union[int, List[int]]):
        if isinstance(values, (int, float)):
            values = [values]

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
        return moving_averages if self.num_values > 1 else moving_averages[0]


if __name__ == "__main__":
    # Scalar value moving average
    scalar_ma = MovingAverage(window_size=3)
    print(scalar_ma.add_values(10))  # Output: 10.0
    print(scalar_ma.add_values(20))  # Output: 15.0
    print(scalar_ma.add_values(30))  # Output: 20.0
    print(scalar_ma.add_values(40))  # Output: 30.0

    # Multiple values moving average
    multiple_ma = MovingAverage(window_size=3, num_values=2)
    print(multiple_ma.add_values([10, 100]))  # Output: [10.0, 100.0]
    print(multiple_ma.add_values([20, 200]))  # Output: [15.0, 150.0]
    print(multiple_ma.add_values([30, 300]))  # Output: [20.0, 200.0]
    print(multiple_ma.add_values([40, 400]))  # Output: [30.0, 300.0]
