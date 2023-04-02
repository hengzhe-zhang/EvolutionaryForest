import numpy as np


class MultiArmBandit:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.arm_values = {}
        self.arm_counts = {}
        self.total_counts = 0

    def choose_arm(self):
        if np.random.random() > self.epsilon:
            return max(self.arm_values, key=self.arm_values.get)
        else:
            return np.random.choice(list(self.arm_values.keys()))

    def normalized_probability(self):
        total = sum(self.arm_values.values())
        if total == 0:
            return {arm: 1 / len(self.arm_values) for arm in self.arm_values}
        else:
            return {arm: value / total for arm, value in self.arm_values.items()}

    def normalized_probability_list(self):
        total = sum(self.arm_values.values())
        if total == 0:
            return [1 / len(self.arm_values)] * len(self.arm_values)
        else:
            return [value / total for arm, value in sorted(self.arm_values.items())]

    def update(self, arm, reward):
        if arm not in self.arm_values:
            self.arm_values[arm] = 0
            self.arm_counts[arm] = 0
        self.total_counts += 1
        self.arm_counts[arm] += 1
        step_size = 1.0 / self.arm_counts[arm]
        self.arm_values[arm] += step_size * (reward - self.arm_values[arm])

    @staticmethod
    def create_multiple(n_bandits, epsilon=0.1):
        return [MultiArmBandit(epsilon=epsilon) for _ in range(n_bandits)]

    @staticmethod
    def update_multiple(bandits, arms_rewards):
        for i, rewards in enumerate(arms_rewards):
            for arm, reward in rewards.items() if isinstance(rewards, dict) else enumerate(rewards):
                bandits[i].update(arm, reward)

    @staticmethod
    def normalized_probability_multiple(bandits):
        return [bandit.normalized_probability() for bandit in bandits]

    @staticmethod
    def normalized_probability_multiple_list(bandits):
        return [bandit.normalized_probability_list() for bandit in bandits]
