import math
import numpy as np
from typing import List


class UCB1:
    def __init__(self, counts: List, values: List):
        self.counts = counts
        self.values = values

    def initialize(self, n_arms: int):
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

    def select_arm(self) -> int:
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0] * n_arms
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        return np.argmax(ucb_values)

    def update(self, chosen_arm: int, reward: float):
        assert 0.0 <= reward <= 1.0

        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        old_value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / float(n)) * old_value + (1 / float(n)) * reward

    def __str__(self):
        return "UCB1"
