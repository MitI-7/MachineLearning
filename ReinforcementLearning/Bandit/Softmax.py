import math
import random
from typing import List


class Softmax:
    # temperatureがNoneの場合焼きなましする
    def __init__(self, temperature: float, counts: List, values: List):
        self.temperature = temperature
        self.counts = counts
        self.values = values

    def initialize(self, n_arms: int):
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

    def select_arm(self):
        temperature = self.temperature
        if temperature is None:
            t = sum(self.counts) + 1
            temperature = 1 / math.log(t + 0.0000001)

        z = sum([math.exp(v / temperature) for v in self.values])
        probs = [math.exp(v / temperature) / z for v in self.values]

        z = random.random()
        cum_prob = 0.0
        for i in range(len(probs)):
            cum_prob += probs[i]
            if cum_prob > z:
                return i

        return len(probs) - 1

    def update(self, chosen_arm: int, reward: float):
        self.counts[chosen_arm] += 1

        n = self.counts[chosen_arm]
        old_value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / float(n)) * old_value + (1 / float(n)) * reward

    def __str__(self):
        return "Softmax(temperature={0})".format(self.temperature)
