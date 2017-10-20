import random


class BernoulliArm:
    def __init__(self, p: float):
        self.p = p

    def draw(self) -> float:
        return 1.0 if random.random() < self.p else 0.0
