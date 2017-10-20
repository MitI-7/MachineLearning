import random
import numpy as np
from typing import List
from matplotlib import pyplot as plt
from Arms import BernoulliArm
from EpsilonGreedy import EpsilonGreedy
from Softmax import Softmax
from UCB1 import UCB1


def calc_time():
    def _calc_time(func):
        import time
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kargs):
            start = time.time()
            ret = func(*args, **kargs)
            print(f"time : {time.time() - start:0.4} sec")
            return ret
        return wrapper
    return _calc_time


@calc_time()
def test_algorithm(algorithm, arms: List, num_sims: int, horizon: int):
    n = num_sims * horizon
    chosen_arms = [0.0] * n
    rewards = [0.0] * n
    cumulative_rewards = [0.0] * n
    sim_nums = [0.0] * n
    times = [0.0] * n

    for sim in range(1, num_sims + 1):
        algorithm.initialize(len(arms))

        for t in range(1, horizon + 1):
            index = (sim - 1) * horizon + t - 1
            sim_nums[index] = sim
            times[index] = t

            chosen_arm = algorithm.select_arm()
            chosen_arms[index] = chosen_arm

            reward = arms[chosen_arm].draw()
            rewards[index] = reward

            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

            algorithm.update(chosen_arm, reward)

    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


def main():
    random.seed(1)

    means = [0.1, 0.25, 0.24, 0.01, 0.05]
    n_arms = len(means)
    random.shuffle(means)
    arms = list(map(lambda m: BernoulliArm(m), means))
    best_arm = np.argmax(means)
    print("means", means)
    print("Best arm", best_arm)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    num_sims = 5000
    horizon = 2500
    algorithm_list = [EpsilonGreedy(None, [], []),
                      Softmax(None, [], []),
                      UCB1([], [])]
    print(f"num_size: {num_sims} horizon: {horizon}")
    for algorithm in algorithm_list:
        print(algorithm)
        algorithm.initialize(n_arms)

        results = test_algorithm(algorithm=algorithm, arms=arms, num_sims=num_sims, horizon=horizon)
        sim_nums, times, chosen_arms, rewards, cumulative_rewards = results

        # 時間ごとの最適腕を引く確率
        y1 = [0] * horizon
        for t in range(horizon):
            y1[t] = sum((chosen_arms[sim * horizon + t] == best_arm) for sim in range(num_sims))
        y1 = list(map(lambda x: x / num_sims, y1))

        # 時間ごとの平均報酬
        y2 = [0] * horizon
        for t in range(horizon):
            y2[t] = sum(rewards[sim * horizon + t] for sim in range(num_sims))
        y2 = list(map(lambda x: x / num_sims, y2))

        # 時間ごとの累積報酬
        y3 = [0] * horizon
        for t in range(horizon):
            y3[t] = sum(cumulative_rewards[sim * horizon + t] for sim in range(num_sims))
        y3 = list(map(lambda x: x / num_sims, y3))

        ax1.plot(range(horizon), y1, label=str(algorithm))
        ax2.plot(range(horizon), y2, label=str(algorithm))
        ax3.plot(range(horizon), y3, label=str(algorithm))

    for ax in (ax1, ax2, ax3):
        ax.legend()
        ax.grid(which="major", color="black", linestyle="-")
        ax.grid(which="minor", color="black", linestyle="-")
    plt.show()

if __name__ == "__main__":
    main()
