import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from enviroment import Environment, Action


class Agent:
    def __init__(self, environment: Environment, gamma=0.9, alpha=0.9):
        self.env = environment
        self.gamma = gamma
        self.alpha = alpha
        self.q_values = np.zeros((self.env.states, len(Action)))

        self.records = []

    def update(self):
        pass

    def _record(self):
        start_state = self.env.start
        q_val = self.q_values[start_state]
        self.records.append(max(q_val))

    def train(self, episode=50):
        for _ in range(episode):
            self.update()
            self._record()
        # print("q-value:", self.q_values)

    def get_result(self, start_state):
        x = start_state
        results = []
        while not self.env.isTerminal(x):
            best_value = np.argmax(self.q_values[x])
            best_action = Action(best_value)

            results.append(x)
            xp = self.env.doAction(x, best_action)
            x = xp
        print("res:", results)
        return results

    def get_policy(self):
        policy = []
        for row in self.q_values:
            actions = []
            max_indices = np.where(row == np.max(row))
            for val in max_indices[0]:
                actions.append(Action(val))
            policy.append(actions)
        return policy


class GreedyQlearning(Agent):

    def __init__(self, environment: Environment, gamma=0.9, epsilon=0.9, alpha=0.9):
        """
        Initialization function.
        :param environment:
        :param gamma: γ, discount parameter.
        :param epsilon: ε, exploration & exploitation balancing parameter of ε-greedy.
        :param alpha: α, learning rate.
        """

        super().__init__(environment, gamma, alpha)

        self.epsilon = epsilon
        self.q_values = np.empty(shape=(environment.states, len(Action)))
        self.q_values.fill(-math.inf)
        for state in range(environment.states):
            # value = 1 if self.env.isTerminal(state) else 0
            for action in environment.actionList(state):
                # self.q_value[state][action] = value
                self.q_values[state][action.value] = 0

    def chooseAction(self, state):
        r = random.random()
        if r >= self.epsilon:
            return self.chooseBestAction(state)
        else:
            return np.random.choice(self.env.actionList(state))

    def allQValues(self, state):
        return self.q_values[state]
        # return self.q_value[state][self.env.actions[state]]

    def chooseBestAction(self, state):
        value = np.argmax(self.q_values[state])
        return Action(value)

    def update_q_value(self, state_from, state_to, action: Action):
        val = action.value
        q_x_a = self.q_values[state_from][val]
        reward_x = self.env.reward[state_to]
        max_q_xp = max(self.allQValues(state_to))
        update_value = self.alpha * (reward_x + self.gamma * max_q_xp - q_x_a)
        self.q_values[state_from][val] = q_x_a + update_value
        return update_value

    def train(self, episode=100):
        start = time.time()
        episode = 2000
        all_diff = []
        span = 1e-30
        turns = 0
        realEpisode = 0
        for i in range(episode):
            max_diff = -math.inf
            for x in self.env.legal_states:
                if self.env.isTerminal(x) or not self.env.isValid(x):
                    continue
                action = self.chooseAction(x)
                xp = self.env.doAction(x, action)
                diff = self.update_q_value(x, xp, action)
                diff = abs(diff)
                if diff > max_diff:
                    max_diff = diff
            all_diff.append(max_diff)
            self._record()
            if span > max_diff:
                turns += 1
                if turns >= 10:
                    realEpisode = i
                    break
        print("total time:", time.time() - start)
        print("total epochs:", realEpisode)
        print("diff:", all_diff)
        print("gradient:", np.gradient(all_diff))
        plt.plot(all_diff)
        plt.show()


class DPQlearning(Agent):
    def update(self):
        # update q value
        new_q_values = np.copy(self.q_values)  # deep copy

        for state in self.env.legal_states:
            for action in self.env.actions[state]:
                new_q_values[state][action.value] = self.calculate_q_value(state, action)
        self.q_values = new_q_values

    def calculate_q_value(self, cur_state, action: Action):
        next_state = self.env.doAction(cur_state, action)
        q_value = self.q_values[cur_state][action.value]
        reward = self.env.reward[next_state]
        max_q = max(self.q_values[next_state])

        diff = reward + self.gamma * max_q - q_value
        new_q_value = q_value + self.alpha * diff

        return new_q_value
