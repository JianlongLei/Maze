import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from consts import ACTION_SIZE
from enviroment import Environment


class Agent:
    # Q value
    q_value = []

    def __init__(self, environment: Environment,
                 # lambda, discount parameter
                 lda=0.9,
                 # epsilon, exploration & exploitation balancing parameter. Epsilon-Greedy
                 eps=0.9,
                 # alpha, learning rate
                 alp=0.9):

        self.lda = lda
        self.eps = eps
        self.alp = alp
        self.env = environment
        self.q_value = np.empty(shape=(environment.states, ACTION_SIZE))
        self.q_value.fill(-math.inf)
        for state in range(environment.states):
            # value = 1 if self.env.isTerminal(state) else 0
            for action in environment.actionList(state):
                # self.q_value[state][action] = value
                self.q_value[state][action] = 0

    def chooseAction(self, state):
        r = random.random()
        if r >= self.eps:
            return self.chooseBestAction(state)
        else:
            return np.random.choice(self.env.actionList(state))

    def nextState(self, state, visited):
        all_actions = self.env.actionList(state)
        all_states = [self.env.doAction(state, action) for action in all_actions]
        available_states = []
        available_actions = []
        for i, x in enumerate(all_states):
            if x not in visited:
                available_states.append(x)
                available_actions.append(all_actions[i])
        available_states = np.array(available_states)
        max_value_states = np.array([np.max(self.q_value[i]) for i in available_states])
        if len(available_states) == 0:
            return state, 0

        max_index = np.argmax(max_value_states)
        max_states = available_states[max_index]
        if len(available_actions) == 1:
            return max_states, all_actions[0]

        r = random.random()
        if r >= self.eps:
            return max_states, all_actions[max_index]
        else:
            if len(available_states) > 1:
                index = random.randint(0, len(available_states) - 1)
            else:
                index = 0
            return available_states[index], available_actions[index]

    def allQValues(self, state):
        return self.q_value[state][self.env.actions[state]]

    def chooseBestAction(self, state):
        return np.argmax(self.q_value[state])

    def update_q_value(self, state_from, state_to, action):
        q_x_a = self.q_value[state_from][action]
        reward_x = self.env.reward[state_to]
        max_q_xp = max(self.allQValues(state_to))
        self.q_value[state_from][action] = q_x_a + self.alp * (reward_x + self.lda * max_q_xp - q_x_a)

    def learn(self):
        start = time.time()
        episode = 500
        a = []
        diff = []
        span = 0.0001
        for i in range(episode):
            # x = 0
            # while not self.isTerminal(x):
            current = np.copy(self.q_value)
            a.append(current)
            x = self.env.start
            visited = []
            # while not self.env.isTerminal(x):
            #     visited.append(x)
            #     xp, action = self.nextState(x, visited)
            #     if xp == x:
            #         break
            #     self.update_q_value(x, xp, action)
            #     x = xp


            for x in range(self.env.states):
                if self.env.isTerminal(x):
                    continue
                if not self.env.isValid(x):
                    continue
                action = self.chooseAction(x)
                xp = self.env.doAction(x, action)
                self.update_q_value(x, xp, action)


            # curr_diff = np.subtract(self.q_value, current)
            # curr_diff = np.absolute(curr_diff[~np.isnan(curr_diff)])
            # max_diff = np.max(curr_diff)
            # diff.append(max_diff)

            # curr_diff = np.subtract(self.q_value[self.env.start], current[self.env.start])
            # curr_diff = np.absolute(curr_diff[~np.isnan(curr_diff)])
            # max_diff = np.max(curr_diff)
            # diff.append(max_diff)
            # if span > max_diff:
            #     break
        print("total time:", time.time() - start)
        a = np.array(a)
        print("all Q values:\n{a}".format(a=a))
        # print("difference between Q values:\n{diff}".format(diff=np.array(diff)))
        # plt.plot(a)
        # plt.show()


    def getResult(self, start_state):
        x = start_state
        results = []
        while not self.env.isTerminal(x):
            bestAction = self.chooseBestAction(x)
            results.append(x)
            xp = self.env.doAction(x, bestAction)
            # print('{x} go {a} to {p}'
            #       .format(x=self.env.stateToAxis(x),
            #               a=ACTION_NAME[bestAction],
            #               p=self.env.stateToAxis(xp)))
            x = xp
        print(results)
        return results
