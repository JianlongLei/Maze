import math
import random
from consts import ACTION_NAME, ACTION_SIZE, ACTION_LIST
from enviroment import Environment
import numpy as np


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
            for action in environment.actionList(state):
                self.q_value[state][action] = 0

    def chooseAction(self, state):
        r = random.random()
        if r >= self.eps:
            return self.chooseBestAction(state)
        else:
            return np.random.choice(self.env.actionList(state))

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
        episode = 100
        for i in range(episode):
            # x = 0
            # while not self.isTerminal(x):
            for x in range(self.env.states):
                if self.env.isTerminal(x):
                    continue
                action = self.chooseAction(x)
                xp = self.env.doAction(x, action)
                self.update_q_value(x, xp, action)
                # x = xp

    def getResult(self, start_state):
        x = start_state
        print(self.q_value)
        results = []
        while not self.env.isTerminal(x):
            bestAction = self.chooseBestAction(x)
            results.append(x)
            xp = self.env.doAction(x, bestAction)
            print('{x} go {a} to {p}'
                  .format(x=self.env.stateToAxis(x),
                          a=ACTION_NAME[bestAction],
                          p=self.env.stateToAxis(xp)))
            x = xp
        return results
