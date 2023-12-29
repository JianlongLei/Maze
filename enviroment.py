from consts import *
import numpy as np


class Environment:
    # from 0 to width * height - 1
    states = 0
    # from 0 to 3, means left, top, right, down four directions
    actions = []
    # V value
    reward = []
    start = 0
    end = 0

    def __init__(self, map, start_position, end_position):
        width = np.shape(map)[1]
        height = np.shape(map)[0]
        self.width = width
        self.height = height
        self.states = np.size(map)
        self.reward = []
        for rows in map:
            for item in rows:
                self.reward.append(item)
        self.start = self.axisToState(start_position)
        self.end = self.axisToState(end_position)
        for i in range(self.states):
            item_action = np.array([True] * ACTION_SIZE)
            # on the left side, no left action
            if i % width == 0:
                item_action[ACTION_LEFT] = False
            if i // width == height - 1:
                item_action[ACTION_DOWN] = False
            if i % width == width - 1:
                item_action[ACTION_RIGHT] = False
            if i // width == 0:
                item_action[ACTION_TOP] = False
            self.actions.append(item_action)

    def doAction(self, state_from, action):
        if not self.actions[state_from][action]:
            return state_from
        state_to = state_from
        if action == ACTION_LEFT:
            state_to = state_from - 1
        elif action == ACTION_TOP:
            state_to = state_from - self.width
        elif action == ACTION_RIGHT:
            state_to = state_from + 1
        elif action == ACTION_DOWN:
            state_to = state_from + self.width
        return state_to

    def actionList(self, state):
        if 0 <= state < self.states:
            return ACTION_LIST[self.actions[state]]
        return []

    def isTerminal(self, state):
        return state == self.end

    def axisToState(self, axis):
        if axis[1] < self.height and axis[0] < self.width:
            return self.width * axis[1] + axis[0]
        return -1

    def stateToAxis(self, state):
        if state >= self.states:
            return -1, -1
        return state % self.width, state // self.width
