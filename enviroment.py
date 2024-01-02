from enum import Enum

import numpy as np

from consts import *


class Action(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


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
        height, width = np.shape(map)
        self.width = width
        self.height = height

        state_size = np.size(map)
        self.legal_states = []
        for x in range(state_size):
            if not self.isTerminal(x) and self.isValid(x):
                self.legal_states.append(x)

        self.states = np.size(map)
        self.reward = []
        for rows in map:
            for item in rows:
                self.reward.append(item)
        self.start = self.axisToState(start_position)
        self.end = self.axisToState(end_position)
        for s in range(self.states):
            # item_action = np.array([True] * ACTION_SIZE)
            # on the left side, no left action
            # if s % width == 0:
            #     item_action[Action.LEFT.value] = False
            # if s // width == height - 1:
            #     item_action[Action.DOWN.value] = False
            # if s % width == width - 1:
            #     item_action[Action.RIGHT.value] = False
            # if s // width == 0:
            #     item_action[Action.UP.value] = False
            state_action = []
            if s % width != 0:
                state_action.append(Action.LEFT)
            if s // width != height - 1:
                state_action.append(Action.DOWN)
            if s % width != width - 1:
                state_action.append(Action.RIGHT)
            if s // width != 0:
                state_action.append(Action.UP)
            # self.actions.append(state_action)
            legal_actions = []
            for action in state_action:
                if self.reward[self.doAction(s, action)] >= 0:
                    legal_actions.append(action)

            self.actions.append(legal_actions)

    def doAction(self, state_from, action: Action):
        # if not self.actions[state_from][action.value]:
        # if action not in self.actions[state_from]:
        #     return state_from
        state_to = state_from
        if action == Action.LEFT:
            state_to = state_from - 1
        elif action == Action.UP:
            state_to = state_from - self.width
        elif action == Action.RIGHT:
            state_to = state_from + 1
        elif action == Action.DOWN:
            state_to = state_from + self.width
        return state_to

    def actionList(self, state):
        if 0 <= state < self.states:
            # return ACTION_LIST[self.actions[state]]
            return self.actions[state]
        return []

    def isTerminal(self, state):
        return state == self.end

    def isValid(self, state):
        return self.reward[state] >= 0

    def axisToState(self, axis):
        if axis[1] < self.height and axis[0] < self.width:
            return self.width * axis[1] + axis[0]
        return -1

    def stateToAxis(self, state):
        if state >= self.states:
            return -1, -1
        return state % self.width, state // self.width
