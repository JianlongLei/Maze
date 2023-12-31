import random

ACTION_LEFT = 0
ACTION_TOP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTION_NAME = ['LEFT', 'TOP', 'RIGHT', 'DOWN']


class Maze:
    # from 0 to width * height - 1
    states = 0
    # from 0 to 3, means left, top, right, down four directions
    actions = []
    # V value
    reward = []
    # Q value
    q_value = []
    # lambda
    lda = 0.9
    # epsilon
    eps = 0.9
    # alpha
    alp = 0.9

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.states = width * height
        self.q_value = [[0, 0, 0, 0] for _ in range(self.states)]
        for i in range(self.states):
            item_action = [ACTION_LEFT, ACTION_TOP, ACTION_RIGHT, ACTION_DOWN]
            # on the left side, no left action
            if i % width == 0:
                item_action.remove(ACTION_LEFT)
                self.q_value[i][ACTION_LEFT] = -1
            if i // width == height - 1:
                item_action.remove(ACTION_TOP)
                self.q_value[i][ACTION_TOP] = -1
            if i % width == width - 1:
                item_action.remove(ACTION_RIGHT)
                self.q_value[i][ACTION_RIGHT] = -1
            if i // width == 0:
                item_action.remove(ACTION_DOWN)
                self.q_value[i][ACTION_DOWN] = -1
            self.actions.append(item_action)
        self.reward = [0 for _ in range(self.states)]
        self.reward[self.states - 1] = 1
        self.reward[self.states // 2] = 2

    def doAction(self, state_from, action):
        if action not in self.actions[state_from]:
            return state_from
        if action == ACTION_LEFT:
            state_to = state_from - 1
        elif action == ACTION_TOP:
            state_to = state_from + self.width
        elif action == ACTION_RIGHT:
            state_to = state_from + 1
        else:
            state_to = state_from - self.width
        return state_to

    def isTerminal(self, state):
        return self.reward[state] > 0

    def chooseAction(self, state):
        r = random.random()
        if r >= self.eps:
            all_actions = self.q_value[state]
            return all_actions.index(max(all_actions))
        else:
            return random.choice(self.actions[state])

    def chooseBestAction(self, state):
        all_actions = self.q_value[state]
        return all_actions.index(max(all_actions))

    def update_q_value(self, state_from, state_to, action):
        q_x_a = self.q_value[state_from][action]
        reward_x = self.reward[state_to]
        max_q_xp = max([self.q_value[state_to][ap] for ap in self.actions[state_to]])
        self.q_value[state_from][action] = q_x_a + self.alp * (reward_x + self.lda * max_q_xp - q_x_a)

    def learn(self):
        episode = 100
        for i in range(episode):
            # x = 0
            # while not self.isTerminal(x):
            for x in range(self.states):
                if self.isTerminal(x):
                    continue
                action = self.chooseAction(x)
                xp = self.doAction(x, action)
                self.update_q_value(x, xp, action)
                # x = xp

    def getResult(self):
        x = 0
        print(self.q_value)
        while not self.isTerminal(x):
            bestAction = self.chooseBestAction(x)
            xp = self.doAction(x, bestAction)
            print('{x} go {a} to {p}'.format(x=x, a=ACTION_NAME[bestAction], p=xp))
            x = xp
