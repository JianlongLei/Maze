import math
import random
import time
from collections import deque, namedtuple

import numpy as np
import torch

from enviroment import Environment, Action
from model import DQNModel


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
        span = 1e-30
        for _ in range(episode):
            self.update()
            self._record()

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

    def __init__(self, environment: Environment, gamma=0.9, alpha=0.9, epsilon=0.9):
        """
        Initialization function.
        :param environment:
        :param gamma: γ, discount parameter.
        :param epsilon: ε, probability of exploration.
        :param alpha: α, learning rate.
        """

        super().__init__(environment, gamma, alpha)

        self.epsilon = epsilon
        self.q_values = np.empty(shape=(environment.states, len(Action)))
        self.q_values.fill(-math.inf)
        for state in range(environment.states):
            for action in environment.actionList(state):
                self.q_values[state][action.value] = 0

    def chooseAction(self, state):
        r = random.random()
        max_val = max(self.q_values[state])
        min_val = min(self.q_values[state])
        if r >= self.epsilon and max_val != min_val:
            return self.chooseBestAction(state)
        else:
            return np.random.choice(self.env.actionList(state))

    def allQValues(self, state):
        return self.q_values[state]

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
            # if span > max_diff:
            #     turns += 1
            #     if turns >= 10:
            #         realEpisode = i
            #         break
        # print("total time:", time.time() - start)
        # print("total epochs:", realEpisode)
        # print("diff:", all_diff)
        # print("gradient:", np.gradient(all_diff))
        # plt.plot(all_diff)
        # plt.show()


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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'is_end'))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(Agent):
    def __init__(self, environment: Environment, gamma=0.9, alpha=0.9, epsilon=0.9, lr=5e-3):
        super().__init__(environment, gamma, alpha)
        self.epsilon = epsilon
        self.memory = ReplayMemory(capacity=5000)

        self.q_network = DQNModel(self.env.states, len(Action))
        self.target_network = DQNModel(self.env.states, len(Action))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_function = torch.nn.MSELoss()
        self.loss_list = []

    def get_best_action(self, state):
        # select best action from legal actions
        legal_action = [a.value for a in self.env.actionList(state)]
        with torch.no_grad():
            one_hot_encoded = np.eye(self.env.states)[state]
            state_tensor = torch.tensor(one_hot_encoded, dtype=torch.float32)
            q_values = self.q_network(state_tensor)

        max_q_value = torch.max(q_values[legal_action])
        a_value = torch.where(q_values == max_q_value)[0].item()
        return Action(a_value)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.env.actionList(state))
        else:
            return self.get_best_action(state)

    def update(self):
        if len(self.memory) < 64:
            return
        transitions = self.memory.sample(64)
        batch = Transition(*zip(*transitions))

        state_batch = np.eye(self.env.states)[list(batch.state)]
        state_batch = torch.tensor(state_batch).float()

        action_batch = [[a.value] for a in batch.action]
        action_batch = torch.tensor(action_batch)

        reward_batch = torch.tensor(batch.reward).float()

        next_state_batch = np.eye(self.env.states)[list(batch.next_state)]
        next_state_batch = torch.tensor(next_state_batch).float()

        is_end_batch = torch.tensor(batch.is_end)

        current_q_values = self.q_network(state_batch)
        current_q_values = current_q_values.gather(1, action_batch)
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - is_end_batch)

        loss = self.loss_function(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_list.append(loss.item())

    def train(self, episode=500):
        for i in range(episode):
            state = self.env.start

            for state in range(self.env.states):
                action = self.select_action(state)
                next_state = self.env.doAction(state, action)
                reward = self.env.reward[next_state]
                is_end = self.env.isTerminal(next_state)

                self.memory.push(state, action, next_state, reward, int(is_end))
                self.update()

                if is_end:
                    break

            self.update_target()

            with torch.no_grad():
                start = self.env.start
                one_hot_encoded = np.eye(self.env.states)[start]
                state_tensor = torch.tensor(one_hot_encoded).float()
                q_values = self.q_network(state_tensor)
            max_q = max(q_values).item()
            self.records.append(max_q)

    def get_result(self, start_state):

        one_hot_encoded = np.eye(self.env.states)[self.env.legal_states]
        state_tensor = torch.tensor(one_hot_encoded).float()
        q_values = self.q_network(state_tensor)
        print(q_values)

        res = [start_state]
        state = start_state
        step = 0
        while True:
            action = self.get_best_action(state)
            next_state = self.env.doAction(state, action)
            print(state, action, next_state)
            state = next_state
            step += 1
            if self.env.isTerminal(state) or step > self.env.states:
                # break dead loop
                break
            res.append(state)

        return res

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
