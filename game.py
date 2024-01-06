import matplotlib.pyplot as plt

from agent import *
from enviroment import Environment

alpha_list = [0.9, 0.7, 0.5, 0.3, 0.1]
epsilon_list = [0.9, 0.7, 0.5, 0.3, 0.1]


class Game:

    def __init__(self, maze_map, start_position, end_position):
        self.maze_map = maze_map
        self.env = Environment(maze_map, start_position, end_position)

        self.dp_agents = [DPQlearning(self.env, alpha=alpha) for alpha in alpha_list]
        self.greedy_agents = [GreedyQlearning(self.env, epsilon=epsilon) for epsilon in epsilon_list]
        self.dqn_agent = DQNAgent(self.env)

        self.start_position = start_position
        self.width = self.env.width
        self.height = self.env.height

    def solve(self):
        for agent in self.dp_agents + self.greedy_agents:
            agent.train(150)

        self.dqn_agent.train()
        self.draw()
        result = self.dqn_agent.get_result(self.env.start)
        print("result", result)
        # result = self.agent.get_result(self.env.axisToState(self.start_position))
        res = self.dp_agents[0].get_result(self.env.start)
        # print(self.dp_agents[0].q_values[self.env.legal_states])
        # print(res)
        policy = self.dp_agents[0].get_policy()
        # policy = self.agent.get_policy()

        return res, policy

    def draw(self):
        dp_records = [agent.records for agent in self.dp_agents]
        greedy_records = [agent.records for agent in self.greedy_agents]
        # draw figures
        for i, dp_record in enumerate(dp_records):
            dp_label = f"α={alpha_list[i]}"
            plt.plot(dp_record, label=dp_label)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.show()

        plt.figure()
        for i, g_record in enumerate(greedy_records):
            g_label = f"ε={epsilon_list[i]}"
            plt.plot(g_record[:75], label=g_label)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.show()

        plt.figure()
        dqn_records = self.dqn_agent.records
        plt.plot(dqn_records)
        plt.show()

        # plt.figure()
        # loss = self.dqn_agent.loss_list
        # plt.plot(loss)
        # plt.show()
