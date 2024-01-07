import matplotlib.pyplot as plt

from agent import *
from enviroment import Environment

alpha_list = [0.9, 0.7, 0.5, 0.3, 0.1]
epsilon_list = [0.9, 0.7, 0.5, 0.3, 0.1]


class Game:

    def __init__(self, maze_map, start_position, end_position, dqn=False):
        self.dqn = dqn
        self.maze_map = maze_map
        self.env = Environment(maze_map, start_position, end_position)

        if self.dqn:
            self.dqn_agent = DQNAgent(self.env)
        else:
            self.dp_agents = [DPQlearning(self.env, alpha=alpha) for alpha in alpha_list]
            self.greedy_agents = [GreedyQlearning(self.env, epsilon=epsilon) for epsilon in epsilon_list]

        self.start_position = start_position
        self.width = self.env.width
        self.height = self.env.height

    def solve(self):
        if self.dqn:
            self.dqn_agent.train()
            result = self.dqn_agent.get_result(self.env.start)
            policy = self.dqn_agent.get_policy()
        else:
            for agent in self.dp_agents + self.greedy_agents:
                agent.train(150)
            result = self.dp_agents[0].get_result(self.env.start)
            policy = self.dp_agents[0].get_policy()
            optimal_q_val1 = max(self.dp_agents[0].q_values[self.env.start])
            optimal_q_val2 = max(self.greedy_agents[0].q_values[self.env.start])
            print(optimal_q_val1, optimal_q_val2)
            print(self.dp_agents[0].q_values)
            print(self.greedy_agents[0].q_values)

        self.draw()
        return result, policy

    def draw(self):
        if self.dqn:
            plt.figure()
            dqn_records = self.dqn_agent.records
            plt.plot(dqn_records)
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            plt.show()

            plt.figure()
            plt.plot(self.dqn_agent.loss_list)
            plt.xlabel('Updates')
            plt.ylabel('Loss')
            plt.show()
        else:
            dp_records = [agent.records for agent in self.dp_agents]
            greedy_records = [agent.records for agent in self.greedy_agents]

            plt.figure()
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

