import matplotlib.pyplot as plt

from agent import *
from enviroment import Environment


class Game:

    def __init__(self, maze_map, start_position, end_position):
        self.maze_map = maze_map
        self.env = Environment(maze_map, start_position, end_position)

        self.dp_agent1 = DPQlearning(self.env)
        self.dp_agent2 = DPQlearning(self.env, alpha=0.8)
        self.dp_agent3 = DPQlearning(self.env, alpha=0.6)

        self.greedy_agent1 = GreedyQlearning(self.env)
        self.greedy_agent2 = GreedyQlearning(self.env, epsilon=0.7)
        self.greedy_agent3 = GreedyQlearning(self.env, epsilon=0.5)
        self.greedy_agent4 = GreedyQlearning(self.env, epsilon=0.3)
        self.greedy_agent5 = GreedyQlearning(self.env, epsilon=0.1)

        self.dqn_agent = DQNAgent(self.env)

        self.start_position = start_position
        self.width = self.env.width
        self.height = self.env.height

    def solve(self):
        self.dp_agent1.train()
        self.dp_agent2.train()
        self.dp_agent3.train()

        # self.greedy_agent1.train()
        # self.greedy_agent2.train()
        # self.greedy_agent3.train()
        # self.greedy_agent4.train()
        # self.greedy_agent5.train()

        self.dqn_agent.train()
        result = self.dqn_agent.get_result(self.env.start)
        print("result", result)
        # result = self.agent.get_result(self.env.axisToState(self.start_position))
        res = self.dp_agent1.get_result(self.env.start)
        print(res)
        policy = self.dp_agent1.get_policy()
        # policy = self.agent.get_policy()
        # self.draw()
        return result, policy

    def draw(self):
        dp_records1 = self.dp_agent1.records
        dp_records2 = self.dp_agent2.records
        dp_records3 = self.dp_agent3.records

        g1_records = self.greedy_agent1.records
        g2_records = self.greedy_agent2.records
        g3_records = self.greedy_agent3.records
        g4_records = self.greedy_agent4.records
        g5_records = self.greedy_agent5.records
        # draw figures
        plt.plot(dp_records1, label="α=0.9")
        plt.plot(dp_records2, label="α=0.8")
        plt.plot(dp_records3, label="α=0.6")
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(g1_records[:100], label="ε=0.9")
        plt.plot(g2_records[:100], label="ε=0.7")
        plt.plot(g3_records[:100], label="ε=0.5")
        plt.plot(g4_records[:100], label="ε=0.3")
        plt.plot(g5_records[:100], label="ε=0.1")
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.show()

        plt.figure()
        dqn_records = self.dqn_agent.records
        plt.plot(dqn_records)
        plt.show()

