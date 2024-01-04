import matplotlib.pyplot as plt

from agent import *
from enviroment import Environment


class Game:

    def __init__(self, maze_map, start_position, end_position):
        self.maze_map = maze_map
        self.env = Environment(maze_map, start_position, end_position)
        self.agent = GreedyQlearning(self.env)
        self.agent2 = DPQlearning(self.env)
        self.agent3 = DPQlearning(self.env, alpha=0.8)
        self.agent4 = DPQlearning(self.env, alpha=0.7)
        self.agent5 = DPQlearning(self.env, alpha=0.5)

        self.greedy_agent2 = GreedyQlearning(self.env, epsilon=0.8)
        self.greedy_agent3 = GreedyQlearning(self.env, epsilon=0.6)

        self.start_position = start_position
        self.width = self.env.width
        self.height = self.env.height

    def solve(self):
        self.agent.train()
        self.agent2.train()
        self.agent3.train()
        self.agent4.train()
        self.agent5.train()

        self.greedy_agent2.train()
        self.greedy_agent3.train()
        # result = self.agent.get_result(self.env.axisToState(self.start_position))
        res = self.agent2.get_result(self.env.axisToState(self.start_position))
        policy = self.agent2.get_policy()
        # policy = self.agent.get_policy()
        records2 = self.agent2.records
        records3 = self.agent3.records
        records4 = self.agent4.records
        records5 = self.agent5.records

        records = self.agent.records
        records_g2 = self.greedy_agent2.records
        records_g3 = self.greedy_agent3.records
        # print(records)
        plt.plot(records2)
        plt.plot(records3)
        plt.plot(records4)
        plt.plot(records5)
        plt.show()

        plt.figure()
        plt.plot(records)
        plt.plot(records_g2)
        plt.plot(records_g3)
        plt.show()
        return res, policy
