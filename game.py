from agent import *
from enviroment import Environment


class Game:

    def __init__(self, maze_map, start_position, end_position):
        self.maze_map = maze_map
        self.env = Environment(maze_map, start_position, end_position)
        self.agent = GreedyQlearning(self.env)
        self.agent2 = DPQlearning(self.env)
        self.start_position = start_position
        self.width = self.env.width
        self.height = self.env.height

    def solve(self):
        self.agent.learn()
        self.agent2.train()
        result = self.agent.get_result(self.env.axisToState(self.start_position))
        res = self.agent2.get_result(self.env.axisToState(self.start_position))

        return result
