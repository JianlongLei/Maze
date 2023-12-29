import time

from agent import Agent
from enviroment import Environment
from tkinter import *


class Game:

    def __init__(self, maze_map, start_position, end_position):
        self.maze_map = maze_map
        self.env = Environment(maze_map, start_position, end_position)
        self.agent = Agent(self.env)
        self.start_position = start_position
        self.width = self.env.width
        self.height = self.env.height


    def solve(self):
        self.agent.learn()
        result = self.agent.getResult(self.env.axisToState(self.start_position))
        return result
