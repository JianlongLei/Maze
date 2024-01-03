import time
import tkinter
from tkinter import *

from enviroment import Action
from game import Game


class GameUI:
    backgroundColor = "#DFE2E6"
    targetColor = "#03fc7b"
    blockColor = "#fc036b"
    pathColor = "#03bafc"
    startColor = "#fcec03"
    black = "#000000"
    itemSize = 50
    speed = 0.3

    def __init__(self, window, game: Game, itemSize=50, speed=0.3, border=True):
        self.window = window
        self.window.title("MAZE")
        self.itemSize = itemSize
        self.speed = speed
        self.border = border
        topFrame = Frame(window)
        topFrame.pack(side=TOP, fill='both')
        newGame = Button(topFrame, text="Run", command=self.drawResult)
        newGame.pack(side=LEFT, padx=10, pady=10)
        self.canvas = Canvas(self.window, width=game.width * self.itemSize, height=game.height * self.itemSize,
                             bg=self.backgroundColor)
        self.canvas.pack()
        self.game = game

    def drawItem(self, x, y, color, line=True):
        x = x * self.itemSize
        y = y * self.itemSize
        if line:
            self.canvas.create_rectangle(x, y, x + self.itemSize, y + self.itemSize, fill=color)
        else:
            self.canvas.create_rectangle(x, y, x + self.itemSize, y + self.itemSize, fill=color, outline=color)

    def drawLine(self, x, y, color, actions):
        length = self.itemSize
        x = x * length + length / 2
        y = y * length + length / 2

        for action in actions:
            x1, y1 = x, y
            if action == Action.LEFT:
                x1 -= length / 2
            elif action == Action.UP:
                y1 -= length / 2
            elif action == Action.RIGHT:
                x1 += length / 2
            elif action == Action.DOWN:
                y1 += length / 2
            self.canvas.create_line(x, y, x1, y1, width=2, arrow=tkinter.LAST, fill=color)

    def drawMap(self):
        for y in range(self.game.height):
            for x in range(self.game.width):
                if self.game.maze_map[y][x] > 0:
                    self.drawItem(x, y, self.targetColor, self.border)
                elif self.game.maze_map[y][x] < 0:
                    self.drawItem(x, y, self.blockColor, self.border)
                else:
                    self.drawItem(x, y, self.backgroundColor, self.border)
        start = self.game.start_position
        self.drawItem(start[0], start[1], self.startColor, self.border)

    def drawResult(self):
        self.canvas.delete(ALL)
        self.drawMap()
        result, policy = self.game.solve()
        for state in result[1:]:
            time.sleep(self.speed)
            axis = self.game.env.stateToAxis(state)
            self.drawItem(axis[0], axis[1], color=self.pathColor, line=False)
            self.canvas.update()

        for state in self.game.env.legal_states:
            axis = self.game.env.stateToAxis(state)
            self.drawLine(axis[0], axis[1], self.black, policy[state])
