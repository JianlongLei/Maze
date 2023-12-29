import time
from tkinter import *
from game import Game


class GameUI:
    backgroundColor = "#DFE2E6"
    targetColor = "#03fc7b"
    blockColor = "#fc036b"
    pathColor = "#03bafc"
    startColor = "#fcec03"
    itemSize = 50

    def __init__(self, window, game: Game):
        self.window = window
        self.window.title("MAZE")
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


    def drawMap(self):
        for y in range(self.game.height):
            for x in range(self.game.width):
                if self.game.maze_map[y][x] > 0:
                    self.drawItem(x, y, self.targetColor)
                elif self.game.maze_map[y][x] < 0:
                    self.drawItem(x, y, self.blockColor)
                else:
                    self.drawItem(x, y, self.backgroundColor)
        start = self.game.start_position
        self.drawItem(start[0], start[1], self.startColor)

    def drawResult(self):
        self.canvas.delete(ALL)
        self.drawMap()
        result = self.game.solve()
        for state in result[1:]:
            time.sleep(0.3)
            axis = self.game.env.stateToAxis(state)
            self.drawItem(axis[0], axis[1], color=self.pathColor, line=False)
            self.canvas.update()