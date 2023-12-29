import numpy as np

from game import Game
from gameUI import GameUI
from tkinter import *

if __name__ == '__main__':
    # map = np.array([
    #     [0, -1, 0, 0, 0, 0, -1, 1],
    #     [0, -1, -1, 0, -1, 0, -1, 0],
    #     [0,  0, 0, 0, -1, 0, -1, 0],
    #     [-1, 0, -1, -1, -1, 0, -1, 0],
    #     [0, 0, -1, 0, 0, 0, -1, 0],
    #     [0, -1, -1, -1, -1, -1, -1, 0],
    #     [0, 0, 0, 0, -1, 0, 0, 0],
    #     [0, 0, -1, 0, 0, 0, -1, 0],
    # ])
    # x_start = 0
    # y_start = 0
    # x_target = 7
    # y_target = 0



    # map = np.array([
    #     [ 1,  0, 0, 0],
    #     [-1, -1, 0, 0],
    #     [ 0, -1, 0, 0],
    #     [ 0,  0, 0, -1]
    # ])
    # x_start = 0
    # y_start = 2
    # x_target = 0
    # y_target = 0

    # map = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    # x_start = 0
    # y_start = 2
    # x_target = 2
    # y_target = 0

    map = np.array([[0, 0, 0, 0, 1],
                    [0, -1, -1, 0, -1],
                    [0, 0, 0, -1, 0],
                    [-1, -1, 0, -1, 0],
                    [0, 0, 0, 0, 0]])
    x_start = 0
    y_start = 4
    x_target = 4
    y_target = 0


    game = Game(map, (x_start, y_start), (x_target, y_target))
    window = Tk()
    gameUI = GameUI(window, game)
    gameUI.drawMap()
    window.mainloop()