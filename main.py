import numpy as np

from examples import *
from game import Game
from gameUI import GameUI
from tkinter import *

if __name__ == '__main__':
    # 8*8 normal maze
    map, x_start, y_start, x_target, y_target = example_1()
    # 4*4 normal maze
    # map, x_start, y_start, x_target, y_target = example_2()
    # 3*3 simple maze
    # map, x_start, y_start, x_target, y_target = example_3()
    # 5*5 normal maze
    # map, x_start, y_start, x_target, y_target = example_4()
    # 5*5 simple maze
    # map, x_start, y_start, x_target, y_target = example_6()

    game = Game(map, (x_start, y_start), (x_target, y_target))
    window = Tk()
    gameUI = GameUI(window, game, itemSize=50, speed=0.2)
    gameUI.drawMap()
    window.mainloop()

    # 30*30 big maze
    # map, x_start, y_start, x_target, y_target = example_5()
    # game = Game(map, (x_start, y_start), (x_target, y_target))
    # window = Tk()
    # gameUI = GameUI(window, game, itemSize=25, speed=0.1, border=False)
    # gameUI.drawMap()
    # window.mainloop()

    # 60*60 big maze
    # map, x_start, y_start, x_target, y_target = example_7()
    # game = Game(map, (x_start, y_start), (x_target, y_target))
    # window = Tk()
    # gameUI = GameUI(window, game, itemSize=10, speed=0.1, border=False)
    # gameUI.drawMap()
    # window.mainloop()
