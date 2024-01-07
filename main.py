from tkinter import *

from examples import *
from game import Game
from gameUI import GameUI

if __name__ == '__main__':

    # 4*4 normal maze
    # map, x_start, y_start, x_target, y_target = example_2()

    # 5*5 normal maze
    # map, x_start, y_start, x_target, y_target = example_4()

    games = []
    # 8*8 normal maze
    # map, start_point, end_points = example_3()
    # games.append(Game(map, start_point, end_points))

    # 3*3 simple maze
    map, start_point, end_points = example_3()
    games.append(Game(map, start_point, end_points))

    # 5*5 simple maze
    map, start_point, end_points = example_8()
    games.append(Game(map, start_point, end_points))

    window = Tk()
    gameUI = GameUI(window, games, itemSize=50, speed=0.2)
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
    # map, start_point, end_points = example_7()
    # game = Game(map, start_point, end_points)
    # window = Tk()
    # gameUI = GameUI(window, game, itemSize=10, speed=0.1, border=False)
    # gameUI.drawMap()
    # window.mainloop()
