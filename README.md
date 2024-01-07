# Maze
This the assignment 4 of DPRL course, which is required to solve a maze problem by Q-learning. 
We implement dynamic programming tabular Q-learning, ε-greedy Q-learning and Deep Q-Network (DQN).
Besides, we provide a UI for this problem and our algorithms.

To run the project, you need to follow instructions below:
1. install libraries in the requirements.
2. run the main.py.
3. you will see a window displaying a 3×3 maze.\
![maze1.png](figures%2Fmaze1.png)
4. click "Run" button, then you will see the solution by DP Q-learning and ε-greedy Q-learning.
5. click "Previous" or "Next" button, you can switch between two different mazes.\
![maze2.png](figures%2Fmaze2.png)
6. Click "Run" button again to run DQN in the 5×5 maze. It takes time, so be patient.

A simple 3*3 maze's data structure will be:
[[0, 0, 1], [0, 0, 0], [0, 0, 0]], start_point=(2, 0), end_points=[(0, 2)]. The result of 
the problem will be: 

![result](figures/optimal%20policy%20partI.png)

The yellow block shows the start point, the green block shows the terminal block and the 
blue blocks show the path. Optimal policy is shown as black arrows.

# Algorithm

We have three implementations of Q-learning. DP tabular Q-learning, ε-greedy combined with
tabular Q-learning and deep Q-leaning. Each implementation will give plots after running.

# Project Structure

environment.py represents a maze. It holds all states, actions, and other important 
information for a maze.
agent.py holds all implementations of Q-learning. 
game.py combines different agents to different maze environments.
gameUI.py can display a game on a window.
model.py is the model we used in deep Q-learning network.
examples.py has several mazes for testing.
maze.py is a simple version for solving the problem.