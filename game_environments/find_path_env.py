from game_environments.environment_template import Environment
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


class FindPath(Environment):
    def __init__(self, start_position=None):
        state_space = [[0, 0, 0],  # The list of possible states the game contains
                       [0, -1, -1],  # With reach (int) value representing the reward of the state
                       [0, 0, 1]]

        action_space = [0, 1, 2, 3]  # [North, East, South, West]
        action_vectors = [[-1, 0], [0, 1], [1, 0], [0, -1]]  # How the actions will effect the position

        super().__init__(state_space, action_space, action_vectors, start_position)

        self.__max_reward = 1

    '''Overrides Environment.step'''
    def step(self, action):
        self.act(action)
        return self.current_observation, self.last_reward, self.is_done(), None

    def is_done(self):
        if self.last_reward == self.__max_reward:
            return True
        else:
            return False

    '''Overrides Environment.render'''
    def render(self): # does nothing atm
        # Loop through obervation, plot character space
        # Loop through state_space, plot reward areas
        # return a matplotlib plot of the game
        cmap = colors.ListedColormap(['red', 'blue'])
        bounds = [0, 10, 20]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-.5, 10, 1))
        ax.set_yticks(np.arange(-.5, 10, 1))

        ax.imshow(self.current_observation, cmap=cmap, norm=norm)
