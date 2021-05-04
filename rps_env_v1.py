from gym import Env
from gym.spaces import Discrete, Box, Dict

from state import *
from copy import deepcopy
from random import choice

import tensorflow as tf
import numpy as np


MAX_TURNS = 360
MAX_SAME_CONFIG = 3
MAX_THROWS = 9
BOARD_SIZE = 9


# Upper
ROCK = 1
PAPER = 2
SCISSOR = 3
UPPER = (ROCK, PAPER, SCISSOR)
# Lower
rock = 4
paper = 5
scissor = 6
LOWER = (rock, paper, scissor)

SYMBOLS = {(1): 'R', (2): 'P', (3): 'S',
           (-1): 'r', (-2): 'p', (-3): 's'}

BEATS_WHAT = {ROCK: scissor, PAPER: rock, SCISSOR: paper,
              rock: SCISSOR, paper: ROCK, scissor: PAPER}

# Axial grid system
GRID = [(4, -4), (4, -3), (4, -2), (4, -1), (4, 0),
        (3, -4), (3, -3), (3, -2), (3, -1), (3, 0), (3, 1),
        (2, -4), (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2),
        (1, -4), (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3),
        (0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, -4),
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3), (-2, -4),
        (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3), (-3, -4),
        (-4, 0), (-4, 1), (-4, 2), (-4, 3), (-4, 4)]
CELL_POS = {}
for pos in range(len(GRID)):
    CELL_POS[pos] = GRID[pos]

LEFT        = (0, -1)
RIGHT       = (0, +1)
UP_LEFT     = (-1, 0)
UP_RIGHT    = (-1, +1)
DOWN_LEFT   = (+1, -1)
DOWN_RIGHT  = (+1, 0)

NEIGHBOURS = [RIGHT, DOWN_RIGHT, DOWN_LEFT, LEFT, UP_LEFT, UP_RIGHT]

class RoPaSci360(Env):
    def __init__(self, player = 'upper', opponent = 'random', log = True):
        assert player=='upper' or player =='lower', "Put player as either 'upper' or 'lower'"

        self.max_turns = MAX_TURNS
        self.log = log
        #
        # Observation + Action spaces
        # ---------------------------
        #  Observations: RoPaSci board containing 61 hexes, with 9 types of maximum number of tokens for each player.
        #  Actions: (Every board position) * (Every board position)
        #
        # Note: not every action is legal
        #

        self.action_space = Dict({"symbol": Discrete(3), "position": Box(0, 61, shape = (2,), dtype=np.int8)})
        self.observation_space = Dict({'upper': Box(low = np.int8(0), high = np.int8(-1), shape = (9, 3), dtype = np.int8),
                                       'lower': Box(low = np.int8(0), high = np.int8(-1), shape = (9, 3), dtype = np.int8)})

        self.player_1 = player
        self.player_2 = self.get_other_player()
        self.opponent = opponent

        self.reset()

    def reset(self):
        self.pieces = {'upper': np.full((9, 3), 0, dtype = np.int8),
                       'lower': np.full((9, 3), 0, dtype = np.int8)}
        self.upper_inv = None
        self.lower_inv = None
        self.upper_throws = MAX_THROWS
        self.lower_throws = MAX_THROWS

        self.upper_turn = 0
        self.lower_turn = 0

        self.done = False

    def render(self): pass

    def get_other_player(self):
        if self.player_1 == 'upper':
            return 'lower'
        elif self.player_1 == 'lower':
            return 'upper'

    def get_upper(self):
        upper = list()
        for i in self.pieces['upper']:
            if i[0] != 0:
                upper.append(i)
        return upper

    def get_lower(self):
        lower = list()
        for i in self.pieces['lower']:
            if i[0] != 0:
                upper.append(i)
        return lower

    @staticmethod
    def get_neighbours(token):
        (s, r, q) = token
        nbr = [(r + x, q + y) for (x, y) in NEIGHBOURS]
        nbr = list(filter(GRID, nbr))
        print(nbr)
