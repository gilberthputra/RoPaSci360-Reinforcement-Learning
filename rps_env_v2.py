from gym import Env
from gym.spaces import Discrete, Box, Dict

from state import *
from copy import deepcopy
from random import choice

import numpy as np

INVALID_ACTION_REWARD = -10
VALID_ACTION_REWARD   = 10
WIN_REWARD            = 100
DRAW_REWARD           = 0
LOSS_REWARD           = -100
CAPTURED              = -10
EAT_TOKEN             = 10

MAX_TURNS             = 360
MAX_SAME_CONFIG       = 3
MAX_THROWS            = 9
BOARD_SIZE            = 9

EMPTY = (0, 0, 0)

TOKEN_EMPTY = 0
TOKEN_VOID = -2

ROCK = 1
PAPER = 2
SCISSOR = 3
SYMBOLS = (ROCK, PAPER, SCISSOR)

PRINT_SYM = {1: 'R', 2: 'P', 3: 'S'}

BEATS_WHAT = {ROCK: SCISSOR, PAPER: ROCK, SCISSOR: PAPER}
WHAT_BEATS = {SCISSOR: ROCK, ROCK: PAPER, PAPER: SCISSOR}

# Axial grid system
AXIAL_GRID = [(4, -4), (4, -3), (4, -2), (4, -1), (4, 0), 
        (3, -4), (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), 
        (2, -4), (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), 
        (1, -4), (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), 
        (0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), 
        (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3), (-2, 4), 
        (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3), (-3, 4), 
        (-4, 0), (-4, 1), (-4, 2), (-4, 3), (-4, 4)] 
AXIAL_POS = {}
for pos in range(len(AXIAL_GRID)):
    AXIAL_POS[pos] = AXIAL_GRID[pos]

LEFT        = (0, -1)
RIGHT       = (0, +1)
UP_LEFT     = (-1, 0)
UP_RIGHT    = (-1, +1)
DOWN_LEFT   = (+1, -1)
DOWN_RIGHT  = (+1, 0)

NEIGHBOURS = [RIGHT, DOWN_RIGHT, DOWN_LEFT, LEFT, UP_LEFT, UP_RIGHT]