from gym import Env
from gym.spaces import Discrete, Box, Dict

from state import *
from copy import deepcopy

import tensorflow as tf
import numpy as np
import random

INVALID_ACTION_REWARD = -10
VALID_ACTION_REWARD = 10
WIN_REWARD = 100
LOSS_REWARD = -100
EAT_TOKEN = 10

LEFT        = (0, -1)
RIGHT       = (0, +1)
UP_LEFT     = (-1, 0)
UP_RIGHT    = (-1, +1)
DOWN_LEFT   = (+1, -1)
DOWN_RIGHT  = (+1, 0)

NEIGHBOURS = [RIGHT, DOWN_RIGHT, DOWN_LEFT, LEFT, UP_LEFT, UP_RIGHT]

TOKEN_EMPTY = '_'
TOKEN_VOID = np.inf

BOARD_SIZE = 9

N_THROWS = 9

WHAT_BEATS = {'r': 'p', 'p': 's', 's': 'r'}

MAX_TURNS = 360
MAX_SAME_CONFIG = 3

# Upper
ROCK = 1
PAPER = 2
SCISSOR = 3
UPPER = (ROCK, PAPER, SCISSOR)
# Lower
rock = -1
paper = -2
scissor = -3
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

class RoPaSci360_game(Env):
    def __init__(self,
                player = 'upper',
                opponent = 'random',
                log = 'True'):
        
        # Constants
        self.max_turns = 360
        self.log = log
                
        #
        # Observation + Action spaces
        # ---------------------------
        #  Observations: RoPaSci board containing 61 hexes, with 9 types of maximum number of tokens for each player.
        #  Actions: (Every board position) * (Every board position)
        #
        # Note: not every action is legal
        #
        
        self.action_space = Dict({"symbol": Discrete(3), "position": Box(0, 61, shape = (2,))})
        self.observation_space = Box(low = np.int8(0), high = np.int8(-1), shape = (9, 9), dtype = np.int8)
        
        self.player = player
        self.player_2 = self.get_other_player()
        self.opponent = opponent
        
        self.board = None
        self.upper_pcs = None
        self.lower_pcs = None

        self.upper_inv = None
        self.lower_inv = None

        # Current
        self.upper_throws = 9
        self.lower_throws = 9
        self.upper_turns = 0
        self.lower_turns = 0
        
        self.reset()
    
    def seed(self ):pass
        
    def step(self, action):
        assert self.action_space.contains(action), "ACTION ERROR {}".format(action)
        
        reward = 0
        info = {'turn' : self.game.upper_turns,
                'move_type' : None,
                'player' : self.player}
        
        symbol = action['symbol']
        pos1, pos2 = action['position']
        piece = (symbol, pos1, pos2)
        print(symbol)
        print(pos1, pos2)
                
    def reset(self):
        self.board = self.game.new_board()
        
    def render(self):
        pass
    
    def get_other_player(self):
        if self.player == 'upper':
            return 'lower'
        elif self.player == 'lower':
            return 'upper'
    
    def random_agent(self):
        possible_moves = self.game.possible_moves(player = self.player_2)
        return np.random.choice(possible_moves)
    
    @staticmethod
    def new_board():
        """
        return a new board with no tokens

        Returns:
            numpy.ndarray: the new board

        Examples:
            >>> print(RoPaSci360.new_board())
               [['_' '_' '_' '_' '_' inf inf inf inf]
                ['_' '_' '_' '_' '_' '_' inf inf inf]
                ['_' '_' '_' '_' '_' '_' '_' inf inf]
                ['_' '_' '_' '_' '_' '_' '_' '_' inf]
                ['_' '_' '_' '_' '_' '_' '_' '_' '_']
                [inf '_' '_' '_' '_' '_' '_' '_' '_']
                [inf inf '_' '_' '_' '_' '_' '_' '_']
                [inf inf inf '_' '_' '_' '_' '_' '_']
                [inf inf inf inf '_' '_' '_' '_' '_']]
        """
        n = BOARD_SIZE
        board = np.full((BOARD_SIZE, BOARD_SIZE), \
                        TOKEN_EMPTY, dtype = np.object_)
        for i in range(BOARD_SIZE // 2):
            board[i, i + 5:BOARD_SIZE] = TOKEN_VOID
            board[i + 5:BOARD_SIZE, i] = TOKEN_VOID
        return board

    @staticmethod
    def convert_to_axial(position):
        (r, q) = position
        return (BOARD_SIZE // 2 - r, q - BOARD_SIZE // 2)

    @staticmethod
    def convert_to_normal(position):
        (r, q) = position
        return (BOARD_SIZE // 2 - r, q + BOARD_SIZE // 2)

    @staticmethod
    def get_tokens(board, symbol):
        res = np.where(board == symbol)
        return [(symbol, p[0], p[1]) for p in list(zip(res[0], res[1]))]

    def get_upper(self, board):
        upper = list()
        for symbol in UPPER:
            for i in self.get_tokens(board, symbol):
                (s, x, y) = i
                coord = self.convert_to_axial((x, y))
                (r, q) = coord
                upper.append((symbol, r , q))
        return upper

    def get_lower(self, board):
        lower = list()
        for symbol in LOWER:
            for i in self.get_tokens(board, symbol):
                (s, x, y) = i
                coord = self.convert_to_axial((x, y))
                (r, q) = coord
                lower.append((symbol, r , q))
        return lower

    @staticmethod
    def get_neighbours(token):
        (s, r, q) = token
        return [(s, r + x, q + y) for (x, y) in NEIGHBOURS]

    #==================== Check Move =========================#
    def inbound(self, token):
        (s, r, q) = token
        if abs(r) > (BOARD_SIZE // 2) or abs(q) > (BOARD_SIZE // 2):
            return False
        res = np.where(self.board == np.inf)
        out_of_bound = [i for i in list(zip(res[0], res[1]))]
        if self.convert_to_normal((r,q)) in out_of_bound:
            return False
        return True

    def check_slide(self, before, after = None, possible = False):
        """
        Input: before - Selected token.
               * before in axial format (s, r, q) and s in number format,
               * where r and q is the coordinates.

               after - Selected target token.
               * after in axial format (s, r, q) and s in number format,
               * where r and q is the coordinates.

               Player - Upper or Lower.

               Possible - set as a default to False, if True return
                          possible slide moves.

        Output: returning the action type and before after
                if move not valid return false.
        """
        slide = list(filter(self.inbound, self.get_neighbours(before)))
        if possible:
            return [("SLIDE", before, s) for s in slide]
        if after in slide:
            return ("SLIDE", before, after)
        return False

    def check_swing(self, player, before, after = None, possible = False):
        """
        Input: Player - Upper or Lower.

               before - Selected token.
               * before in axial format (s, r, q) and s in number format,
               * where r and q is the coordinates.

               after - Selected target token.
               * after in axial format (s, r, q) and s in number format,
               * where r and q is the coordinates.

               Possible - set as a default to False, if True return
                          possible swing moves.

        Output: returning the action type and before after
                if move not valid return false.
        """
        nbr = self.get_neighbours(before)

        pivot = list()
        for n in nbr:
            (s_b, x, y) = n
            if player == 'upper':
                tokens = self.get_upper(self.board)
                for s in UPPER:
                    if (s, x, y) in tokens:
                        pivot.append((s, x, y))
            if player == 'lower':
                tokens = self.get_lower(self.board)
                for s in LOWER:
                    if (s, x, y) in tokens:
                        pivot.append((s, x, y))
        swing = list()
        for token in pivot:
            for n in self.get_neighbours(token):
                if n != before and n not in nbr:
                    swing.append(n)
        swing = list(dict.fromkeys(swing))
        swing = list(filter(self.inbound, swing))
        if possible:
            return [("SWING", before, s) for s in swing]
        if after in swing:
            return ("SWING", before, after)
        return False

    def check_throw(self, player, token, possible = False):
        """
        Input: token - a token piece
               * (s, r, q) s is in number format, where r and q is in
               * axial format.

               Player - Upper or Lower.

               Possible - set as a default to False, if True return
                          possible throw moves.

        Output: returning the action type, before equal to None and after
                if move not valid return false.
        """
        throwable = []
        if player == 'upper':
            for s in UPPER:
                for coord in GRID:
                    if coord[0] >= self.upper_throws - 5:
                        throwable.append(("THROW", s, coord))

        if player == 'lower':
            for s in LOWER:
                for coord in GRID:
                    if coord[0] <= 5 - self.lower_throws:
                        throwable.append(("THROW", s, coord))
        if possible:
            return throwable
        elif token:
            (s, r, q) = token # s is number format
            if ("THROW", s, (r, q)) in throwable:
                return ("THROW", None, token)
        return False

    def validate_moves(self, symbol, before, after, player):
        """
        Input: symbol - in number format

               before - Selected token.
               *before in positional format from 0 to 61, 0 meaning
               its a throws action.

               after - Selected target token.
               *after in positional format from 0 to 61.

               Player - Upper or Lower.

        Output: returning the action type and before after
                if move not valid return false.
        """
        after = CELL_POS[after - 1]
        after = (symbol, after[0], after[1])
        # If the select a throw action
        if before == 0:
            return self.check_throw(player, after)

        pieces = []
        if player == 'upper':
            pieces = self.get_upper(self.board)
        elif player == 'lower':
            pieces = self.get_lower(self.board)

        before = CELL_POS[before - 1]
        before = (symbol, before[0], before[1])
        # If selected piece not in any of the available pieces.
        if before not in pieces:
            return False

        move_check = self.check_swing(player, before = before, after = after)
        print(move_check)
        if move_check:
            return move_check

        move_check = self.check_slide(before, after)
        if move_check:
            return move_check

        return False

    def possible_moves(self, player):
        possible = True
        actions = list()
        actions.extend(self.check_throw(player, token=None, possible=possible))
        if player == 'upper':
            pieces = self.get_upper(self.board)
        if player == 'lower':
            pieces = self.get_lower(self.board)
        for p in pieces:
            actions.extend(self.check_slide(p, possible=possible))
            actions.extend(self.check_swing(player = player, before = p, possible=possible))
        return actions

    def action_handler(self, symbol, before, after, player):
        move_check = self.validate_moves(symbol, before, after, player)
        if move_check:
            return move_check
        return False

    def update(self, upper_board, lower_board):
        if self.upper_turns == self.lower_turns:
            upper = self.get_upper(upper_board)
            print(upper)