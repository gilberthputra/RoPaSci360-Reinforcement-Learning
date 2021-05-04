import numpy as np
from state import *
from copy import deepcopy

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

BEATS_WHAT = {'r': 's', 'p': 'r', 's': 'p'}
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

class RoPaSci360:
    def __init__(self):
        self.board = None
        self.upper_pcs = None
        self.lower_pcs = None

        self.upper_inv = None
        self.lower_inv = None

        # Current
        self.upper_throws = 9
        self.lower_throws = 9
        self.upper_turns = None
        self.lower_turns = None
        self.game_state = None

    def reset(self, player = 'Upper', random_player = True):
        self.board = new_board()

    def render(self):
        inbound = np.where(self.board != TOKEN_VOID)
        inbound = [i for i in list(zip(inbound[0], inbound[1]))]
        token_pos = [i for i in inbound if self.board[i[0]][i[1]] != TOKEN_EMPTY]
        tokens = [self.board[i[0]][i[1]] for i in token_pos]
        symbols = []
        for t in tokens:
            if t in (1, 2, 3):
                symbols.append(('R', 'P', 'S')[t - 1])
            if t in (-1, -2, -3):
                symbols.append(('r', 'p', 's')[abs(t) - 1])
        new_board = deepcopy(self.board)
        for i in range(len(token_pos)):
            (x, y) = token_pos[i]
            new_board[x][y] = symbols[i]

        template ="""
####################### RoPaSci360 ########################
#                                                         #
#              .-'-._.-'-._.-'-._.-'-._.-'-.              #
#             |{00:}|{01:}|{02:}|{03:}|{04:}|             #
#           .-'-._.-'-._.-'-._.-'-._.-'-._.-'-.           #
#          |{05:}|{06:}|{07:}|{08:}|{09:}|{10:}|          #
#        .-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-.        #
#       |{11:}|{12:}|{13:}|{14:}|{15:}|{16:}|{17:}|       #
#     .-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-.     #
#    |{18:}|{19:}|{20:}|{21:}|{22:}|{23:}|{24:}|{25:}|    #
#  .-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-.  #
# |{26:}|{27:}|{28:}|{29:}|{30:}|{31:}|{32:}|{33:}|{34:}| #
# '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-' #
#    |{35:}|{36:}|{37:}|{38:}|{39:}|{40:}|{41:}|{42:}|    #
#    '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'    #
#       |{43:}|{44:}|{45:}|{46:}|{47:}|{48:}|{49:}|       #
#       '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'       #
#          |{50:}|{51:}|{52:}|{53:}|{54:}|{55:}|          #
#          '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'          #
#             |{56:}|{57:}|{58:}|{59:}|{60:}|             #
#             '-._.-'-._.-'-._.-'-._.-'-._.-'             #
#                                                         #
###########################################################"""

        cells = list()
        for coord in inbound:
            if coord in token_pos:
                cell = str(new_board[coord[0]][coord[1]]).center(5)
            else:
                cell = "     "
            cells.append(cell)
        # fill in the template to create the board drawing, then print!
        board = template.format(*cells)
        print(board)


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

    def get_upper(self):
        upper = list()
        for symbol in UPPER:
            for i in self.get_tokens(self.board, symbol):
                (s, x, y) = i
                coord = self.convert_to_axial((x, y))
                (r, q) = coord
                upper.append((symbol, r , q))
        return upper

    def get_lower(self):
        lower = list()
        for symbol in LOWER:
            for i in self.get_tokens(self.board, symbol):
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
            if player == 'Upper':
                tokens = self.get_upper()
                for s in UPPER:
                    if (s, x, y) in tokens:
                        pivot.append((s, x, y))
            if player == 'Lower':
                tokens = self.get_lower()
                for s in LOWER:
                    if (s, x, y) in tokens:
                        pivot.append((s, x, y))
        swing = list()
        for token in pivot:
            for n in self.get_neighbours(token):
                if n != before and n not in nbr:
                    swing.append(n)
        swing = list(dict.fromkeys(swing))
        if possible:
            return [("SWING", before, s) for s in swing]
        if after in swing:
            return ("SWING", before, after)
        return False

    def check_throw(self, player, token = None, possible = False):
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
        if player == 'Upper':
            for s in UPPER:
                for coord in GRID:
                    if coord[0] >= self.upper_throws - 5:
                        throwable.append(("THROW", s, coord))

        if player == 'Lower':
            for s in LOWER:
                for coord in GRID:
                    if coord[0] <= 5 - self.lower_throws:
                        throwable.append(("THROW", s, coord))
        if possible:
            return throwable
        else:
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

        after = CELL_POS[after - 1] # Convert into axial format
        (r2, q2) = after

        # If the select a throw action
        if before == 0:
            return self.check_throw(player, (symbol, r2, q2))

        pieces = []
        if player == 'Upper':
            pieces = self.get_upper()
        elif player == 'Lower':
            pieces = self.get_lower()

        before = CELL_POS[before - 1] # Convert into axial format
        (r1, q1) = before

        # If selected piece not in any of the available pieces.
        if (symbol, r1, q1) not in pieces:
            return False

        move_check = self.check_swing((symbol, r1, q1), (symbol, r2, q2), player)
        if move_check:
            return move_check

        move_check = self.check_slide((symbol, r1, q1), (symbol, r2, q2))
        if move_check:
            return move_check

        return False

    def possible_moves(self, player):
        possible = True
        throw = self.check_throw(player, possible)
        print(throw)

    """
    TODO:
    POSSIBLE MOVES
    VALIDATE MOVES
    UPDATE FUNCTION


    """
