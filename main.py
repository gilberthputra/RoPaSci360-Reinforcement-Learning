from ropasci_game import *

def main():
    g = RoPaSci360()


    g.board = g.new_board()

    g.board[2, 6] = ROCK
    g.board[8, 8] = scissor
    g.board[7, 7] = scissor
    #upper = g.get_upper()
    #lower = g.get_lower()

    #for i in lower:
    #    print(g.check_swing((scissor, -4, 4), (scissor, -2, 2), 'Lower'))"""
    g.render()
    g.possible_moves('upper')
    g.possible_moves('lower')
    u = g.action_handler(1, 18, 25, 'upper')
    l = g.action_handler(-3, 61, 48, 'lower')


main()
