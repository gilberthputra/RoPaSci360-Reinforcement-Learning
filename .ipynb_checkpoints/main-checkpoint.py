from rps_env import *

def main():
    env = RoPaSci360_game()


    env.board = env.new_board()

    #upper = g.get_upper()
    #lower = g.get_lower()

    #for i in lower:
    #    print(g.check_swing((scissor, -4, 4), (scissor, -2, 2), 'Lower'))"""
    g.render()
    print(g.possible_moves('upper'))
    print(g.possible_moves('lower'))
    


main()
