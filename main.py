from rps_env import *

def main():
    env = RoPaSci360_game()
    env.reset()
    env.board[0, 0] = ROCK
    env.board[0, 2] = paper
    #upper = g.get_upper()
    #lower = g.get_lower()

    #for i in lower:
    #    print(g.check_swing((scissor, -4, 4), (scissor, -2, 2), 'Lower'))"""
    #g.render()
    #print(g.possible_moves('upper'))
    #print(g.possible_moves('lower'))
    env.update(('SLIDE', (1, 4,-4), (1, 4, -3)),('SLIDE', (-2, 4,-2), (-2, 4, -3)))

main()
