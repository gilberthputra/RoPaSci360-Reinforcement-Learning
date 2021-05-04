from rps_env_v1 import *

def main():
    env = RoPaSci360()
    env.reset()
    env.get_neighbours((1, -4, 4))

main()
