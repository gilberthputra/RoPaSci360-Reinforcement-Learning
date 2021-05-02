# RoPaSci360 Reinforcement Learning
> Game Project for COMP30024 Universtiy of Melbourne  Semester 1, 2021.
> __NOTE__: This is just additional project for me to learn (NOT MARKED). The original solution made for the project can be found at https://github.com/gilberthputra/RoPaSci360.git

## The Game
There will be 2 players (Upper and Lower), with each have a maximum number of 9
throws. The player have an option to throw 3 types of tokens which are __Rock__,
__Paper__ and __Scissor__ (Symbols). There is 2 another possible actions which are slide
and swing. (To understand the rule better please have a read of the game rules,
which can be found in the original solution repo as game_rules.pdf)

## The Environment:
In this project, I use OpenAI gym to utilise their Proximal Policy Optimization algorithm to be specific PPO2.

Action space here is defined as a dictionary of Symbol which is Discrete (0, 1, 2)
and Position which is Box from 0 to 61 with shape (2, ). As there is 3 symbols for
each player, add the result of action space by 1 (1, 2, 3) representing (ROCK, PAPER, SCISSOR).
If player is *LOWER*, multiply the symbol by -1 to represent the lower pieces.

> Dict({"symbol": Discrete(3), "position": Box(0, 61, shape = (2,)})
For example, {"symbol": 2, "position": (1, 7)}.

The position will produce 2 positions which is before and after from position 0 to 61. Position 0 representing a __*THROW*__ action and 1 to 61 will represent the position in the board. For example,
position 1 represent (4, -4) in the Axial Coordinate system.

```
                  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
                 |   1   |   2   |   3   |   4   |   5   |
                 |  4,-4 |  4,-3 |  4,-2 |  4,-1 |  4, 0 |
              ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
             |   6   |   7   |   8   |   9   |   10  |   11  |
             |  3,-4 |  3,-3 |  3,-2 |  3,-1 |  3, 0 |  3, 1 |
          ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
         |   12  |   13  |   14  |   15  |   16  |   17  |   18  |
         |  2,-4 |  2,-3 |  2,-2 |  2,-1 |  2, 0 |  2, 1 |  2, 2 |
      ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
     |   19  |   20  |   21  |   22  |   23  |   24  |   25  |   26  |
     |  1,-4 |  1,-3 |  1,-2 |  1,-1 |  1, 0 |  1, 1 |  1, 2 |  1, 3 |
  ,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-.
 |   27  |   28  |   29  |   30  |   31  |   32  |   33  |   34  |   35  |
 |  0,-4 |  0,-3 |  0,-2 |  0,-1 |  0, 0 |  0, 1 |  0, 2 |  0, 3 |  0, 4 |
  `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
     |   35  |   36  |   37  |   38  |   39  |   40  |   41  |   42  |
     | -1,-3 | -1,-2 | -1,-1 | -1, 0 | -1, 1 | -1, 2 | -1, 3 | -1, 4 |
      `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
         |   43  |   44  |   45  |   46  |   47  |   48  |   49  |
         | -2,-2 | -2,-1 | -2, 0 | -2, 1 | -2, 2 | -2, 3 | -2, 4 |
          `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'
             |   50  |   51  |   52  |   53  |   54  |   55  |
             | -3,-1 | -3, 0 | -3, 1 | -3, 2 | -3, 3 | -3, 4 |   key:
              `-._,-' `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'     ,-' `-.
                 |   56  |   57  |   58  |   59  |   60  |       | input |
                 | -4, 0 | -4, 1 | -4, 2 | -4, 3 | -4, 4 |       |  r, q |
                  `-._,-' `-._,-' `-._,-' `-._,-' `-._,-'         `-._,-'
```
>>ADD OBSERVATION SPACE

#### *Step*
In the step function, the action will be check against a validation function. This
validation function will check if the move from before and after. If before is 0, then
check against a throw function. If the move from before and after is valid then return
the move otherwise return False which will return a penalty for making invalid move.

In this case, the reward will be in the order of priority of the movement which
"SWING" will be the highest reward, "SLIDE" being the second and "THROW" being the last.

The board will be updated when each player have decide a move as this is a simultaneous game.

### TODO:
* Add additional moves function in the game engine
* Add update function in the game engine
* Implement a render environment by using a GUI
* Add reset function in the game engine
* Try to just train using PPO2 against original solution,
* then add a self-play wrapper
