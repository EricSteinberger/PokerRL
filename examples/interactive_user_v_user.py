# Copyright (c) 2019 Eric Steinberger


"""
This script will start a game of 3-player No-Limit Texas Hold'em with discrete bet sizes in which you can play against
yourself.
"""

from PokerRL.game.InteractiveGame import InteractiveGame
from PokerRL.game.games import DiscretizedNLHoldem

if __name__ == '__main__':
    game_cls = DiscretizedNLHoldem
    args = game_cls.ARGS_CLS(n_seats=3,
                             bet_sizes_list_as_frac_of_pot=[
                                 0.2,
                                 0.5,
                                 1.0,
                                 2.0,
                                 1000.0  # Note that 1000x pot will always be >pot and thereby represents all-in
                             ],
                             stack_randomization_range=(0, 0,),
                             )

    game = InteractiveGame(env_cls=game_cls,
                           env_args=args,
                           seats_human_plays_list=[0, 1, 2],
                           )

    game.start_to_play()
