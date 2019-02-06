# Copyright (c) 2019 Eric Steinberger


import copy

from PokerRL.game import bet_sets
from PokerRL.game.poker_env_args import DiscretizedPokerEnvArgs, LimitPokerEnvArgs, NoLimitPokerEnvArgs


class LBRArgs:
    """
    Argument object for LBR
    """

    def __init__(self,
                 lbr_bet_set=bet_sets.OFF_TREE_11,
                 n_lbr_hands_per_seat=30000,
                 lbr_check_to_round=None,  # recommended to set to Poker.TURN for 4-round games.
                 n_parallel_lbr_workers=10,
                 use_gpu_for_batch_eval=True,
                 DISTRIBUTED=False,
                 ):
        """

        Args:
            lbr_bet_set (list):             List of floats; fractions of pot that LBR can consider to bet. (Note:
                                            only relevant in discretized and no-limit games. Limit games always allow 3
                                            actions)
            n_lbr_hands_per_seat (int):              Number of LBR hands to compute.

            lbr_check_to_round (game round from Poker):
                                            In the original paper, LBR has been shown to perform better, when it only
                                            check/calls until the Turn in No-Limit Texas Hold'em. It is also faster if
                                            one does that. We recommend setting this to Poker.TURN in games with 4
                                            rounds and to None in Leduc-like games.
            n_parallel_lbr_workers (int):   Number of workers. Only relevant if running distributed
            use_gpu_for_batch_eval (bool):  Whether to use the GPU for batched strategy queries. Recommended for
                                            big neural networks and games with many different hands (like Hold'em), if
                                            available.
            DISTRIBUTED (bool):             Whether to use ray and run distributed or to run locally.
        """
        self.lbr_bet_set = lbr_bet_set
        self.n_lbr_hands = n_lbr_hands_per_seat
        self.lbr_check_to_round = lbr_check_to_round
        self.n_workers = n_parallel_lbr_workers if DISTRIBUTED else 1
        self.use_gpu_for_batch_eval = use_gpu_for_batch_eval
        self.DISTRIBUTED = DISTRIBUTED

    def get_lbr_env_args(self, agents_env_args):
        arg_cls = type(agents_env_args)

        if arg_cls is DiscretizedPokerEnvArgs:
            return DiscretizedPokerEnvArgs(
                n_seats=agents_env_args.n_seats,
                starting_stack_sizes_list=copy.deepcopy(agents_env_args.starting_stack_sizes_list),
                bet_sizes_list_as_frac_of_pot=copy.deepcopy(self.lbr_bet_set),
                stack_randomization_range=(0, 0),
                use_simplified_headsup_obs=agents_env_args.use_simplified_headsup_obs,
                uniform_action_interpolation=False
            )

        elif arg_cls is LimitPokerEnvArgs:
            return LimitPokerEnvArgs(
                n_seats=agents_env_args.n_seats,
                starting_stack_sizes_list=copy.deepcopy(agents_env_args.starting_stack_sizes_list),
                stack_randomization_range=(0, 0),
                use_simplified_headsup_obs=agents_env_args.use_simplified_headsup_obs,
                uniform_action_interpolation=False
            )

        elif arg_cls is NoLimitPokerEnvArgs:
            raise NotImplementedError("Currently not supported")

        else:
            raise TypeError(arg_cls)
