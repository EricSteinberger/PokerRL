# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.base.PokerEnv import PokerEnv as _PokerEnv
from PokerRL.game.poker_env_args import LimitPokerEnvArgs


class LimitPokerEnv(_PokerEnv):
    """
    Adapts the dynamics of PokerEnv to fit to fixed-limit rules.
    """

    ARGS_CLS = LimitPokerEnvArgs

    def __init__(self,
                 env_args,
                 is_evaluating,
                 lut_holder):
        assert isinstance(env_args, LimitPokerEnvArgs)
        super().__init__(env_args=env_args, lut_holder=lut_holder, is_evaluating=is_evaluating)
        self.uniform_action_interpolation = False
        self.N_ACTIONS = env_args.N_ACTIONS

    def _get_env_adjusted_action_formulation(self, action_int):
        if action_int == 0:
            return [0, -1]
        if action_int == 1:
            return [1, -1]
        elif action_int == 2:
            return [2, -1]  # has to be fixed in a call of ._get_fixed_a() !!!
        else:
            raise ValueError(action_int)

    def _adjust_raise(self, raise_total_amount_in_chips):
        b = self.BIG_BET if self.current_round >= self.ROUND_WHERE_BIG_BET_STARTS else self.SMALL_BET
        return (self.n_raises_this_round + 1) * b

    def get_legal_actions(self):
        """

        Returns:
            list:   a list with none, one or multiple actions of the set [FOLD, CHECK/CALL, BET]
        """
        legal_actions = []
        for a_int in [Poker.FOLD, Poker.CHECK_CALL]:
            _a = self._get_env_adjusted_action_formulation(action_int=a_int)
            if self._get_fixed_action(action=_a)[0] == a_int:
                legal_actions.append(a_int)

        adj_a = self._get_env_adjusted_action_formulation(action_int=Poker.BET_RAISE)
        fixed_a = self._get_fixed_action(action=adj_a)
        if ((self.n_raises_this_round < self.MAX_N_RAISES_PER_ROUND[self.current_round])
            and (adj_a[0] == fixed_a[0])):
            legal_actions.append(Poker.BET_RAISE)

        return legal_actions

    def get_hand_rank(self, hand_2d, board_2d):
        raise NotImplementedError

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        This feature can allows batch computing.
        """
        raise NotImplementedError

    def get_random_action(self):
        legal = self.get_legal_actions()
        return legal[np.random.randint(len(legal))]

    def print_tutorial(self):
        print("____________________________________________ TUTORIAL ____________________________________________")
        print("Actions:")
        print("0 \tFold")
        print("1 \tCall")
        for i in range(2, self.N_ACTIONS):
            print(i, "\tRaise according to current fixed limit")

    def human_api_ask_action(self):
        """ Returns action in Tuple form. """
        while True:
            try:
                action_idx = int(
                    input("What action do you want to take as player " + str(self.current_player.seat_id) + "?"))
            except ValueError:
                print("You need to enter one of the allowed actions. Refer to the tutorial please.")
                continue
            if action_idx not in [Poker.FOLD, Poker.CHECK_CALL, Poker.BET_RAISE]:
                print("Invalid action_idx! Please enter one of the allowed actions as described in the tutorial above.")
                continue
            break
        return self._get_fixed_action(self._get_env_adjusted_action_formulation(action_int=action_idx))
