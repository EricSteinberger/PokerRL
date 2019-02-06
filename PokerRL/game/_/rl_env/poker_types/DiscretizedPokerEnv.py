# Copyright (c) 2019 Eric Steinberger


import time

import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.base.PokerEnv import PokerEnv as _PokerEnv
from PokerRL.game.poker_env_args import DiscretizedPokerEnvArgs


class DiscretizedPokerEnv(_PokerEnv):
    """
    To discretize No-Limit or Pot-Limit poker games, subclass this baseclass instaed of PokerEnv. It allows to define
    a set of bet_sizes (as fractions of the pot) that are then part of the action space. Contrary to the action format
    of PokerEnv tuple(action, raise_size), discretized envs have integer actions, where 0 is FOLD, 1 is CHECK/CALL and
    then come all specified raise sizes sorted ascending.
    """
    ARGS_CLS = DiscretizedPokerEnvArgs

    def __init__(self,
                 env_args,
                 lut_holder,
                 is_evaluating):

        """
        Args:
            env_args (DiscretePokerEnvArgs):    an instance of DiscretePokerEnvArgs, passing an instance of PokerEnvArgs
                                                will not work.
            is_evaluating (bool):               Whether the environment shall be spawned in evaluation mode
                                                (i.e. no randomization) or not.

        """
        assert isinstance(env_args, DiscretizedPokerEnvArgs)
        assert isinstance(env_args.bet_sizes_list_as_frac_of_pot, list)
        assert isinstance(env_args.uniform_action_interpolation, bool)
        super().__init__(env_args=env_args, lut_holder=lut_holder, is_evaluating=is_evaluating)

        self.bet_sizes_list_as_frac_of_pot = sorted(env_args.bet_sizes_list_as_frac_of_pot)  # ascending
        self.N_ACTIONS = env_args.N_ACTIONS
        self.uniform_action_interpolation = env_args.uniform_action_interpolation

    def _adjust_raise(self, raise_total_amount_in_chips):
        return max(self._get_current_total_min_raise(), raise_total_amount_in_chips)

    def _get_env_adjusted_action_formulation(self, action_int):
        """

        Args:
            action_int: integer representation of discretized action

        Returns:
            list: (action, raise_size) in "continuous" PokerEnv format

        """
        if action_int == 0:
            return [0, -1]
        if action_int == 1:
            return [1, -1]
        elif action_int > 1:
            selected = self.get_fraction_of_pot_raise(fraction=self.bet_sizes_list_as_frac_of_pot[action_int - 2],
                                                      player_that_bets=self.current_player)

            if self.uniform_action_interpolation and not self.IS_EVALUATING:
                # _________________________________________ The maximal amount _________________________________________
                if action_int == self.N_ACTIONS - 1:  # if highest bet in repertoire
                    if self.IS_POT_LIMIT_GAME:  # if PL game: max is pot
                        max_amnt = self.get_fraction_of_pot_raise(fraction=1.0, player_that_bets=self.current_player)
                    elif self.IS_FIXED_LIMIT_GAME:
                        raise EnvironmentError("Should not get here with a limit game!")
                    else:  # if NL game: max is allin
                        max_amnt = self.current_player.stack + self.current_player.current_bet

                else:  # else, max is the mean of the selected and the next-bigger bet size
                    bigger = self.get_fraction_of_pot_raise(fraction=self.bet_sizes_list_as_frac_of_pot[action_int - 1],
                                                            player_that_bets=self.current_player)
                    max_amnt = int(float(selected + bigger) / 2)

                # _________________________________________ The minimal amount _________________________________________
                if action_int == 2:  # if lowest bet in repertoire, min is minbet
                    min_amnt = self._get_current_total_min_raise()

                else:  # else, min is the mean of the selected and the next-smaller bet size
                    smaller = self.get_fraction_of_pot_raise(
                        fraction=self.bet_sizes_list_as_frac_of_pot[action_int - 3],
                        player_that_bets=self.current_player)
                    min_amnt = int(float(selected + smaller) / 2)

                if min_amnt >= max_amnt:  # can happen. The sampling would always be the same -> save time.
                    return [2, min_amnt]
                return [2, np.random.randint(low=min_amnt, high=max_amnt)]

            else:
                return [2, selected]
        else:
            raise ValueError(action_int)

    def get_legal_actions(self):
        """

        Returns:
            list:   a list with none, one or multiple actions of the set [FOLD, CHECK/CALL, BETSIZE_1, BETSIZE_2, ...]

        """
        legal_actions = []

        # Fold, Check/Call
        for a_int in [Poker.FOLD, Poker.CHECK_CALL]:
            _a = self._get_env_adjusted_action_formulation(action_int=a_int)
            if self._get_fixed_action(action=_a)[0] == a_int:
                legal_actions.append(a_int)

        # since raises are ascending in the list, we can simply loop and break
        _last_too_small = None
        for a in range(2, self.N_ACTIONS):  # only loops through raises
            adj_a = self._get_env_adjusted_action_formulation(action_int=a)
            fixed_a = self._get_fixed_action(action=adj_a)

            if adj_a[0] != fixed_a[0]:  # if we wanted to raise, but env told us we can not, dont append a raise
                break

            if (adj_a[1] < fixed_a[1]) and a < self.N_ACTIONS:
                _last_too_small = a
            else:
                if _last_too_small is not None:
                    legal_actions.append(_last_too_small)  # was to small, but rounded to minraise & is unique -> append
                    _last_too_small = None

                legal_actions.append(a)  # this action might be modified by env, but is legal and unique -> append

            if adj_a[1] > fixed_a[1]:  # if the raise was too big, an even bigger one will yield the same result
                break
        assert len(legal_actions) > 0
        return legal_actions

    def get_hand_rank(self, hand_2d, board_2d):
        """
        For docs, refer to PokerEnv.get_hand_rank_all_hands_on_given_boards()
        """
        raise NotImplementedError

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        For docs, refer to PokerEnv.get_hand_rank_all_hands_on_given_boards()
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
            print(i, "\tRaise ", self.bet_sizes_list_as_frac_of_pot[i - 2] * 100, "% of the pot")

    def human_api_ask_action(self):
        """ Returns action in Tuple form. """
        while True:
            try:
                action_idx = int(
                    input("What action do you want to take as player " + str(self.current_player.seat_id) + "?"))
            except ValueError:
                print("You need to enter one of the allowed actions. Refer to the tutorial please.")
                continue
            if action_idx < Poker.FOLD or action_idx >= self.N_ACTIONS:
                print("Invalid action_idx! Please enter one of the allowed actions as described in the tutorial above.")
                continue
            break
        time.sleep(0.01)

        return self._get_fixed_action(self._get_env_adjusted_action_formulation(action_int=action_idx))
