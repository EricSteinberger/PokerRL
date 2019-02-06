# Copyright (c) 2019 Eric Steinberger


from PokerRL.game._.rl_env.base.PokerEnv import PokerEnv as _PokerEnv

from PokerRL.game.poker_env_args import NoLimitPokerEnvArgs


class NoLimitPokerEnv(_PokerEnv):
    """
    Since the PokerEnv base-class is written to be no-limit by default, this wrapper just passes straight through; it
    only exists for consistent style.
    """
    ARGS_CLS = NoLimitPokerEnvArgs

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
        assert isinstance(env_args, NoLimitPokerEnvArgs)
        super().__init__(env_args=env_args, lut_holder=lut_holder, is_evaluating=is_evaluating)
        self.N_ACTIONS = env_args.N_ACTIONS

    def _adjust_raise(self, raise_total_amount_in_chips):
        return max(self._get_current_total_min_raise(), raise_total_amount_in_chips)
