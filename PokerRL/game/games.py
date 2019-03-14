# Copyright (c) 2019 Eric Steinberger


"""
A collection of Poker games often used in computational poker research.
"""

from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.game_rules import HoldemRules, LeducRules, FlopHoldemRules, BigLeducRules
from PokerRL.game._.rl_env.poker_types.DiscretizedPokerEnv import DiscretizedPokerEnv
from PokerRL.game._.rl_env.poker_types.LimitPokerEnv import LimitPokerEnv
from PokerRL.game._.rl_env.poker_types.NoLimitPokerEnv import NoLimitPokerEnv


# """""""""""""""
# Leduc Family
# """""""""""""""
class StandardLeduc(LeducRules, LimitPokerEnv):
    """
    Leduc Hold'em is a very small poker game meant for fast experimentation with new algorithms. It is played with 3
    ranks and 2 suits. Typically players place an ante of 1, the small_bet is 2, and the big_bet is 4.
    """

    RULES = LeducRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False
    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 2,
        Poker.FLOP: 2,
    }

    SMALL_BLIND = 0
    BIG_BLIND = 0
    ANTE = 1
    SMALL_BET = 2
    BIG_BET = 4
    DEFAULT_STACK_SIZE = 13

    EV_NORMALIZER = 1000.0 / ANTE  # Milli Antes
    WIN_METRIC = Poker.MeasureAnte

    ROUND_WHERE_BIG_BET_STARTS = Poker.FLOP

    def __init__(self, env_args, lut_holder, is_evaluating):
        LeducRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)


class BigLeduc(BigLeducRules, LimitPokerEnv):
    RULES = BigLeducRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False
    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 6,
        Poker.FLOP: 6,
    }

    SMALL_BLIND = 0
    BIG_BLIND = 0
    ANTE = 1
    SMALL_BET = 2
    BIG_BET = 4
    DEFAULT_STACK_SIZE = 100

    EV_NORMALIZER = 1000.0 / ANTE  # Milli Antes
    WIN_METRIC = Poker.MeasureAnte

    ROUND_WHERE_BIG_BET_STARTS = Poker.FLOP

    def __init__(self, env_args, lut_holder, is_evaluating):
        BigLeducRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)


class NoLimitLeduc(LeducRules, NoLimitPokerEnv):
    """
    A variant of Leduc with no bet-cap in the no-limit format. It uses blinds instead of antes.
    """

    RULES = LeducRules
    IS_FIXED_LIMIT_GAME = False
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        LeducRules.__init__(self)
        NoLimitPokerEnv.__init__(self,
                                 env_args=env_args,
                                 lut_holder=lut_holder,
                                 is_evaluating=is_evaluating)


class DiscretizedNLLeduc(LeducRules, DiscretizedPokerEnv):
    """
    Discretized version of No-Limit Leduc Hold'em (i.e. agents can only select from a predefined set of betsizes)
    """

    RULES = LeducRules
    IS_FIXED_LIMIT_GAME = False
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        LeducRules.__init__(self)
        DiscretizedPokerEnv.__init__(self,
                                     env_args=env_args,
                                     lut_holder=lut_holder,
                                     is_evaluating=is_evaluating)


# """""""""""""""
# Hold'em Family
# """""""""""""""
class LimitHoldem(HoldemRules, LimitPokerEnv):
    """
    Fixed-Limit Texas Hold'em is a long-standing benchmark game that has been essentially solved by Bowling et al
    (http://science.sciencemag.org/content/347/6218/145) using an efficient distributed implementation of CFR+, an
    optimized version of regular CFR.
    """

    RULES = HoldemRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False
    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 4,
        Poker.FLOP: 4,
        Poker.TURN: 4,
        Poker.RIVER: 4,
    }
    ROUND_WHERE_BIG_BET_STARTS = Poker.TURN

    SMALL_BLIND = 1
    BIG_BLIND = 2
    ANTE = 0
    SMALL_BET = 2
    BIG_BET = 4
    DEFAULT_STACK_SIZE = 48

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        HoldemRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)


class NoLimitHoldem(HoldemRules, NoLimitPokerEnv):
    """
    No-Limit Texas Hold'em is the largest poker game in which AI beat humans as of 31.08.2018. It has been the focus in
    work such as DeepStack (https://arxiv.org/abs/1701.01724) and Libratus
    (http://science.sciencemag.org/content/early/2017/12/15/science.aao1733).
    """

    RULES = HoldemRules
    IS_FIXED_LIMIT_GAME = False
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        HoldemRules.__init__(self)
        NoLimitPokerEnv.__init__(self,
                                 env_args=env_args,
                                 lut_holder=lut_holder,
                                 is_evaluating=is_evaluating)


class DiscretizedNLHoldem(HoldemRules, DiscretizedPokerEnv):
    """
    Discretized version of No-Limit Texas Hold'em (i.e. agents can only select from a predefined set of betsizes)
    """

    RULES = HoldemRules
    IS_FIXED_LIMIT_GAME = False
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        HoldemRules.__init__(self)
        DiscretizedPokerEnv.__init__(self,
                                     env_args=env_args,
                                     lut_holder=lut_holder,
                                     is_evaluating=is_evaluating)


class Flop5Holdem(FlopHoldemRules, LimitPokerEnv):
    RULES = FlopHoldemRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 2,  # is actually 1, but BB counts as a raise in this codebase
        Poker.FLOP: 2,
    }
    ROUND_WHERE_BIG_BET_STARTS = Poker.TURN

    UNITS_SMALL_BET = None
    UNITS_BIG_BET = None

    FIRST_ACTION_NO_CALL = True

    def __init__(self, env_args, lut_holder, is_evaluating):
        FlopHoldemRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)

    def _adjust_raise(self, raise_total_amount_in_chips):
        return self.get_fraction_of_pot_raise(fraction=1.0, player_that_bets=self.current_player)


"""
register all new envs here!
"""
ALL_ENVS = [
    StandardLeduc,
    BigLeduc,
    NoLimitLeduc,
    DiscretizedNLLeduc,
    LimitHoldem,
    NoLimitHoldem,
    DiscretizedNLHoldem,
    Flop5Holdem,
]
