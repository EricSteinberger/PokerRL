# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game.PokerRange import PokerRange

"""
classes in this file are utilities that poker environments based on PokerEnv can inherit from. They override PokerEnv's
class variables and thereby set it to a certain rule set.
"""


class LeducRules:
    """
    General rules of Leduc Hold'em poker games
    """

    N_HOLE_CARDS = 1
    N_RANKS = 3
    N_SUITS = 2
    N_CARDS_IN_DECK = N_RANKS * N_SUITS
    RANGE_SIZE = PokerRange.get_range_size(n_hole_cards=N_HOLE_CARDS, n_cards_in_deck=N_CARDS_IN_DECK)

    BTN_IS_FIRST_POSTFLOP = True

    N_FLOP_CARDS = 1
    N_TURN_CARDS = 0
    N_RIVER_CARDS = 0
    N_TOTAL_BOARD_CARDS = N_FLOP_CARDS + N_TURN_CARDS + N_RIVER_CARDS
    ALL_ROUNDS_LIST = [Poker.PREFLOP, Poker.FLOP]

    SUITS_MATTER = False

    ROUND_BEFORE = {
        Poker.PREFLOP: Poker.PREFLOP,
        Poker.FLOP: Poker.PREFLOP
    }
    ROUND_AFTER = {
        Poker.PREFLOP: Poker.FLOP,
        Poker.FLOP: None
    }

    RANK_DICT = {i: str(i + 2) for i in range(N_RANKS)}
    SUIT_DICT = {k: ["a", "b", "c", "d", "e", "f", "g"][k] for k in range(N_SUITS)} \
        if N_SUITS < 8 \
        else {i: str(i) for i in range(N_SUITS)}

    STRING = "LEDUC_RULES"

    def __init__(self):
        pass

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        for general docs refer to PokerEnv
        """
        hand_ranks = np.full(shape=(boards_1d.shape[0], LeducRules.RANGE_SIZE), fill_value=-1, dtype=np.int32)
        for board_idx in range(boards_1d.shape[0]):
            for range_idx in range(LeducRules.RANGE_SIZE):
                hand_ranks[board_idx, range_idx] = self.get_hand_rank(
                    hand_2d=lut_holder.get_2d_hole_cards_from_range_idx(range_idx=range_idx),
                    board_2d=lut_holder.get_2d_cards(cards_1d=boards_1d[board_idx]))

        return hand_ranks

    def get_hand_rank(self, hand_2d, board_2d):
        """
        for docs refer to PokerEnv
        """
        if board_2d[0, 0] == hand_2d[0, 0]:
            return 100 + hand_2d[0, 0]
        else:
            return hand_2d[0, 0]

    @classmethod
    def get_lut_holder(cls):
        from PokerRL.game._.look_up_table import LutHolderLeduc

        return LutHolderLeduc(cls)


class BigLeducRules:
    N_HOLE_CARDS = 1
    N_RANKS = 12
    N_SUITS = 2
    N_CARDS_IN_DECK = N_RANKS * N_SUITS
    RANGE_SIZE = PokerRange.get_range_size(n_hole_cards=N_HOLE_CARDS, n_cards_in_deck=N_CARDS_IN_DECK)

    BTN_IS_FIRST_POSTFLOP = True

    N_FLOP_CARDS = 1
    N_TURN_CARDS = 0
    N_RIVER_CARDS = 0
    N_TOTAL_BOARD_CARDS = N_FLOP_CARDS + N_TURN_CARDS + N_RIVER_CARDS
    ALL_ROUNDS_LIST = [Poker.PREFLOP, Poker.FLOP]

    SUITS_MATTER = False

    ROUND_BEFORE = {
        Poker.PREFLOP: Poker.PREFLOP,
        Poker.FLOP: Poker.PREFLOP
    }
    ROUND_AFTER = {
        Poker.PREFLOP: Poker.FLOP,
        Poker.FLOP: None
    }

    RANK_DICT = {i: str(i + 2) for i in range(N_RANKS)}
    SUIT_DICT = {k: ["a", "b", "c", "d", "e", "f", "g"][k] for k in range(N_SUITS)} \
        if N_SUITS < 8 \
        else {i: "_" + str(i) for i in range(N_SUITS)}

    STRING = "BIG_LEDUC_RULES"

    def __init__(self):
        pass

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        for general docs refer to PokerEnv
        """
        hand_ranks = np.full(shape=(boards_1d.shape[0], LeducRules.RANGE_SIZE), fill_value=-1, dtype=np.int32)
        for board_idx in range(boards_1d.shape[0]):
            for range_idx in range(LeducRules.RANGE_SIZE):
                hand_ranks[board_idx, range_idx] = self.get_hand_rank(
                    hand_2d=lut_holder.get_2d_hole_cards_from_range_idx(range_idx=range_idx),
                    board_2d=lut_holder.get_2d_cards(cards_1d=boards_1d[board_idx]))

        return hand_ranks

    def get_hand_rank(self, hand_2d, board_2d):
        """
        for docs refer to PokerEnv
        """
        if board_2d[0, 0] == hand_2d[0, 0]:
            return 10000 + hand_2d[0, 0]
        else:
            return hand_2d[0, 0]

    @classmethod
    def get_lut_holder(cls):
        from PokerRL.game._.look_up_table import LutHolderLeduc

        return LutHolderLeduc(cls)


class HoldemRules:
    """
    General rules of Texas Hold'em poker games
    """
    N_HOLE_CARDS = 2
    N_RANKS = 13
    N_SUITS = 4
    N_CARDS_IN_DECK = N_RANKS * N_SUITS
    RANGE_SIZE = PokerRange.get_range_size(n_hole_cards=N_HOLE_CARDS, n_cards_in_deck=N_CARDS_IN_DECK)

    BTN_IS_FIRST_POSTFLOP = False

    N_FLOP_CARDS = 3
    N_TURN_CARDS = 1
    N_RIVER_CARDS = 1
    N_TOTAL_BOARD_CARDS = N_FLOP_CARDS + N_TURN_CARDS + N_RIVER_CARDS
    ALL_ROUNDS_LIST = [Poker.PREFLOP, Poker.FLOP, Poker.TURN, Poker.RIVER]

    SUITS_MATTER = True

    ROUND_BEFORE = {
        Poker.PREFLOP: Poker.PREFLOP,
        Poker.FLOP: Poker.PREFLOP,
        Poker.TURN: Poker.FLOP,
        Poker.RIVER: Poker.TURN
    }
    ROUND_AFTER = {
        Poker.PREFLOP: Poker.FLOP,
        Poker.FLOP: Poker.TURN,
        Poker.TURN: Poker.RIVER,
        Poker.RIVER: None
    }

    RANK_DICT = {
        Poker.CARD_NOT_DEALT_TOKEN_1D: "",
        0: "2",
        1: "3",
        2: "4",
        3: "5",
        4: "6",
        5: "7",
        6: "8",
        7: "9",
        8: "T",
        9: "J",
        10: "Q",
        11: "K",
        12: "A"
    }
    SUIT_DICT = {
        Poker.CARD_NOT_DEALT_TOKEN_1D: "",
        0: "h",
        1: "d",
        2: "s",
        3: "c"
    }

    STRING = "HOLDEM_RULES"

    def __init__(self):
        from PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval

        self._clib = CppHandeval()

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        for docs refer to PokerEnv
        """
        return self._clib.get_hand_rank_all_hands_on_given_boards_52_holdem(boards_1d=boards_1d, lut_holder=lut_holder)

    def get_hand_rank(self, hand_2d, board_2d):
        """
        for docs refer to PokerEnv
        """
        return self._clib.get_hand_rank_52_holdem(hand_2d=hand_2d, board_2d=board_2d)

    @classmethod
    def get_lut_holder(cls):
        from PokerRL.game._.look_up_table import LutHolderHoldem

        return LutHolderHoldem(cls)


class FlopHoldemRules:
    """
    General rules of Texas Hold'em poker games
    """
    N_HOLE_CARDS = 2
    N_RANKS = 13
    N_SUITS = 4
    N_CARDS_IN_DECK = N_RANKS * N_SUITS
    RANGE_SIZE = PokerRange.get_range_size(n_hole_cards=N_HOLE_CARDS, n_cards_in_deck=N_CARDS_IN_DECK)

    BTN_IS_FIRST_POSTFLOP = False

    N_FLOP_CARDS = 5
    N_TURN_CARDS = 0
    N_RIVER_CARDS = 0
    N_TOTAL_BOARD_CARDS = N_FLOP_CARDS
    ALL_ROUNDS_LIST = [Poker.PREFLOP, Poker.FLOP]

    SUITS_MATTER = True

    ROUND_BEFORE = {
        Poker.PREFLOP: Poker.PREFLOP,
        Poker.FLOP: Poker.PREFLOP,
        Poker.TURN: None,
        Poker.RIVER: None,
    }
    ROUND_AFTER = {
        Poker.PREFLOP: Poker.FLOP,
        Poker.FLOP: None,
        Poker.TURN: None,
        Poker.RIVER: None,
    }

    RANK_DICT = {
        Poker.CARD_NOT_DEALT_TOKEN_1D: "",
        0: "2",
        1: "3",
        2: "4",
        3: "5",
        4: "6",
        5: "7",
        6: "8",
        7: "9",
        8: "T",
        9: "J",
        10: "Q",
        11: "K",
        12: "A"
    }
    SUIT_DICT = {
        Poker.CARD_NOT_DEALT_TOKEN_1D: "",
        0: "h",
        1: "d",
        2: "s",
        3: "c"
    }

    STRING = "FLOP_HOLDEM_RULES"

    def __init__(self):
        from PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval

        self._clib = CppHandeval()

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        for docs refer to PokerEnv
        """
        return self._clib.get_hand_rank_all_hands_on_given_boards_52_holdem(boards_1d=boards_1d, lut_holder=lut_holder)

    def get_hand_rank(self, hand_2d, board_2d):
        """
        for docs refer to PokerEnv
        """
        return self._clib.get_hand_rank_52_holdem(hand_2d=hand_2d, board_2d=board_2d)

    @classmethod
    def get_lut_holder(cls):
        from PokerRL.game._.look_up_table import LutHolderHoldem

        return LutHolderHoldem(cls)
