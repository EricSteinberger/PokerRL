# Copyright (c) 2019 Eric Steinberger


import numpy as np


class Poker:
    """
    Global constants
    """

    # All common rounds of a poker game. When creating subclasses with different rules, you can remove, but not add
    # rounds from the game by simply overriding ALL_ROUNDS_LIST.
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

    INT2STRING_ROUND = {
        PREFLOP: "preflop",
        FLOP: "flop",
        TURN: "turn",
        RIVER: "river",
    }

    STRING2INT_ROUND = {
        "preflop": PREFLOP,
        "flop": FLOP,
        "turn": TURN,
        "river": RIVER
    }

    # Possible actions in a poker game.
    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2

    # The public board has this value wherever a card has not been dealt. 1D and 2D refers to the two different
    # representations of cards: as an int or as a tuple of (rank, suit) respectively.
    CARD_NOT_DEALT_TOKEN_1D = -127
    CARD_NOT_DEALT_TOKEN_2D = np.array([CARD_NOT_DEALT_TOKEN_1D,
                                        CARD_NOT_DEALT_TOKEN_1D])

    # Metrics to measure winnings and exploitability.
    MeasureAnte = "MA_per_G"  # Milli Antes per game
    MeasureBB = "MBB_per_G"  # Milli Big Blinds per game
