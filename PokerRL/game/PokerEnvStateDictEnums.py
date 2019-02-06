# Copyright (c) 2019 Eric Steinberger


"""
Enum keys to PokerEnv's state_dict objects.
"""


class EnvDictIdxs:
    current_round = "current_round"
    side_pots = "side_pots"
    main_pot = "main_pot"
    board_2d = "board_2d"
    last_action = "last_action"
    capped_raise = "capped_raise"
    current_player = "current_player"
    last_raiser = "last_raiser"
    deck = "deck_remaining"
    seats = "seats"
    n_actions_this_episode = "n_actions_this_episode"
    # only relevant for Fixed-Limit games
    n_raises_this_round = "n_raises_this_round"

    is_evaluating = "is_evaluating"


class PlayerDictIdxs:
    hand = "hand"
    hand_rank = "hand_rank"
    stack = "stack"
    current_bet = "current_bet"
    is_allin = "is_allin"
    folded_this_episode = "folded_this_episode"
    has_acted_this_round = "has_acted_this_round"
    side_pot_rank = "side_pot_rank"
    seat_id = "seat_id"
