# Copyright (c) 2019 Eric Steinberger


import os
from os.path import join as ospj

import numpy as np

from PokerRL._.CppWrapper import CppWrapper
from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.game_rules import HoldemRules


class CppLibHoldemLuts(CppWrapper):

    def __init__(self, n_boards_lut, n_cards_out_lut):
        super().__init__(path_to_dll=ospj(os.path.dirname(os.path.realpath(__file__)),
                                          "lib_luts." + self.CPP_LIB_FILE_ENDING))
        self._n_boards_lut = n_boards_lut
        self._n_cards_out_lut = n_cards_out_lut

        self._clib.get_hole_card_2_idx_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_hole_card_2_idx_lut.restype = None

        self._clib.get_idx_2_hole_card_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_idx_2_hole_card_lut.restype = None

        self._clib.get_idx_2_flop_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_idx_2_flop_lut.restype = None

        self._clib.get_idx_2_turn_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_idx_2_turn_lut.restype = None

        self._clib.get_idx_2_river_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_idx_2_river_lut.restype = None

    # __________________________________________________ LUTs __________________________________________________________
    def get_idx_2_hole_card_lut(self):
        lut = np.full(shape=(HoldemRules.RANGE_SIZE, 2), fill_value=-2, dtype=np.int8)
        self._clib.get_idx_2_hole_card_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_hole_card_2_idx_lut(self):
        lut = np.full(shape=(HoldemRules.N_CARDS_IN_DECK, HoldemRules.N_CARDS_IN_DECK),
                      fill_value=-2, dtype=np.int16)
        self._clib.get_hole_card_2_idx_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_idx_2_flop_lut(self):
        lut = np.full(shape=(
            self._n_boards_lut[Poker.FLOP],
            self._n_cards_out_lut[Poker.FLOP]),
            fill_value=-2, dtype=np.int8)
        self._clib.get_idx_2_flop_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_idx_2_turn_lut(self):
        lut = np.full(shape=(
            self._n_boards_lut[Poker.TURN],
            self._n_cards_out_lut[Poker.TURN]),
            fill_value=-2, dtype=np.int8)
        self._clib.get_idx_2_turn_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_idx_2_river_lut(self):
        lut = np.full(shape=(
            self._n_boards_lut[Poker.RIVER],
            self._n_cards_out_lut[Poker.RIVER]),
            fill_value=-2, dtype=np.int8)
        self._clib.get_idx_2_river_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_1d_card(self, card_2d):
        """
        Args:
            card_2d (np.ndarray):    array of 2 int8s. [rank, suit]

        Returns:
            int8: 1d representation of card_2d

        """
        return self._clib.get_1d_card(self.np_1d_arr_to_c(card_2d))

    def get_2d_card(self, card_1d):
        """
        Args:
            card_1d (int): 

        Returns:
            np.ndarray(shape=2, dtype=np.int8): 2d representation of card_1d
        """
        card_2d = np.empty(shape=2, dtype=np.int8)
        self._clib.get_2d_card(card_1d, self.np_1d_arr_to_c(card_2d))
        return card_2d
