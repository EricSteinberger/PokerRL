# Copyright (c) 2019 Eric Steinberger


import unittest
from unittest import TestCase

import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game._.look_up_table import LutHolderHoldem, _LutGetterHoldem, _LutGetterLeduc
from PokerRL.game.games import StandardLeduc, DiscretizedNLHoldem


class TestLutGetterHoldem(TestCase):

    def test_get_1d_card_2_2d_card_lut(self):
        lg = _LutGetterHoldem(env_cls=DiscretizedNLHoldem)
        lut = lg.get_1d_card_2_2d_card_LUT()
        assert lut.shape == (52, 2)
        assert not np.any(lut == -2)

    def test_get_2d_card_2_1d_card_lut(self):
        lg = _LutGetterHoldem(env_cls=DiscretizedNLHoldem)
        lut = lg.get_2d_card_2_1d_card_LUT()
        assert lut.shape == (13, 4)
        assert not np.any(lut == -2)

    def test_get_idx_2_hole_card_lut(self):
        lg = _LutGetterHoldem(env_cls=DiscretizedNLHoldem)
        lut = lg.get_idx_2_hole_card_LUT()
        assert lut.shape == (1326, 2)
        assert not np.any(lut == -2)

    def test_get_hole_card_2_idx_lut(self):
        lg = _LutGetterHoldem(env_cls=DiscretizedNLHoldem)
        lut = lg.get_hole_card_2_idx_LUT()
        assert lut.shape == (52, 52)
        for i in range(52):
            for i2 in range(i + 1, 52):
                assert lut[i, i2] != -2
            for _i2 in range(0, i):
                assert lut[i, _i2] == -2

    def test_get_lut_card_in_what_range_idxs(self):
        lg = _LutGetterHoldem(env_cls=DiscretizedNLHoldem)
        lut = lg.get_card_in_what_range_idxs_LUT()
        assert lut.shape == (52, 51)
        assert not np.any(lut == -2)

        counts = np.zeros(1326, np.int32)

        for c in range(52):
            for h in lut[c]:
                counts[h] += 1

        assert np.all(counts == 2)


class TestLutGetterLeduc(TestCase):

    def test_get_1d_card_2_2d_card_lut(self):
        lg = _LutGetterLeduc(env_cls=StandardLeduc)
        lut = lg.get_1d_card_2_2d_card_LUT()
        assert lut.shape == (6, 2)
        assert not np.any(lut == -2)

    def test_get_2d_card_2_1d_card_lut(self):
        lg = _LutGetterLeduc(env_cls=StandardLeduc)
        lut = lg.get_2d_card_2_1d_card_LUT()
        assert lut.shape == (3, 2)
        assert not np.any(lut == -2)

    def test_get_idx_2_hole_card_lut(self):
        lg = _LutGetterLeduc(env_cls=StandardLeduc)
        lut = lg.get_idx_2_hole_card_LUT()
        assert lut.shape == (6, 1)
        assert not np.any(lut == -2)

    def test_get_hole_card_2_idx_lut(self):
        lg = _LutGetterLeduc(env_cls=StandardLeduc)
        lut = lg.get_hole_card_2_idx_LUT()
        assert lut.shape == (6, 1)
        assert not np.any(lut == -2)

    def test_get_lut_card_in_what_range_idxs(self):
        lg = _LutGetterLeduc(env_cls=StandardLeduc)
        lut = lg.get_card_in_what_range_idxs_LUT()
        assert lut.shape == (6, 1)
        assert not np.any(lut == -2)

        counts = np.zeros(6, np.int32)

        for c in range(6):
            for h in lut[c]:
                counts[h] += 1

        assert np.all(counts == 1)


class TestLutHolderHoldem(TestCase):

    def test_create(self):
        lh = DiscretizedNLHoldem.get_lut_holder()

        assert lh.LUT_1DCARD_2_2DCARD.dtype == np.dtype(np.int8)
        assert lh.LUT_2DCARD_2_1DCARD.dtype == np.dtype(np.int8)
        assert lh.LUT_IDX_2_HOLE_CARDS.dtype == np.dtype(np.int8)
        assert lh.LUT_HOLE_CARDS_2_IDX.dtype == np.dtype(np.int16)

    def test_get_1d_card(self):
        lh = DiscretizedNLHoldem.get_lut_holder()
        assert lh.get_1d_card(card_2d=[0, 3]) == 3
        assert lh.get_1d_card(card_2d=[12, 3]) == 51
        assert lh.get_1d_card(card_2d=np.array([0, 0], dtype=np.int8)) == 0
        assert lh.get_1d_card(card_2d=np.array([0, 0], dtype=np.int32)) == 0
        assert lh.get_1d_card(card_2d=Poker.CARD_NOT_DEALT_TOKEN_2D) == Poker.CARD_NOT_DEALT_TOKEN_1D

    def test_get_1d_cards(self):
        lh = DiscretizedNLHoldem.get_lut_holder()
        assert np.array_equal(lh.get_1d_cards(cards_2d=np.array([[0, 3]])), np.array([3]))
        assert np.array_equal(lh.get_1d_cards(cards_2d=np.array([[0, 3], [12, 3]])), np.array([3, 51]))
        assert np.array_equal(lh.get_1d_cards(cards_2d=np.array([])), np.array([], np.int8))
        assert np.array_equal(lh.get_1d_cards(
            cards_2d=np.concatenate((np.array([[0, 0]]), Poker.CARD_NOT_DEALT_TOKEN_2D.reshape(-1, 2)))),
            np.array([0, Poker.CARD_NOT_DEALT_TOKEN_1D], dtype=np.int8))

    def test_get_2d_cards(self):
        lh = DiscretizedNLHoldem.get_lut_holder()
        assert np.array_equal(lh.get_2d_cards(cards_1d=np.array([])), np.array([]))
        assert np.array_equal(lh.get_2d_cards(cards_1d=np.array([3])), np.array([[0, 3]]))
        assert np.array_equal(lh.get_2d_cards(cards_1d=np.array([3, 51])), np.array([[0, 3], [12, 3]]))
        assert np.array_equal(lh.get_2d_cards(cards_1d=np.array([])), np.array([]))
        assert np.array_equal(lh.get_2d_cards(cards_1d=np.array([0, Poker.CARD_NOT_DEALT_TOKEN_1D], dtype=np.int8)),
                              np.concatenate((np.array([[0, 0]]), Poker.CARD_NOT_DEALT_TOKEN_2D.reshape(-1, 2))))

    def test_get_range_idx_from_hole_cards(self):
        lh = DiscretizedNLHoldem.get_lut_holder()
        n = 0
        for c1 in range(51):
            for c2 in range(c1 + 1, 52):
                assert lh.LUT_HOLE_CARDS_2_IDX[c1, c2] == n

                n += 1

    def test_hole_card_luts(self):
        """ tests reversibility """
        lh = DiscretizedNLHoldem.get_lut_holder()
        for h in range(1326):
            _c_1d = lh.get_1d_hole_cards_from_range_idx(h)  # Tests 1d conversion
            _c_2d = lh.get_2d_hole_cards_from_range_idx(h)  # Tests 2d conversion

            c_1d = (lh.LUT_IDX_2_HOLE_CARDS[h])
            c_2d = np.array([lh.LUT_1DCARD_2_2DCARD[c_1d[0]],
                             lh.LUT_1DCARD_2_2DCARD[c_1d[1]]],
                            dtype=np.int8)

            assert np.array_equal(c_1d, _c_1d)
            assert np.array_equal(c_2d, _c_2d)

        # Tests inverse and thus validates that the mapping is unique for both 1d and 2d implicitly
        for c1 in range(51):
            for c2 in range(c1 + 1, 50):
                cc1 = lh.LUT_1DCARD_2_2DCARD[c1]
                cc2 = lh.LUT_1DCARD_2_2DCARD[c2]
                hole_cards = np.array([cc1, cc2], dtype=np.int8)
                assert lh.get_range_idx_from_hole_cards(hole_cards) == \
                       lh.LUT_HOLE_CARDS_2_IDX[c1, c2]


if __name__ == '__main__':
    unittest.main()
