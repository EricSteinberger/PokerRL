# Copyright (c) 2019 Eric Steinberger


import unittest
from unittest import TestCase

import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game.PokerRange import PokerRange
from PokerRL.game.games import DiscretizedNLHoldem, DiscretizedNLLeduc
from PokerRL.game.wrappers import VanillaEnvBuilder


class TestGeneralStaticFunctions(TestCase):

    def test_get_range_size(self):
        assert PokerRange.get_range_size(n_hole_cards=2, n_cards_in_deck=52) == 1326
        assert PokerRange.get_range_size(n_hole_cards=4, n_cards_in_deck=52) == 270725
        assert PokerRange.get_range_size(n_hole_cards=1, n_cards_in_deck=52) == 52
        assert PokerRange.get_range_size(n_hole_cards=1, n_cards_in_deck=6) == 6
        assert PokerRange.get_range_size(n_hole_cards=1, n_cards_in_deck=6) == 6
        assert PokerRange.get_range_size(n_hole_cards=3, n_cards_in_deck=6) == 20


class TestRangeInLeduc(TestCase):

    def test_create(self):
        env_bldr = get_leduc_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)

        assert np.allclose(np.sum(range_._range.reshape(-1, env_bldr.rules.RANGE_SIZE), axis=1), 1, atol=0.0001)

    def test_save_load(self):
        env_bldr = get_leduc_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)
        range_.load_state_dict(range_.state_dict())

    def test_normalize(self):
        env_bldr = get_leduc_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)

        range_._range = np.random.random(size=env_bldr.rules.RANGE_SIZE)
        range_.normalize()
        np.testing.assert_allclose(np.sum(range_._range), 1, atol=0.0001)

    def test_normalize_all_zero(self):
        env_bldr = get_leduc_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)

        range_._range = np.zeros_like(range_._range)
        range_.normalize()
        np.testing.assert_allclose(np.sum(range_._range), 1, atol=0.0001)

    def test_get_new_blockers_1d_leduc(self):
        env_bldr = get_leduc_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)

        full_board = np.array([[2, 1]], dtype=np.int8)

        should_be = {
            Poker.PREFLOP: env_bldr.lut_holder.get_1d_cards(full_board[:0]),
            Poker.FLOP: env_bldr.lut_holder.get_1d_cards(full_board),
        }

        for _round in [Poker.FLOP]:
            _n = env_bldr.lut_holder.DICT_LUT_N_CARDS_OUT[Poker.FLOP] - env_bldr.lut_holder.DICT_LUT_N_CARDS_OUT[_round]

            if _round == Poker.FLOP:
                board_2d = np.copy(full_board)
            else:
                board_2d = np.concatenate((full_board[:env_bldr.lut_holder.DICT_LUT_N_CARDS_OUT[_round]],
                                           np.array([Poker.CARD_NOT_DEALT_TOKEN_2D for _ in range(_n)],
                                                    dtype=np.int8)))

            result = range_._get_new_blockers_1d(game_round=_round, board_2d=board_2d)
            assert np.array_equal(a1=result, a2=should_be[_round])

    def test_remove_cards_from_raw_range_leduc(self):
        leduc_cards = np.array([[1, 1]], dtype=np.int8)

        env_bldr = get_leduc_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)

        range_.set_cards_to_zero_prob(cards_2d=leduc_cards)

        _ra = range_._range.reshape(-1, env_bldr.rules.RANGE_SIZE)
        for i in range(_ra.shape[0]):
            np.testing.assert_allclose(np.sum(_ra[i]), 1, atol=0.00001)

        _assert_cards_not_in_ranges(cards_2d=leduc_cards, ranges=_ra, rules=env_bldr.rules,
                                    lut_holder=env_bldr.lut_holder)

    def test_get_card_probs_leduc(self):
        env_bldr = get_leduc_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)

        assert np.array_equal(range_.get_card_probs(), range_._range)

    def test_get_possible_range_idxs_leduc(self):

        for n in range(2, 9):
            env_bldr = get_leduc_env_bldr()

            # if actually blocked
            for c in range(env_bldr.rules.N_CARDS_IN_DECK):
                board_2d = env_bldr.lut_holder.get_2d_cards(np.array([c], dtype=np.int32))
                result = PokerRange.get_possible_range_idxs(rules=env_bldr.rules, lut_holder=env_bldr.lut_holder,
                                                            board_2d=board_2d)

                should_be = np.delete(np.arange(env_bldr.rules.RANGE_SIZE, dtype=np.int32), c)

                assert np.array_equal(a1=result, a2=should_be)

            # if nothing blocked
            board_2d = np.array([Poker.CARD_NOT_DEALT_TOKEN_2D], dtype=np.int8)
            result = PokerRange.get_possible_range_idxs(rules=env_bldr.rules, lut_holder=env_bldr.lut_holder,
                                                        board_2d=board_2d)

            should_be = np.arange(env_bldr.rules.RANGE_SIZE, dtype=np.int32)

            assert np.array_equal(a1=result, a2=should_be)


class TestRangeInHoldem(TestCase):

    def test_create(self):
        env_bldr = get_holdem_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)

        assert np.allclose(np.sum(range_._range.reshape(-1, env_bldr.rules.RANGE_SIZE), axis=1), 1, atol=0.0001)

    def test_save_load(self):
        env_bldr = get_holdem_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)
        range_.load_state_dict(range_.state_dict())

    def test_get_new_blockers_1d_holdem(self):
        env_bldr = get_holdem_env_bldr()

        range_ = PokerRange(env_bldr=env_bldr)

        full_board = np.array([[1, 2], [3, 3], [12, 1], [5, 2], [6, 0]], dtype=np.int8)

        should_be = {
            Poker.PREFLOP: env_bldr.lut_holder.get_1d_cards(full_board[:0]),
            Poker.FLOP: env_bldr.lut_holder.get_1d_cards(full_board[0:3]),
            Poker.TURN: env_bldr.lut_holder.get_1d_cards(full_board[3:4]),
            Poker.RIVER: env_bldr.lut_holder.get_1d_cards(full_board[4:5]),
        }

        for _round in [Poker.PREFLOP, Poker.FLOP, Poker.TURN, Poker.RIVER]:
            _n = env_bldr.lut_holder.DICT_LUT_N_CARDS_OUT[Poker.RIVER] - \
                 env_bldr.lut_holder.DICT_LUT_N_CARDS_OUT[_round]

            if _round == Poker.RIVER:
                board_2d = np.copy(full_board)
            else:
                board_2d = np.concatenate(
                    (full_board[:env_bldr.lut_holder.DICT_LUT_N_BOARD_BRANCHES[_round]],
                     np.array([Poker.CARD_NOT_DEALT_TOKEN_2D for _ in range(_n)], dtype=np.int8))
                )

            result = range_._get_new_blockers_1d(game_round=_round, board_2d=board_2d)
            assert np.array_equal(a1=result, a2=should_be[_round])

    def test_remove_cards_from_raw_range_holdem(self):
        env_bldr = get_holdem_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)

        holdem_cards = np.array([[7, 2], [6, 0]], dtype=np.int8)

        range_.set_cards_to_zero_prob(cards_2d=holdem_cards)

        _ra = range_._range.reshape(-1, env_bldr.rules.RANGE_SIZE)
        for i in range(_ra.shape[0]):
            np.testing.assert_allclose(np.sum(_ra[i]), 1, atol=0.00001)
        _assert_cards_not_in_ranges(cards_2d=holdem_cards, ranges=_ra, rules=env_bldr.rules,
                                    lut_holder=env_bldr.lut_holder)

    def test_get_card_probs_holdem(self):
        env_bldr = get_holdem_env_bldr()
        range_ = PokerRange(env_bldr=env_bldr)

        cards_to_remove = np.array([0, 3, 6, 33, 21, 51], np.int8)

        # use previously tested method to make this test easier
        range_.set_cards_to_zero_prob(cards_2d=env_bldr.lut_holder.get_2d_cards(cards_to_remove))

        r = range_.get_card_probs()
        assert np.allclose(np.sum(r), 2, atol=0.00001)
        for c in cards_to_remove:
            assert np.allclose(r[c], 0, atol=0.00001)

    def test_get_possible_range_idxs_holdem(self):
        env_bldr = get_holdem_env_bldr()
        for n in range(2, 9):
            board_2d = np.array([[0, 0], [5, 2], [12, 3], Poker.CARD_NOT_DEALT_TOKEN_2D], dtype=np.int8)
            result = PokerRange.get_possible_range_idxs(rules=env_bldr.rules, lut_holder=env_bldr.lut_holder,
                                                        board_2d=board_2d)

            assert result.shape[0] == 1176  # nck of 49:2

            # all of these should be blocked
            for e in [0, 1, 2, 3, 4, 50, 1325]:
                assert not np.any(result == e)


def _assert_cards_not_in_ranges(cards_2d, ranges, rules, lut_holder):
    if rules.N_HOLE_CARDS == 1:  # leduc
        for p in range(ranges.shape[0]):
            for c in cards_2d:
                np.testing.assert_allclose(
                    ranges[p, lut_holder.get_range_idx_from_hole_cards(hole_cards_2d=c.reshape(-1, 2))], 0,
                    atol=0.0001)

    elif rules.N_HOLE_CARDS == 2:  # holdem
        cards_1d = lut_holder.get_1d_cards(cards_2d=cards_2d.reshape(-1, 2))
        for p in range(ranges.shape[0]):
            for c in cards_1d:
                for c1 in range(0, c):
                    np.testing.assert_allclose(ranges[p, lut_holder.LUT_HOLE_CARDS_2_IDX[c1, c]], 0,
                                               atol=0.00001)
                for c2 in range(c + 1, rules.N_CARDS_IN_DECK):
                    np.testing.assert_allclose(ranges[p, lut_holder.LUT_HOLE_CARDS_2_IDX[c, c2]], 0,
                                               atol=0.00001)
    else:
        raise NotImplementedError


def get_holdem_env_bldr():
    return VanillaEnvBuilder(env_cls=DiscretizedNLHoldem,
                             env_args=DiscretizedNLHoldem.ARGS_CLS(n_seats=3,
                                                                   starting_stack_sizes_list=[234] * 3,
                                                                   bet_sizes_list_as_frac_of_pot=[0.1, 0.4, 1.0],
                                                                   stack_randomization_range=(-123, 234)))


def get_leduc_env_bldr():
    return VanillaEnvBuilder(env_cls=DiscretizedNLLeduc,
                             env_args=DiscretizedNLLeduc.ARGS_CLS(n_seats=3,
                                                                  starting_stack_sizes_list=[234] * 3,
                                                                  bet_sizes_list_as_frac_of_pot=[0.1, 0.4, 1.0],
                                                                  stack_randomization_range=(-123, 234)))


if __name__ == '__main__':
    unittest.main()
