# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.game.Poker import Poker


class PokerRange:
    """
    Track a distribution over hands for a player.
    """

    def __init__(self, env_bldr):
        assert env_bldr.rules.N_HOLE_CARDS <= 2

        self._env_bldr = env_bldr
        self._range = None  # np.array(range_size)
        self.reset()

    @property
    def range(self):
        return self._range

    def get_range(self):
        return np.copy(self._range)

    def get_card_probs(self):
        if self._env_bldr.rules.N_HOLE_CARDS == 1:
            return np.copy(self._range)
        elif self._env_bldr.rules.N_HOLE_CARDS == 2:
            card_probs = np.zeros(shape=self._env_bldr.rules.N_CARDS_IN_DECK, dtype=np.float32)

            for c in range(self._env_bldr.rules.N_CARDS_IN_DECK):
                card_probs[c] = np.sum(self._range[self._env_bldr.lut_holder.LUT_CARD_IN_WHAT_RANGE_IDXS[c]])
            return card_probs

        else:
            raise NotImplementedError()

    def mul_and_norm(self, mul_vector):
        self._range *= mul_vector
        self.normalize()

    def normalize(self):
        norm_factor = np.sum(self._range, axis=-1)
        if norm_factor == 0:
            self._reset_range()
        else:
            self._range = self._range / norm_factor

    def update_after_action(self, action, all_a_probs_for_all_hands):
        self._range *= all_a_probs_for_all_hands[:, action]
        self.normalize()

    def update_after_new_round(self, new_round, board_now_2d):
        """
        1) Remove the new blockers from the range by setting all hands including them to probability 0
        2) Normalize
        """
        self.set_cards_to_zero_prob(cards_2d=self._get_new_blockers_2d(game_round=new_round, board_2d=board_now_2d))

    def reset(self):
        """
        Before any cards are dealt ranges are uniformly random distributions
        """
        self._reset_range()

    def set_cards_to_zero_prob(self, cards_2d):
        cards_1d_to_remove = self._env_bldr.lut_holder.get_1d_cards(cards_2d=cards_2d)
        if self._env_bldr.rules.N_HOLE_CARDS == 1:
            self._range[cards_1d_to_remove] = 0

        elif self._env_bldr.rules.N_HOLE_CARDS == 2:
            for c in cards_1d_to_remove:
                # instead of looping we make use of the fact that the LUT is sorted and just use numpy slicing
                self._range[self._env_bldr.lut_holder.LUT_HOLE_CARDS_2_IDX[0:c, c]] = 0
                self._range[
                    self._env_bldr.lut_holder.LUT_HOLE_CARDS_2_IDX[c, c + 1:self._env_bldr.rules.N_CARDS_IN_DECK]] = 0
        else:
            raise NotImplementedError("We don't currently support games with >2 hole cards")

        self.normalize()

    @staticmethod
    def get_possible_range_idxs(rules, lut_holder, board_2d):
        arr = np.arange(rules.RANGE_SIZE)

        if board_2d.shape[0] == 0:  # if nothing is blocked
            return arr

        # filter not-dealt-cards
        lut_holder.get_1d_cards(cards_2d=board_2d)
        blocked_cards_1d = np.array(
            [c for c in lut_holder.get_1d_cards(cards_2d=board_2d) if c != Poker.CARD_NOT_DEALT_TOKEN_1D])

        if rules.N_HOLE_CARDS == 1:
            arr = np.delete(arr, obj=blocked_cards_1d)
            return arr

        elif rules.N_HOLE_CARDS == 2:
            hands = []

            for c in blocked_cards_1d:
                for c1 in range(0, c):
                    hands.append(lut_holder.get_range_idx_from_hole_cards(
                        lut_holder.get_2d_cards(np.array([c1, c], dtype=np.int8))))
                for c2 in range(c + 1, rules.N_CARDS_IN_DECK):
                    hands.append(lut_holder.get_range_idx_from_hole_cards(
                        lut_holder.get_2d_cards(np.array([c, c2], dtype=np.int8))))

            blocked_idxs = np.unique(np.array(hands))
            arr = np.delete(arr, obj=blocked_idxs)

            return arr

        else:
            raise NotImplementedError("self.N_HOLE_CARDS > 2:  " + str(rules.N_HOLE_CARDS))

    @staticmethod
    def get_range_size(n_hole_cards, n_cards_in_deck):
        """
        Args:
            n_hole_cards:       number of cards each player is dealt
            n_cards_in_deck:    number of unique cards in the deck

        Returns:
            int:                the number of possible hands (order of cards does not matter) given a set number of
                                holecards and cards in the deck.
        """
        range_size = 1
        for i in range(n_hole_cards):
            range_size *= n_cards_in_deck - i

        n_hc_factorial = np.prod(np.arange(1, n_hole_cards + 1))
        return int(range_size / n_hc_factorial)

    def load_state_dict(self, state):
        self._range = np.copy(state['range'])

    def state_dict(self):
        return {
            'range': np.copy(self._range),
        }

    def _get_new_blockers_1d(self, game_round, board_2d):
        _1d = self._env_bldr.lut_holder.get_1d_cards(
            self._get_new_blockers_2d(game_round=game_round, board_2d=board_2d))
        return _1d

    def _get_new_blockers_2d(self, game_round, board_2d):
        n = self._env_bldr.lut_holder.DICT_LUT_N_CARDS_OUT[game_round]
        nm1 = self._env_bldr.lut_holder.DICT_LUT_N_CARDS_OUT[self._env_bldr.rules.ROUND_BEFORE[game_round]]
        new_blockers = board_2d[nm1:n].reshape(-1, 2)
        return new_blockers

    def _reset_range(self):
        self._range = np.full(shape=self._env_bldr.rules.RANGE_SIZE,
                              fill_value=1.0 / self._env_bldr.rules.RANGE_SIZE,
                              dtype=np.float32)
