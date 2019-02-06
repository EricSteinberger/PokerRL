# Copyright (c) 2019 Eric Steinberger
# Inspiration of architecture from DeepStack-Leduc (https://github.com/lifrordi/DeepStack-Leduc/tree/master/Source)

import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
from PokerRL.game._.tree._.nodes import PlayerActionNode


class ValueFiller:

    def __init__(self, tree):
        self._tree = tree
        self._env_bldr = tree.env_bldr
        self._env = self._env_bldr.get_new_env(is_evaluating=True)

        # This only works for 1-Card games!
        self._eq_const = (self._env_bldr.rules.N_CARDS_IN_DECK / (self._env_bldr.rules.N_CARDS_IN_DECK - 1))

    def compute_cf_values_heads_up(self, node):
        """
        The functionality is extremely simplified compared to n-agent evaluations and made for HU Leduc only!
        Furthermore, this BR implementation is *VERY* inefficient and not suitable for anything much bigger than Leduc.
        """

        assert self._tree.n_seats == 2

        if node.is_terminal:
            assert node.strategy is None
        else:
            assert node.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.children),)

        if node.is_terminal:
            """
            equity: -1*reach=always lose. 1*reach=always win. 0=50%/50%
            """
            assert isinstance(node, PlayerActionNode)

            # Fold
            if node.action == Poker.FOLD:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_fold_eq_final_street(node=node)
                else:
                    equity = self._get_fold_eq_preflop(node=node)

            # Check / Call
            else:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_call_eq_final_street(reach_probs=node.reach_probs,
                                                            board_2d=node.env_state[EnvDictIdxs.board_2d])

                else:  # preflop
                    equity = self._get_call_eq_preflop(node=node)

            # set boardcards to 0
            for c in self._env_bldr.lut_holder.get_1d_cards(node.env_state[EnvDictIdxs.board_2d]):
                if c != Poker.CARD_NOT_DEALT_TOKEN_1D:
                    equity[:, c] = 0.0

            node.ev = equity * node.env_state[EnvDictIdxs.main_pot] / 2
            node.ev_br = np.copy(node.ev)

        else:
            N_ACTIONS = len(node.children)
            ev_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                      dtype=np.float32)
            ev_br_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                         dtype=np.float32)

            for i, child in enumerate(node.children):
                self.compute_cf_values_heads_up(node=child)
                ev_all_actions[i] = child.ev
                ev_br_all_actions[i] = child.ev_br

            if node.p_id_acting_next == self._tree.CHANCE_ID:
                node.ev = np.sum(ev_all_actions, axis=0)
                node.ev_br = np.sum(ev_br_all_actions, axis=0)

            else:
                node.ev = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)
                node.ev_br = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

                plyr = node.p_id_acting_next
                opp = 1 - node.p_id_acting_next

                node.ev[plyr] = np.sum(node.strategy.T * ev_all_actions[:, plyr], axis=0)
                node.ev[opp] = np.sum(ev_all_actions[:, opp], axis=0)

                node.ev_br[opp] = np.sum(ev_br_all_actions[:, opp], axis=0)
                node.ev_br[plyr] = np.max(ev_br_all_actions[:, plyr], axis=0)

                node.br_a_idx_in_child_arr_for_each_hand = np.argmax(ev_br_all_actions[:, plyr], axis=0)

        # weight ev by reach prob
        node.ev_weighted = node.ev * node.reach_probs
        node.ev_br_weighted = node.ev_br * node.reach_probs
        assert np.allclose(np.sum(node.ev_weighted), 0, atol=0.001), np.sum(node.ev_weighted)  # Zero Sum check

        node.epsilon = node.ev_br_weighted - node.ev_weighted
        node.exploitability = np.sum(node.epsilon, axis=1)

    def _get_fold_eq_preflop(self, node):
        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

        for p in range(self._tree.n_seats):
            opp = 1 - p

            # sum reach probs for all hands and subtracts the reach prob of the hand player p holds batched for all
            equity[p] = np.sum(node.reach_probs[opp]) - node.reach_probs[opp]

        equity[node.p_id_acted_last] *= -1
        return equity * self._eq_const

    def _get_fold_eq_final_street(self, node):
        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

        for p in range(self._tree.n_seats):
            opp = 1 - p

            # sum reach probs for all hands and subtracts the reach prob of the hand player p holds batched for all
            equity[p] = np.sum(node.reach_probs[opp]) - node.reach_probs[opp]

        equity[node.p_id_acted_last] *= -1
        return equity * self._eq_const

    def _get_call_eq_final_street(self, reach_probs, board_2d):
        """
        Returns:
            equity: negative=lose. positive=win. 0=50%/50%

        """
        c = self._env_bldr.lut_holder.get_1d_cards(board_2d)[0]

        assert c != Poker.CARD_NOT_DEALT_TOKEN_1D

        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)
        handranks = np.empty(shape=self._env_bldr.rules.RANGE_SIZE, dtype=np.int32)

        for h in range(self._env_bldr.rules.RANGE_SIZE):
            handranks[h] = self._env.get_hand_rank(
                board_2d=board_2d,
                hand_2d=self._env_bldr.lut_holder.get_2d_hole_cards_from_range_idx(range_idx=h))

        for p in range(self._tree.n_seats):
            opp = 1 - p
            for h in range(self._env_bldr.rules.RANGE_SIZE):
                if h != c:
                    for h_opp in range(self._env_bldr.rules.RANGE_SIZE):
                        if h_opp != h and h_opp != c:
                            # when same handrank, would be += 0
                            if handranks[h] > handranks[h_opp]:
                                equity[p, h] += reach_probs[opp, h_opp]
                            elif handranks[h] < handranks[h_opp]:
                                equity[p, h] -= reach_probs[opp, h_opp]

        assert np.allclose(equity[:, c], 0)
        return equity * self._eq_const

    def _get_call_eq_preflop(self, node):
        # very Leduc specific
        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

        for c in range(self._env_bldr.rules.N_CARDS_IN_DECK):
            """ ._get_call_eq() returns 0 for blocked hands, so we are summing 5 hands for each board. """
            _board_1d = np.array([c], dtype=np.int8)
            _board_2d = self._env_bldr.lut_holder.get_2d_cards(_board_1d)
            _reach_probs = np.copy(node.reach_probs)
            _reach_probs[:, c] = 0

            equity += self._get_call_eq_final_street(reach_probs=_reach_probs, board_2d=_board_2d)

        # mean :: using (N_CARDS_IN_DECK - 2) because two boards are blocked (by agent's and opp's cards)
        equity /= (self._env_bldr.rules.N_CARDS_IN_DECK - 2)
        return equity  # * self._eq_const
