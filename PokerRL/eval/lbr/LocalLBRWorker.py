# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch

from PokerRL.eval.lbr import _util
from PokerRL.game.Poker import Poker
from PokerRL.game.PokerRange import PokerRange


class LocalLBRWorker:
    """
    Slave to EvalLBRMaster. Does the LBR computation as described in https://arxiv.org/abs/1612.07547
    """

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        assert t_prof.n_seats == 2

        self.t_prof = t_prof
        self.lbr_args = t_prof.module_args["lbr"]
        self._eval_env_bldr = _util.get_env_builder_lbr(t_prof=t_prof)
        self.check_to_round = self.lbr_args.lbr_check_to_round

        self.chief_handle = chief_handle

        self.agent = _AgentWrapper(t_prof=t_prof, lbr_args=self.lbr_args, eval_agent_cls=eval_agent_cls)

        # has different raise sizes than agent's env! This needs to be considered when updating the envs after opp acts
        self._env = None
        self.agent_range = PokerRange(env_bldr=self._eval_env_bldr)

        assert self.check_to_round is None or (self.check_to_round in self._eval_env_bldr.rules.ALL_ROUNDS_LIST)

    def run(self, agent_seat_id, n_iterations, mode, stack_size):
        """ returns an estimate of a lower bound of the exploitablity of the agent """

        self.agent.set_mode(mode=mode)
        self.agent.to_stack_size(stack_size)
        self.agent_range.reset()

        self._env = self._eval_env_bldr.get_new_env(is_evaluating=True, stack_size=stack_size)

        if not self.agent.can_compute_mode():
            return None

        if self._eval_env_bldr.env_cls.IS_FIXED_LIMIT_GAME:
            return self._run_limit(agent_seat_id=agent_seat_id, n_iterations=n_iterations)
        else:
            return self._run_no_limit(agent_seat_id=agent_seat_id, n_iterations=n_iterations)

    def update_weights(self, weights_for_eval_agent):
        self.agent.update_weights(weights_for_eval_agent)

    def _reset_episode(self):
        ret = self._env.reset()
        self.agent.reset(deck_state_dict=self._env.cards_state_dict())
        self.agent_range.reset()
        return ret

    def _run_limit(self, agent_seat_id, n_iterations):
        total_lbr_winnings = np.empty(shape=n_iterations, dtype=np.float32)
        lbr_seat_id = 1 - agent_seat_id

        for iteration_id in range(n_iterations):
            if iteration_id % 50 == 0:
                print("LBR hand: ", iteration_id)

            # """""""""""""""""
            # Reset
            # """""""""""""""""
            env_obs, reward, terminal, info = self._reset_episode()

            lbr_hand = self._env.get_hole_cards_of_player(p_id=lbr_seat_id)
            self.agent_range.set_cards_to_zero_prob(cards_2d=lbr_hand)

            # """""""""""""""""
            # Play Episode
            # """""""""""""""""
            while not terminal:
                p_id_acting = self._env.current_player.seat_id

                if self.t_prof.DEBUGGING:
                    assert p_id_acting == self.agent.cpu_agent._internal_env_wrapper.env.current_player.seat_id

                if p_id_acting == lbr_seat_id:
                    # optional feature: check the first N rounds 100% as LBR
                    if (self.check_to_round is not None) and (self._env.current_round < self.check_to_round):
                        action_int = Poker.CHECK_CALL

                    else:
                        _rollout_mngr = _LBRRolloutManager(t_prof=self.t_prof, env_bldr=self._eval_env_bldr,
                                                           env=self._env, lbr_hand_2d=lbr_hand)

                        # illegal: -1, fold: 0, all other: any float
                        _utility = np.full(shape=3, fill_value=-1.0, dtype=np.float32)

                        # ev(s, lbr_a=fold)
                        _utility[Poker.FOLD] = 0.0

                        # ev(s, lbr_a=check_call)
                        _wp = _rollout_mngr.get_lbr_checkdown_equity(
                            agent_range=self.agent_range)  # if check/called down
                        _asked = self._env.seats[agent_seat_id].current_bet - self._env.seats[lbr_seat_id].current_bet
                        _pot_before_action = self._env.get_all_winnable_money()
                        _utility[Poker.CHECK_CALL] = _wp * _pot_before_action - (1 - _wp) * _asked

                        # prepare for raise simulation
                        if Poker.BET_RAISE in self._env.get_legal_actions():
                            _saved_env_state = self._env.state_dict()
                            _saved_agent_env_state = self.agent.env_state_dict()
                            _saved_agent_range_state = self.agent_range.state_dict()

                            # compute ev for raise
                            # _________________________________ simulate LBR play r ____________________________________
                            self._env.step(action=Poker.BET_RAISE)
                            _pot_after_raise = self._env.get_all_winnable_money()

                            self.agent.notify_of_action(p_id_acted=lbr_seat_id, action_he_did=Poker.BET_RAISE)

                            # what agent would do after LBR raises. DOESN'T STEP INTERNAL ENV!
                            _, a_probs_each_hand = self.agent.get_action(step_env=False, need_probs=True)

                            # _______________________________ simulate agent reaction __________________________________
                            # p(agent_fold)
                            _fold_prob = np.sum(self.agent_range.range * a_probs_each_hand[:, Poker.FOLD])

                            # p(not agent_fold | hand)
                            _p_not_fold_per_hand = (1 - a_probs_each_hand[:, Poker.FOLD])

                            # agent_range after not folding
                            self.agent_range.mul_and_norm(_p_not_fold_per_hand)

                            # p(lbr_win | lbr play r -> agent play not fold)
                            _wp_now = _rollout_mngr.get_lbr_checkdown_equity(agent_range=self.agent_range)

                            # ev(state, lbr_a=r)
                            _chips_lbr_puts_in_pot = _pot_after_raise - _pot_before_action
                            _ev_if_fold = _pot_before_action
                            _ev_if_not_fold = (_wp_now * _pot_after_raise) - ((1 - _wp_now) * _chips_lbr_puts_in_pot)
                            _utility[Poker.BET_RAISE] = _fold_prob * _ev_if_fold + (1 - _fold_prob) * _ev_if_not_fold

                            # ________________________________________ reset ___________________________________________
                            self.agent_range.load_state_dict(_saved_agent_range_state)
                            self._env.load_state_dict(_saved_env_state)
                            self.agent.load_env_state_dict(_saved_agent_env_state)

                        # select action with highest approximated EV
                        action_int = np.argmax(_utility)

                    # ________________________________________ notify agent ____________________________________________
                    self.agent.notify_of_action(p_id_acted=lbr_seat_id, action_he_did=action_int)

                else:  # agent has to act
                    action_int, a_probs_each_hand = self.agent.get_action(step_env=True, need_probs=True)
                    self.agent_range.update_after_action(action=action_int,
                                                         all_a_probs_for_all_hands=a_probs_each_hand)

                # _____________________________________________ step ___________________________________________________
                old_game_round = self._env.current_round

                env_obs, reward, terminal, info = self._env.step(action=action_int)

                if self._env.current_round != old_game_round:
                    self.agent_range.update_after_new_round(new_round=self._env.current_round,
                                                            board_now_2d=self._env.board)

            total_lbr_winnings[iteration_id] = reward[lbr_seat_id] * self._env.REWARD_SCALAR * self._env.EV_NORMALIZER

        return total_lbr_winnings

    def _run_no_limit(self, agent_seat_id, n_iterations):
        total_lbr_winnings = np.empty(shape=n_iterations, dtype=np.float32)
        lbr_seat_id = 1 - agent_seat_id
        n_lbr_bets = len(self._env.bet_sizes_list_as_frac_of_pot)

        for iteration_id in range(n_iterations):
            if iteration_id % 50 == 0:
                print("LBR hand: ", iteration_id)

            # """""""""""""""""
            # Reset
            # """""""""""""""""
            env_obs, reward, done, info = self._reset_episode()

            lbr_hand = self._env.get_hole_cards_of_player(p_id=lbr_seat_id)
            self.agent_range.set_cards_to_zero_prob(cards_2d=lbr_hand)

            # """""""""""""""""
            # Play Episode
            # """""""""""""""""
            while not done:
                p_id_acting = self._env.current_player.seat_id

                if self.t_prof.DEBUGGING:
                    assert p_id_acting == self.agent.cpu_agent._internal_env_wrapper.env.current_player.seat_id

                if p_id_acting == lbr_seat_id:

                    # optional feature: check the first N rounds 100% as LBR
                    if (self.check_to_round is not None) and (self._env.current_round < self.check_to_round):
                        action_int = Poker.CHECK_CALL

                    else:
                        _rollout_mngr = _LBRRolloutManager(t_prof=self.t_prof, env_bldr=self._eval_env_bldr,
                                                           env=self._env, lbr_hand_2d=lbr_hand)

                        # illegal: -1, fold: 0, all other: any float
                        _utility = np.full(shape=2 + n_lbr_bets, fill_value=-1.0, dtype=np.float32)

                        # ev(s, lbr_a=fold)
                        _utility[Poker.FOLD] = 0.0

                        # ev(s, lbr_a=check_call)
                        _wp = _rollout_mngr.get_lbr_checkdown_equity(agent_range=self.agent_range)
                        _asked = self._env.seats[agent_seat_id].current_bet - self._env.seats[lbr_seat_id].current_bet
                        _pot_before_action = self._env.get_all_winnable_money()
                        _utility[Poker.CHECK_CALL] = _wp * _pot_before_action - (1 - _wp) * _asked

                        # prepare for raise simulation
                        _saved_env_state = self._env.state_dict()
                        _saved_agent_env_state = self.agent.env_state_dict()
                        _saved_agent_range_state = self.agent_range.state_dict()
                        _legal_raises = self._env.get_legal_actions()
                        for a in [Poker.FOLD, Poker.CHECK_CALL]:
                            if a in _legal_raises:
                                _legal_raises.remove(a)

                        # compute ev for all raise sizes LBR can choose from
                        for r in _legal_raises:
                            raise_frac = self._env.bet_sizes_list_as_frac_of_pot[r - 2]

                            # _________________________________ simulate LBR play r ____________________________________
                            self._env.step(action=r)
                            _pot_after_raise = self._env.get_all_winnable_money()

                            self.agent.notify_of_raise_frac_action(p_id_acted=lbr_seat_id, frac=raise_frac)

                            if self.t_prof.DEBUGGING:
                                assert agent_seat_id == self.agent.cpu_agent._internal_env_wrapper.env.current_player.seat_id

                            # what agent would do after LBR raises. DOESN'T STEP INTERNAL ENV!
                            a_probs_each_hand = self.agent.get_a_probs_for_each_hand()

                            # _______________________________ simulate agent reaction __________________________________
                            # p(agent_fold)
                            _fold_prob = np.sum(self.agent_range.range * a_probs_each_hand[:, Poker.FOLD])

                            # p(not agent_fold | hand)
                            _p_not_fold_per_hand = (1 - a_probs_each_hand[:, Poker.FOLD])

                            # agent_range after not folding
                            self.agent_range.mul_and_norm(_p_not_fold_per_hand)

                            # p(lbr_win | lbr play r -> agent play not fold)
                            _wp_now = _rollout_mngr.get_lbr_checkdown_equity(agent_range=self.agent_range)

                            # ev(state, lbr_a=r)
                            _chips_lbr_puts_in_pot = _pot_after_raise - _pot_before_action
                            _ev_if_fold = _pot_before_action
                            _ev_if_not_fold = (_wp_now * _pot_after_raise) - ((1 - _wp_now) * _chips_lbr_puts_in_pot)
                            _utility[r] = _fold_prob * _ev_if_fold + (1 - _fold_prob) * _ev_if_not_fold

                            # ________________________________________ reset ___________________________________________
                            self.agent_range.load_state_dict(_saved_agent_range_state)
                            self._env.load_state_dict(_saved_env_state)
                            self.agent.load_env_state_dict(_saved_agent_env_state)

                        # select action with highest approximated EV
                        action_int = np.argmax(_utility)

                    # ________________________________________ notify agent ____________________________________________
                    if action_int >= 2:
                        raise_frac = self._env.bet_sizes_list_as_frac_of_pot[action_int - 2]
                        self.agent.notify_of_raise_frac_action(p_id_acted=lbr_seat_id,
                                                               frac=raise_frac)
                    else:
                        self.agent.notify_of_action(p_id_acted=lbr_seat_id,
                                                    action_he_did=action_int)

                else:  # agent has to act
                    if self.t_prof.DEBUGGING:
                        assert p_id_acting == self.agent.cpu_agent._internal_env_wrapper.env.current_player.seat_id

                    action_int, a_probs_each_hand = self.agent.get_action(step_env=True, need_probs=True)

                    self.agent_range.update_after_action(action=action_int,
                                                         all_a_probs_for_all_hands=a_probs_each_hand)
                    if action_int >= 2:
                        # querying what the bet size is in the agent's env_args (these might differ from LBR's!).
                        raise_frac = \
                            self.agent.cpu_agent.env_bldr.env_args.bet_sizes_list_as_frac_of_pot[action_int - 2]

                # _____________________________________________ step ___________________________________________________
                old_game_round = self._env.current_round

                if action_int >= 2:  # step with fraction because agent and LBR have different raise sizes
                    env_obs, reward, done, info = self._env.step_raise_pot_frac(pot_frac=raise_frac)
                else:
                    env_obs, reward, done, info = self._env.step(action=action_int)

                if self._env.current_round != old_game_round:
                    self.agent_range.update_after_new_round(new_round=self._env.current_round,
                                                            board_now_2d=self._env.board)

            total_lbr_winnings[iteration_id] = reward[lbr_seat_id] * self._env.REWARD_SCALAR * self._env.EV_NORMALIZER

        return total_lbr_winnings


class _AgentWrapper:

    def __init__(self, t_prof, lbr_args, eval_agent_cls):
        self.USE_GPU = (t_prof.HAVE_GPU and lbr_args.use_gpu_for_batch_eval and torch.cuda.is_available())

        self.cpu_agent = eval_agent_cls(t_prof=t_prof, device=torch.device("cpu"))

        if self.USE_GPU:
            self.gpu_agent = eval_agent_cls(t_prof=t_prof, device=torch.device("cuda:0"))
            self.gpu_agent.set_env_wrapper(self.cpu_agent.get_env_wrapper())

    # ________________________________ intensive tasks on GPU. Single samples on CPU ___________________________________
    def get_action(self, step_env, need_probs):
        if need_probs and self.USE_GPU:  # if need_probs is True, strategy for all hands is computed --> batch
            return self.gpu_agent.get_action(step_env=step_env, need_probs=need_probs)
        else:
            return self.cpu_agent.get_action(step_env=step_env, need_probs=need_probs)

    def get_a_probs_for_each_hand(self):
        if self.USE_GPU:
            return self.gpu_agent.get_a_probs_for_each_hand()
        else:
            return self.cpu_agent.get_a_probs_for_each_hand()

    # _________________________________________________ just wrapped ___________________________________________________
    def get_mode(self):
        if self.USE_GPU:
            return self.gpu_agent.get_mode()
        return self.cpu_agent.get_mode()

    def set_mode(self, mode):
        self.cpu_agent.set_mode(mode)
        if self.USE_GPU:
            self.gpu_agent.set_mode(mode)

    def to_stack_size(self, stack_size):
        # Automatically sets gpu agent, if applicable, because they share an environment.
        self.cpu_agent.set_stack_size(stack_size=stack_size)

    def can_compute_mode(self):
        return self.cpu_agent.can_compute_mode()

    def update_weights(self, w):
        self.cpu_agent.update_weights(w)
        if self.USE_GPU:
            self.gpu_agent.update_weights(w)

    def reset(self, deck_state_dict):
        self.cpu_agent.reset(deck_state_dict=deck_state_dict)
        if self.USE_GPU:
            self.gpu_agent.reset(deck_state_dict=deck_state_dict)

    def notify_of_action(self, p_id_acted, action_he_did):
        # GPU uses same env. must not do on both!
        self.cpu_agent.notify_of_action(p_id_acted=p_id_acted, action_he_did=action_he_did)

    def notify_of_raise_frac_action(self, p_id_acted, frac):
        self.cpu_agent.notify_of_raise_frac_action(p_id_acted=p_id_acted, frac=frac)

    def env_state_dict(self):
        return self.cpu_agent.env_state_dict()

    def load_env_state_dict(self, state_dict):
        self.cpu_agent.load_env_state_dict(state_dict)


class _LBRRolloutManager:

    def __init__(self, t_prof, env_bldr, env, lbr_hand_2d):
        self.t_prof = t_prof
        self.env_bldr = env_bldr

        self._bigger_idxs = []
        self._equal_idxs = []

        self._env = env

        self._lbr_hand_1d = self.env_bldr.lut_holder.get_1d_cards(cards_2d=lbr_hand_2d)
        self._lbr_hand_range_idx = self.env_bldr.lut_holder.get_range_idx_from_hole_cards(hole_cards_2d=lbr_hand_2d)

        self._board_2d = np.copy(self._env.board)  # still has not-dealt cards
        self._board_1d = self.env_bldr.lut_holder.get_1d_cards(self._board_2d)  # still has not-dealt cards

        self._cards_dealt = np.array([c for c in self._board_1d if c != Poker.CARD_NOT_DEALT_TOKEN_1D])
        self._possible_cards = np.arange(self.env_bldr.rules.N_CARDS_IN_DECK, dtype=np.int32)
        self._possible_cards = np.delete(self._possible_cards, np.concatenate((self._cards_dealt, self._lbr_hand_1d)))

        self._n_cards_to_deal = env_bldr.lut_holder.DICT_LUT_N_CARDS_OUT[self._env.ALL_ROUNDS_LIST[-1]] \
                                - env_bldr.lut_holder.DICT_LUT_N_CARDS_OUT[self._env.current_round]

        # recursion
        self._build_eq_vecs(board_1d=np.copy(self._board_1d),
                            n_cards_to_deal=self._n_cards_to_deal,
                            possible_cards_1d=np.copy(self._possible_cards))

    def _build_eq_vecs(self, board_1d, n_cards_to_deal, possible_cards_1d):
        if n_cards_to_deal > 0:
            """
            only considers cards in possible_cards_1d (sorted arr of ints). For the next street all cards (1d)
            smaller than the one dealt now are NOT CONSIDERED. This way we guarantee to deal each board only once.
            """
            for c in range(possible_cards_1d.shape[0] - (n_cards_to_deal - 1)):
                _possible_cards_1d_next = possible_cards_1d[c + 1:]
                board_1d[-n_cards_to_deal] = possible_cards_1d[c]
                self._build_eq_vecs(board_1d=board_1d,
                                    n_cards_to_deal=n_cards_to_deal - 1,
                                    possible_cards_1d=_possible_cards_1d_next)

        else:  # if we get here, the last round was already dealt. now evaluate.
            handranks = self._env.get_hand_rank_all_hands_on_given_boards(
                boards_1d=board_1d.reshape(1, board_1d.shape[0]), lut_holder=self.env_bldr.lut_holder)[0]

            lbr_hand_rank = handranks[self._lbr_hand_range_idx]
            self._bigger_idxs.append(np.argwhere(handranks < lbr_hand_rank))
            self._equal_idxs.append(np.argwhere(handranks == lbr_hand_rank))

    def get_lbr_checkdown_equity(self, agent_range):
        agent_range_start_sate_dict = agent_range.state_dict()

        # adjust card_probs based on agent's range
        agent_card_probs = agent_range.get_card_probs()
        card_probs = np.subtract(1, agent_card_probs)
        card_probs[self._lbr_hand_1d] = 0.0

        if self.t_prof.DEBUGGING:
            assert np.all(np.less(agent_card_probs[self._lbr_hand_1d], 0.0001))
            assert np.allclose(np.sum(agent_card_probs), self.env_bldr.rules.N_HOLE_CARDS, atol=0.0001)
        cards_dealt = np.array([c for c in self._board_1d if c != Poker.CARD_NOT_DEALT_TOKEN_1D])
        if cards_dealt.shape[0] > 0:
            if self.t_prof.DEBUGGING:
                assert np.all(np.less(agent_card_probs[cards_dealt], 0.00001))
            card_probs[cards_dealt] = 0.0

        # This can be division by 0 in some games, but if it is, it implicitly doesn't matter.
        _s = np.sum(card_probs)
        if _s > 0:
            card_probs /= np.sum(card_probs)

        # recursion
        lbr_win_prob = [0.0]
        self._calc_eq(_win_prob=lbr_win_prob,
                      _i=[0],
                      _board_1d=np.copy(self._board_1d),
                      _n_cards_to_deal=self._n_cards_to_deal,
                      _card_probs=card_probs,
                      _possible_cards_1d=np.copy(self._possible_cards),
                      _reach_prob=1.0,
                      _agent_range=agent_range,
                      _agent_range_start_sate_dict=agent_range_start_sate_dict)

        # winprob has to be multiplied by n_c_to_deal_factorial because we only check one permutation - not all.
        n_c_to_deal_factorial = 1
        for m in range(1, self._n_cards_to_deal + 1):
            n_c_to_deal_factorial *= m

        return lbr_win_prob[0] * n_c_to_deal_factorial

    def _calc_eq(self,
                 _win_prob,
                 _i,
                 _board_1d,
                 _n_cards_to_deal,
                 _card_probs,
                 _possible_cards_1d,
                 _reach_prob,
                 _agent_range,
                 _agent_range_start_sate_dict,
                 ):

        if _n_cards_to_deal > 0:
            """
            only considers cards in possible_cards_1d (sorted arr of ints). For the next street all cards (1d)
            smaller than the one dealt now are NOT CONSIDERED. This way we guarantee to deal each board only once.
            """
            for c in range(_possible_cards_1d.shape[0] - (_n_cards_to_deal - 1)):
                _possible_cards_1d_next = _possible_cards_1d[c + 1:]
                _board_1d[-_n_cards_to_deal] = _possible_cards_1d[c]
                _card_probs_next = np.copy(_card_probs)
                _card_probs_next[_possible_cards_1d[c]] = 0.0
                _card_probs_next /= np.sum(_card_probs_next)
                self._calc_eq(_win_prob=_win_prob,
                              _i=_i,
                              _board_1d=_board_1d,
                              _n_cards_to_deal=_n_cards_to_deal - 1,
                              _possible_cards_1d=_possible_cards_1d_next,
                              _card_probs=_card_probs_next,
                              _reach_prob=_reach_prob * _card_probs[_possible_cards_1d[c]],
                              _agent_range=_agent_range,
                              _agent_range_start_sate_dict=_agent_range_start_sate_dict)

        else:  # if we get here, the last round was already dealt. now evaluate.
            _board_2d = self.env_bldr.lut_holder.get_2d_cards(cards_1d=_board_1d)
            _agent_range.set_cards_to_zero_prob(cards_2d=_board_2d)

            if self.t_prof.DEBUGGING:
                assert not np.any(self._env.CARD_NOT_DEALT_TOKEN_1D == _board_1d)

            lbr_equity = np.sum(_agent_range.range[self._bigger_idxs[_i[0]]])
            lbr_equity += np.sum(_agent_range.range[self._equal_idxs[_i[0]]]) / 2.0

            _win_prob[0] += lbr_equity * _reach_prob
            _agent_range.load_state_dict(_agent_range_start_sate_dict)
