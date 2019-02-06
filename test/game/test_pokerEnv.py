# Copyright (c) 2019 Eric Steinberger


import copy
import unittest
from unittest import TestCase

import numpy as np

from PokerRL.game.games import NoLimitHoldem


class TestPokerEnv(TestCase):
    ITERATIONS = 30
    MIN_P = 2
    MAX_P = 6
    DO_RENDER = False

    def test_sync_deck(self):
        args_1 = NoLimitHoldem.ARGS_CLS(n_seats=3,
                                        stack_randomization_range=(0, 0),
                                        starting_stack_sizes_list=[20000] * 3)
        args_2 = NoLimitHoldem.ARGS_CLS(n_seats=3,
                                        stack_randomization_range=(-12, 34),
                                        starting_stack_sizes_list=[123] * 3)
        env1 = NoLimitHoldem(is_evaluating=False, env_args=args_1, lut_holder=NoLimitHoldem.get_lut_holder())
        env2 = NoLimitHoldem(is_evaluating=False, env_args=args_2, lut_holder=NoLimitHoldem.get_lut_holder())

        env1.load_cards_state_dict(cards_state_dict=env2.cards_state_dict())

        for n in range(env2.N_CARDS_IN_DECK - env2.N_HOLE_CARDS * env2.N_SEATS):
            assert np.array_equal(env1.deck.draw(1), env2.deck.draw(1))

        env1.reset()
        env2.reset()
        env1.load_cards_state_dict(cards_state_dict=env2.cards_state_dict())

        for n in range(int((env2.N_CARDS_IN_DECK - (env2.N_HOLE_CARDS * env2.N_SEATS)) / 2)):
            assert np.array_equal(env1.deck.draw(2), env2.deck.draw(2))

    def test_get_current_obs(self):
        args = NoLimitHoldem.ARGS_CLS(n_seats=3,
                                      stack_randomization_range=(0, 0),
                                      starting_stack_sizes_list=[1000] * 3)
        env = NoLimitHoldem(is_evaluating=False, env_args=args, lut_holder=NoLimitHoldem.get_lut_holder())
        env.reset()
        env.step([1, -1])
        a = env.get_current_obs(is_terminal=False)
        b = env.get_current_obs(is_terminal=False)
        assert np.array_equal(a, b)

        # terminal should be all 0
        assert np.array_equal(np.zeros_like(a), env.get_current_obs(is_terminal=True))

    def test_seat_id(self):
        for n in range(TestPokerEnv.MIN_P, TestPokerEnv.MAX_P + 1):
            env = _get_new_nlh_env(n)
            for p_idx in range(len(env.seats)):
                assert env.seats[p_idx].seat_id == p_idx

    def test_chip_consistency_all_stacks_equal(self):
        for n in range(TestPokerEnv.MIN_P, TestPokerEnv.MAX_P + 1):

            for _ in range(TestPokerEnv.ITERATIONS):

                env = _get_new_nlh_env(n)
                original_stack_sum = sum([p.stack for p in env.seats])

                terminal = False
                env.reset()
                if TestPokerEnv.DO_RENDER:
                    env.render()

                while not terminal:
                    obs, reward, terminal, info = env.step(action=env.get_random_action())

                    if TestPokerEnv.DO_RENDER:
                        env.render()

                current_stack_sum = sum([p.stack for p in env.seats])
                assert current_stack_sum == original_stack_sum

    def test_chip_consistency_random_stacks(self):
        for n in range(TestPokerEnv.MIN_P, TestPokerEnv.MAX_P + 1):

            for _ in range(TestPokerEnv.ITERATIONS):

                env = _get_new_nlh_env(n, 100, 1000, True)
                original_stack_sum = sum([p.stack for p in env.seats])

                terminal = False
                env.reset()
                if TestPokerEnv.DO_RENDER:
                    env.render()

                while not terminal:
                    obs, reward, terminal, info = env.step(action=env.get_random_action())
                    if TestPokerEnv.DO_RENDER:
                        env.render()

                current_stack_sum = sum([p.stack for p in env.seats])
                assert current_stack_sum == original_stack_sum

    def test_reward_is_0_for_not_terminal(self):
        for n in range(TestPokerEnv.MIN_P, TestPokerEnv.MAX_P + 1):
            env = _get_new_nlh_env(n, 100, 1000)
            for _ in range(TestPokerEnv.ITERATIONS):

                terminal = False
                env.reset()
                if TestPokerEnv.DO_RENDER:
                    env.render()
                while not terminal:
                    obs, reward, terminal, info = env.step(action=env.get_random_action())
                    if TestPokerEnv.DO_RENDER:
                        env.render()
                    if not terminal:
                        assert np.array_equal(reward, np.zeros(shape=env.N_SEATS, dtype=np.int32))

                env.reset()

    def test_action_space_sample(self):
        env = _get_new_nlh_env(6)
        env.reset()
        action = env.get_random_action()
        assert (isinstance(action, tuple) and len(action) == 2)

    def test_get_and_set_env(self):
        for n in range(TestPokerEnv.MIN_P, TestPokerEnv.MAX_P + 1):
            env = _get_new_nlh_env(n, 100, 1000, True)
            env.reset()
            for _ in range(TestPokerEnv.ITERATIONS):

                # save state after some actions
                repeat_cause_terminal = True
                while repeat_cause_terminal:
                    o_obs, reward, terminal, info = env.reset()
                    i = 0
                    while not terminal and i < np.random.randint(low=0, high=n * 6):
                        o_obs, reward, terminal, info = env.step(env.get_random_action())
                        i += 1
                        repeat_cause_terminal = terminal

                saved_state = env.state_dict()

                # manipulate env to test if old state is fully restored
                i = 0
                while not terminal and i < np.random.randint(low=0, high=n * 4):
                    obs, reward, terminal, info = env.step(action=env.get_random_action())
                    i += 1

                env.load_state_dict(saved_state)
                obs = env.get_current_obs(False)
                np.array_equal(obs, o_obs)

                env.reset()

    def test_get_hole_cards_of_player(self):
        for n in range(TestPokerEnv.MIN_P, TestPokerEnv.MAX_P + 1):
            env = _get_new_nlh_env(n, 100, 1000)
            env.reset()
            for p in range(env.N_SEATS):
                h = env.get_hole_cards_of_player(p_id=p)
                _h = env.seats[p].hand
                assert np.array_equal(h, _h)

    def test_get_fraction_of_pot_raise(self):
        for n_plyrs in range(TestPokerEnv.MIN_P, TestPokerEnv.MAX_P + 1):
            n = np.random.randint(low=0, high=12)
            for f in [0.1, 0.3, 0.5, 0.7, 1.0, 1.4, 2.2]:
                for _ in range(TestPokerEnv.ITERATIONS):
                    env = _get_new_nlh_env(n_plyrs, 100, 1000)
                    env.reset()

                    for _ in range(n_plyrs):
                        env.step((1, -1))

                    last_raiser_id = env.current_player.seat_id
                    for _ in range(n):  # step a couple times to avoid only testing edge cases
                        # env.step((2, np.random.randint(low=0, high=100)))
                        last_raiser_id = env.current_player.seat_id
                        env.step((2, np.random.randint(low=0, high=180)))

                    next_raiser_id = env.current_player.seat_id

                    suggested_raise = env.get_fraction_of_pot_raise(fraction=f,
                                                                    player_that_bets=env.seats[next_raiser_id])

                    s = sum(env.side_pots) + env.main_pot + sum([p.current_bet for p in env.seats])
                    s += env.seats[last_raiser_id].current_bet - env.seats[next_raiser_id].current_bet

                    last_raiser_tocall_if_next_bets = suggested_raise - env.seats[last_raiser_id].current_bet
                    should_be = s * f

                    assert should_be - 1 <= last_raiser_tocall_if_next_bets <= should_be + 1

    def test_get_frac_from_chip_amt(self):
        for i in range(self.ITERATIONS):
            for n_plyrs in range(TestPokerEnv.MIN_P, TestPokerEnv.MAX_P + 1):
                n = np.random.randint(low=0, high=2)
                for amt in [10, 233, 412, 5001]:
                    env = _get_new_nlh_env(n_plyrs, 300, 1000)
                    env.reset()

                    for _ in range(n_plyrs):
                        env.step((1, -1))

                    for _ in range(n):  # step a couple times to avoid only testing edge cases
                        env.step((2, np.random.randint(low=0, high=255)))
                        if np.random.random() > 0.5:
                            env.step((1, -1))

                    next_raiser_id = env.current_player.seat_id

                    frac = env.get_frac_from_chip_amt(amt=amt, player_that_bets=env.seats[next_raiser_id])
                    suggested_raise = env.get_fraction_of_pot_raise(fraction=frac,
                                                                    player_that_bets=env.seats[next_raiser_id])

                    assert np.allclose(amt, suggested_raise, atol=1.1)

    def test_get_filtered_action_but_change_nothing(self):
        env = _get_new_nlh_env(7, 100, 1000)
        env.reset()

        for _ in range(TestPokerEnv.ITERATIONS):

            orig_action = (np.random.randint(low=0, high=2), np.random.randint(low=0, high=env.MAX_CHIPS))
            _orig_action = copy.deepcopy(orig_action)

            a = env._get_fixed_action(action=orig_action)

            obs, rew, terminal, _ = env.step(action=tuple(a))

            if terminal:
                env.reset()

            assert orig_action == _orig_action


def _get_new_nlh_env(n_seats, min_stack=100, max_stack=1000, random_stacks=False):
    r_m = 0
    if random_stacks:
        r_m = min_stack - max_stack
    args = NoLimitHoldem.ARGS_CLS(n_seats=n_seats,
                                  stack_randomization_range=(r_m, 0),
                                  starting_stack_sizes_list=[1000] * n_seats)
    return NoLimitHoldem(env_args=args, is_evaluating=True, lut_holder=NoLimitHoldem.get_lut_holder())


if __name__ == '__main__':
    unittest.main()
