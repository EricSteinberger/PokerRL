# Copyright (c) 2019 Eric Steinberger


import numpy as np


class AgentTournament:

    def __init__(self, env_cls, env_args, eval_agent_1, eval_agent_2):
        self._eval_agents = [eval_agent_1, eval_agent_2]

        self._env_cls = env_cls
        self._env_args = env_args
        self._lut_holder = self._env_cls.get_lut_holder()

        assert env_args.n_seats == 2

    def run(self, n_games_per_seat):
        REFERENCE_AGENT = 0
        _env = self._env_cls(env_args=self._env_args, is_evaluating=True, lut_holder=self._lut_holder)

        winnings = np.empty(shape=(n_games_per_seat * _env.N_SEATS), dtype=np.float32)

        for seat_p0 in range(_env.N_SEATS):
            seat_p1 = 1 - seat_p0

            for _hand_nr in range(n_games_per_seat):
                # """""""""""""""""
                # Reset
                # """""""""""""""""
                _, r_for_all, done, info = _env.reset()
                for e in self._eval_agents:
                    e.reset(deck_state_dict=_env.cards_state_dict())

                # """""""""""""""""
                # Play Episode
                # """""""""""""""""
                while not done:
                    p_id_acting = _env.current_player.seat_id

                    if p_id_acting == seat_p0:
                        action_int, _ = self._eval_agents[REFERENCE_AGENT].get_action(step_env=True, need_probs=False)
                        self._eval_agents[1 - REFERENCE_AGENT].notify_of_action(p_id_acted=p_id_acting,
                                                                                action_he_did=action_int)
                    elif p_id_acting == seat_p1:
                        action_int, _ = self._eval_agents[1 - REFERENCE_AGENT].get_action(step_env=True,
                                                                                          need_probs=False)
                        self._eval_agents[REFERENCE_AGENT].notify_of_action(p_id_acted=p_id_acting,
                                                                            action_he_did=action_int)
                    else:
                        raise ValueError("Only HU supported!")

                    _, r_for_all, done, info = _env.step(action_int)

                # """""""""""""""""
                # Add Rews
                # """""""""""""""""
                winnings[_hand_nr + (seat_p0 * n_games_per_seat)] = r_for_all[seat_p0] \
                                                                    * _env.REWARD_SCALAR \
                                                                    * _env.EV_NORMALIZER
        mean = np.mean(winnings).item()
        std = np.std(winnings).item()

        _d = 1.96 * std / np.sqrt(n_games_per_seat * _env.N_SEATS)
        lower_conf95 = mean - _d
        upper_conf95 = mean + _d

        print()
        print("Played", n_games_per_seat * 2, "hands of poker.")
        print("Player ", self._eval_agents[REFERENCE_AGENT].get_mode() + ":", mean, "+/-", _d)
        print("Player ", self._eval_agents[1 - REFERENCE_AGENT].get_mode() + ":", (-mean), "+/-", _d)

        return float(mean), float(upper_conf95), float(lower_conf95)
