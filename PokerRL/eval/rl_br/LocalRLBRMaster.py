# Copyright (c) 2019 Eric Steinberger


import copy

import numpy as np

from PokerRL.eval._.EvaluatorMasterBase import EvaluatorMasterBase
from PokerRL.eval.rl_br import _util
from PokerRL.rl.agent_modules.DDQN import DDQN


class LocalRLBRMaster(EvaluatorMasterBase):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        super().__init__(t_prof=t_prof,
                         eval_env_bldr=_util.get_env_builder_rlbr(t_prof=t_prof),
                         chief_handle=chief_handle,
                         eval_type="RL-BR",
                         log_conf_interval=True)

        assert self._eval_env_bldr.N_SEATS == 2, "only works for 2 players at the moment"

        self._args = t_prof.module_args["rlbr"]
        self._eval_agent = eval_agent_cls(t_prof=t_prof)

        self._la_handles = None
        self._ps_handle = None

    def set_learner_actors(self, *las):
        self._la_handles = list(las)

    def set_param_server(self, param_server):
        self._ps_handle = param_server

    def update_weights(self):
        w = self.pull_current_strat_from_chief()
        self._eval_agent.update_weights(copy.deepcopy(w))

    def evaluate(self, global_iter_nr):

        for mode in self._t_prof.eval_modes_of_algo:
            for stack_size_idx, stack_size in enumerate(self._t_prof.eval_stack_sizes):
                self._eval_agent.set_mode(mode=mode)
                self._eval_agent.set_stack_size(stack_size=stack_size)
                if self._eval_agent.can_compute_mode():
                    self._retrain(mode=mode, stack_size=stack_size, stack_size_idx=stack_size_idx,
                                  global_iter_nr=global_iter_nr)
                    # """""""""""""""""""
                    # Compute RL-BR
                    # """""""""""""""""""
                    print("Running rollout matches between RL-BR and agent.")
                    ddqn_states = self._ray.get(self._ray.remote(self._ps_handle.get_eval_ddqn_state_dicts))
                    ddqns = [
                        DDQN.inference_version_from_state_dict(state_dict=ddqn_states[p],
                                                               env_bldr=self._eval_env_bldr)
                        for p in range(self._t_prof.n_seats)
                    ]
                    scores = self._compute_rlbr(
                        n_hands_each_seat=self._args.n_hands_each_seat,
                        rlbr_dqn_each_seat=ddqns,
                        rlbr_env_wrapper=self._eval_env_bldr.get_new_wrapper(is_evaluating=True, stack_size=stack_size),
                        opponent=self._eval_agent
                    )

                    mean, d = self._get_95confidence(scores=scores)

                    self._log_results(iter_nr=global_iter_nr, agent_mode=mode, stack_size_idx=stack_size_idx,
                                      score=mean, lower_conf95=mean - d, upper_conf95=mean + d)

    def _retrain(self, mode, stack_size, stack_size_idx, global_iter_nr):
        # """""""""""""""""""
        # Prepare Logging
        # """""""""""""""""""
        if self._t_prof.log_verbose:
            running_rew_exp = self._ray.get(
                self._ray.remote(self._chief_handle.create_experiment,
                                 self._t_prof.name + "_M_" + mode + "_S" + str(stack_size_idx) + "_I" + str(
                                     global_iter_nr) + "Running Rew RL-BR"))
            eps_exp = self._ray.get([
                self._ray.remote(self._chief_handle.create_experiment,
                                 self._t_prof.name + "_M_" + mode + "_S" + str(stack_size_idx) + "_I" + str(
                                     global_iter_nr) + "Epsilon RL-BR P" + str(p))
                for p in range(self._eval_env_bldr.N_SEATS)
            ])

        logging_scores_each_p = [[] for _ in range(self._eval_env_bldr.N_SEATS)]
        logging_eps_per_p = [[] for _ in range(self._eval_env_bldr.N_SEATS)]
        logging_timesteps = []

        # """""""""""""""""""
        # Training
        # """""""""""""""""""
        self._ray.wait([
            self._ray.remote(self._ps_handle.reset, p, global_iter_nr)
            for p in range(self._eval_env_bldr.N_SEATS)
        ])

        for p_training in range(self._eval_env_bldr.N_SEATS):
            print("Training RL-BR seat", p_training, "with agent mode", mode, "and stack size", stack_size_idx)

            # """"""""
            # Reset
            # """"""""
            self._eval_agent.reset()

            self._ray.wait([
                self._ray.remote(la.reset,
                                 p_training, self._eval_agent.state_dict(), stack_size)
                for la in self._la_handles
            ])
            self._update_leaner_actors(update_eps_for_plyrs=[p_training], update_net_for_plyrs=[p_training])

            # """"""""
            # Pre-play
            # """"""""
            self._ray.wait([
                self._ray.remote(la.play, self._args.pretrain_n_games)
                for la in self._la_handles
            ])

            # """"""""
            # Learn
            # """"""""
            SMOOTHING = 200
            accum_score = 0.0
            for training_iter_id in range(self._args.n_iterations):
                self._ray.wait([
                    self._ray.remote(self._ps_handle.update_eps, p_training, training_iter_id)
                ])

                self._update_leaner_actors(update_eps_for_plyrs=[p_training], update_net_for_plyrs=[p_training])

                # Play
                scores_all_las = self._ray.get([
                    self._ray.remote(la.play, self._args.play_n_games_per_iter)
                    for la in self._la_handles
                ])

                accum_score += sum(scores_all_las) / self._args.n_las

                # Get Gradients
                grads_from_all_las = self._get_gradients(p_id=p_training)

                # Applying gradients
                self._ray.wait([
                    self._ray.remote(self._ps_handle.apply_grads,
                                     p_training, grads_from_all_las)
                ])

                # Update weights on all las
                self._update_leaner_actors(update_net_for_plyrs=[p_training])

                # Periodically update target net
                if (training_iter_id + 1) % self._args.ddqn_args.target_net_update_freq:
                    self._ray.wait([
                        self._ray.remote(la.update_target_net,
                                         p_training)
                        for la in self._la_handles
                    ])

                if (training_iter_id + 1) % SMOOTHING == 0:
                    print("RL-BR", "P" + str(p_training) + " iter", training_iter_id + 1)
                    accum_score /= SMOOTHING
                    logging_scores_each_p[p_training].append(accum_score)
                    logging_eps_per_p[p_training].append(
                        self._ray.get(self._ray.remote(self._ps_handle.get_eps, p_training)))
                    if p_training == 0:  # Only need to collect that once
                        logging_timesteps.append(training_iter_id + 1)
                    accum_score = 0.0

        # """""""""""""""""""
        # Logging
        # """""""""""""""""""
        if self._t_prof.log_verbose:
            logging_scores_avg = [
                sum([
                    logging_scores_each_p[i][t]
                    for i in range(self._eval_env_bldr.N_SEATS)
                ]) / self._eval_env_bldr.N_SEATS
                for t in range(len(logging_scores_each_p[0]))
            ]

            for i, logging_iter in enumerate(logging_timesteps):
                self._ray.remote(
                    self._chief_handle.add_scalar,
                    running_rew_exp, "RL-BR/Running Reward While Training", logging_iter,
                    logging_scores_avg[i])

                for p in range(self._eval_env_bldr.N_SEATS):
                    self._ray.remote(
                        self._chief_handle.add_scalar,
                        eps_exp[p], "RL-BR/Training Epsilon", logging_iter, logging_eps_per_p[p][i])

    def _get_gradients(self, p_id):
        grads = [
            self._ray.remote(la.get_grads,
                             p_id)
            for la in self._la_handles
        ]
        self._ray.wait(grads)

        return grads

    def _update_leaner_actors(self, update_eps_for_plyrs=None, update_net_for_plyrs=None):
        assert isinstance(update_net_for_plyrs, list) or update_net_for_plyrs is None
        assert isinstance(update_eps_for_plyrs, list) or update_eps_for_plyrs is None

        _update_net_per_p = [
            True if (update_net_for_plyrs is not None) and (p in update_net_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]

        _update_eps_per_p = [
            True if (update_eps_for_plyrs is not None) and (p in update_eps_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]

        la_batches = []
        n = len(self._la_handles)
        c = 0
        while n > c:
            s = min(n, c + self._t_prof.max_n_las_sync_simultaneously)
            la_batches.append(self._la_handles[c:s])
            if type(la_batches[-1]) is not list:
                la_batches[-1] = [la_batches[-1]]
            c = s

        eps = [None for _ in range(self._t_prof.n_seats)]
        nets = [None for _ in range(self._t_prof.n_seats)]
        for p_id in range(self._t_prof.n_seats):
            eps[p_id] = None if not _update_eps_per_p[p_id] else self._ray.remote(
                self._ps_handle.get_eps, p_id)

            nets[p_id] = None if not _update_net_per_p[p_id] else self._ray.remote(
                self._ps_handle.get_weights, p_id)

        for batch in la_batches:
            self._ray.wait([
                self._ray.remote(la.update,
                                 eps,
                                 nets)
                for la in batch
            ])

    @staticmethod
    def _compute_rlbr(n_hands_each_seat, rlbr_dqn_each_seat, rlbr_env_wrapper, opponent):
        agent_losses = np.empty(shape=n_hands_each_seat * 2, dtype=np.float32)

        for rlbr_seat_id in range(rlbr_env_wrapper.env.N_SEATS):

            rlbr_agent = rlbr_dqn_each_seat[rlbr_seat_id]

            for iteration_id in range(n_hands_each_seat):

                # """""""""""""""""
                # Reset
                # """""""""""""""""
                obs, r_for_all, done, info = _util.reset_episode_multi_action_space(rlbr_env_wrapper=rlbr_env_wrapper,
                                                                                    opponent_agent=opponent)
                range_idx_rlbr = rlbr_env_wrapper.env.get_range_idx(p_id=rlbr_seat_id)

                # """""""""""""""""
                # Play Episode
                # """""""""""""""""
                while not done:
                    p_id_acting = rlbr_env_wrapper.env.current_player.seat_id

                    # RL-BR acting
                    if p_id_acting == rlbr_seat_id:
                        action_int = rlbr_agent.select_br_a(
                            pub_obses=[obs],
                            range_idxs=np.array([range_idx_rlbr], dtype=np.int32),
                            legal_actions_lists=[rlbr_env_wrapper.env.get_legal_actions()],
                            explore=False,
                        )[0]
                        _util.notify_agent_multi_action_space(action_int=action_int, rlbr_seat_id=rlbr_seat_id,
                                                              rlbr_env_wrapper=rlbr_env_wrapper,
                                                              opponent_agent=opponent)

                        # Step
                        obs, r_for_all, done, info = rlbr_env_wrapper.step(action=action_int)

                    # EvalAgent (opponent) acting
                    else:
                        action_int, _ = opponent.get_action(step_env=True, need_probs=False)
                        obs, r_for_all, done, info = _util.step_from_opp_action(action_int=action_int,
                                                                                opponent=opponent,
                                                                                rlbr_env_wrapper=rlbr_env_wrapper)

                # add rews
                agent_losses[iteration_id + (rlbr_seat_id * n_hands_each_seat)] = r_for_all[rlbr_seat_id] \
                                                                                  * rlbr_env_wrapper.env.REWARD_SCALAR \
                                                                                  * rlbr_env_wrapper.env.EV_NORMALIZER

        return agent_losses
