# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.eval._.EvaluatorMasterBase import EvaluatorMasterBase
from PokerRL.rl import rl_util


class LocalHead2HeadMaster(EvaluatorMasterBase):
    """
    Evaluates two modes of an EvalAgent against each other.
    """

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        super().__init__(t_prof=t_prof, eval_env_bldr=rl_util.get_env_builder(t_prof=t_prof), chief_handle=chief_handle,
                         eval_type="Head2Head_Winnings", log_conf_interval=True)

        self._args = t_prof.module_args["h2h"]
        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)

        assert self._env_bldr.N_SEATS == 2

        self._eval_agents = [
            eval_agent_cls(t_prof=t_prof)
            for _ in range(self._env_bldr.N_SEATS)
        ]

        self._REFERENCE_AGENT = 0

    def set_modes(self, modes):
        for e, mode in zip(self._eval_agents, modes):
            e.set_mode(mode)

    def update_weights(self):
        w = self.pull_current_strat_from_chief()

        # Assumes that each eval_agent knows what to do with this dict. This should be the case after .set_modes() has
        # been called
        for e in self._eval_agents:
            e.update_weights(w)

    def evaluate(self, iter_nr):
        """ assumes same action space between all eval agents!! """

        if self._is_multi_stack:
            total_of_all_stacks = []
            pm_of_all_stacks = []

        for stack_size_idx, stack_size in enumerate(self._t_prof.eval_stack_sizes):
            for e in self._eval_agents:
                e.set_stack_size(stack_size=stack_size)

            do_it = True
            for e in self._eval_agents:
                if not e.can_compute_mode():
                    do_it = False
                    break

            if do_it:
                mean, d = self._run_eval(stack_size=stack_size)
                self._log_results(iter_nr=iter_nr,
                                  agent_mode=self._eval_agents[self._REFERENCE_AGENT].get_mode(),
                                  stack_size_idx=stack_size_idx,
                                  score=mean, upper_conf95=mean + d, lower_conf95=mean - d)

            if self._is_multi_stack:
                total_of_all_stacks.append(mean)
                pm_of_all_stacks.append(d)

        if self._is_multi_stack:
            _mean = sum(total_of_all_stacks) / float(len(total_of_all_stacks))
            _d = sum(pm_of_all_stacks) / float(len(pm_of_all_stacks))
            self._log_multi_stack(agent_mode="Head2Head",
                                  iter_nr=iter_nr,
                                  score_total=_mean,
                                  lower_conf95=_mean - _d,
                                  upper_conf95=_mean + _d,
                                  )

    def _run_eval(self, stack_size):
        winnings = np.empty(shape=(self._args.n_hands * self._env_bldr.N_SEATS), dtype=np.float32)

        _env = self._eval_env_bldr.get_new_env(is_evaluating=True, stack_size=stack_size)

        for seat_p0 in range(self._env_bldr.N_SEATS):
            seat_p1 = 1 - seat_p0

            for iteration_id in range(self._args.n_hands):

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
                        action_int, _ = self._eval_agents[self._REFERENCE_AGENT].get_action(step_env=True,
                                                                                            need_probs=False)
                        self._eval_agents[1 - self._REFERENCE_AGENT].notify_of_action(p_id_acted=p_id_acting,
                                                                                      action_he_did=action_int)
                    elif p_id_acting == seat_p1:
                        action_int, _ = self._eval_agents[1 - self._REFERENCE_AGENT].get_action(step_env=True,
                                                                                                need_probs=False)
                        self._eval_agents[self._REFERENCE_AGENT].notify_of_action(p_id_acted=p_id_acting,
                                                                                  action_he_did=action_int)
                    else:
                        raise ValueError("Only HU supported!")

                    _, r_for_all, done, info = _env.step(action_int)

                # """""""""""""""""
                # Add Rews
                # """""""""""""""""
                winnings[iteration_id + (seat_p0 * self._args.n_hands)] = r_for_all[seat_p0] \
                                                                          * _env.REWARD_SCALAR \
                                                                          * _env.EV_NORMALIZER

        return self._get_95confidence(winnings)
