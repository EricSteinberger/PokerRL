# Copyright (c) 2019 Eric Steinberger


"""
The local master contains logic to manage many workers, but if instantiated as local, will only manage one. It handles
logging and provides an interface to the LBR computation.
"""
import numpy as np

from PokerRL.eval._.EvaluatorMasterBase import EvaluatorMasterBase
from PokerRL.eval.lbr import _util


class LocalLBRMaster(EvaluatorMasterBase):
    """
    LBR computation as described in https://arxiv.org/abs/1612.07547
    
    EvalLBRMaster manages a sub-cluster of EvalLBRWorker nodes.
    """

    def __init__(self, t_prof, chief_handle):
        assert t_prof.n_seats == 2

        EvaluatorMasterBase.__init__(self, t_prof=t_prof, eval_env_bldr=_util.get_env_builder_lbr(t_prof=t_prof),
                                     chief_handle=chief_handle, eval_type="LBR", log_conf_interval=True)

        self.lbr_args = t_prof.module_args["lbr"]

        self.weights_for_eval_agent = None
        self.alive_worker_handles = None

    def set_worker_handles(self, *worker_handles):
        self.alive_worker_handles = list(worker_handles)

    def evaluate(self, iter_nr):
        # __________________________________ send weights from Master to all Workers ___________________________________
        self._ray.wait([
            self._ray.remote(
                worker.update_weights,
                self.weights_for_eval_agent
            )
            for worker in self.alive_worker_handles
        ])

        # _______________________________ Evaluate for all specified eval_modes_of_algo ________________________________
        for mode in self._t_prof.eval_modes_of_algo:
            if self._is_multi_stack:
                total_of_all_stacks = []
                pm_of_all_stacks = []

            for stack_size_idx, stack_size in enumerate(self._t_prof.eval_stack_sizes):
                scores = []
                for p_id in range(self._t_prof.n_seats):
                    scores += (self._ray.get(
                        [
                            self._ray.remote(worker.run,
                                             p_id,
                                             int(self.lbr_args.n_lbr_hands / self.lbr_args.n_workers),
                                             mode,
                                             stack_size
                                             )
                            for worker in self.alive_worker_handles
                        ]
                    ))

                scores = [s for s in scores if s is not None]
                scores = np.concatenate(scores, axis=0)
                if len(scores) > 0:
                    mean, d = self._get_95confidence(scores)

                    self._log_results(iter_nr=iter_nr,
                                      agent_mode=mode,
                                      stack_size_idx=stack_size_idx,
                                      score=mean, upper_conf95=mean + d, lower_conf95=mean - d)

                    if self._is_multi_stack:
                        total_of_all_stacks.append(mean)

            if self._is_multi_stack:
                _mean = sum(total_of_all_stacks) / float(len(total_of_all_stacks))
                _d = sum(pm_of_all_stacks) / float(len(pm_of_all_stacks))
                self._log_multi_stack(agent_mode=mode,
                                      iter_nr=iter_nr,
                                      score_total=_mean,
                                      upper_conf95=_mean + _d,
                                      lower_conf95=_mean - _d,
                                      )

    def update_weights(self):
        self.weights_for_eval_agent = self.pull_current_strat_from_chief()
