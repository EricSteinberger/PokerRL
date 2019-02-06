# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase


class EvaluatorMasterBase(WorkerBase):
    """
    Baseclass to all Evaluators. An Evaluator is an algorithm to evaluate an agent's performance in a certain metric.
    """

    def __init__(self, t_prof, eval_env_bldr, chief_handle, eval_type, log_conf_interval=False):
        """
        Args:
            t_prof (TrainingProfile)
            chief_handle (class instance or ray ActorHandle)
            eval_type (str):                Name of the evaluator
        """
        super().__init__(t_prof=t_prof)
        self._eval_env_bldr = eval_env_bldr
        self._chief_handle = chief_handle
        self._chief_info = [None for _ in range(self._t_prof.n_seats)]

        self._is_multi_stack = len(self._t_prof.eval_stack_sizes) > 1

        self._log_conf_interval = log_conf_interval

        self._exp_name_total, self._exp_names_conf = self._create_experiments(self_name=eval_type)

        if self._is_multi_stack:
            self._exp_name_multi_stack = {
                eval_mode:
                    self._ray.get(
                        self._ray.remote(self._chief_handle.create_experiment,
                                         self._t_prof.name
                                         + " " + eval_mode
                                         + "Multi_Stack"
                                         + ": " + eval_type
                                         + " Averaged Total"))
                for eval_mode in self._t_prof.eval_modes_of_algo
            }
            if self._log_conf_interval:
                self._exp_names_multi_stack_conf = {
                    eval_mode:
                        self._ray.get(
                            [
                                self._ray.remote(self._chief_handle.create_experiment,
                                                 self._t_prof.name
                                                 + " " + eval_mode
                                                 + ": " + eval_type
                                                 + " Conf_" + bound_end)
                                for bound_end in ["lower95", "upper95"]
                            ]
                        )
                    for eval_mode in self._t_prof.eval_modes_of_algo
                }

    @property
    def is_multi_stack(self):
        """
        Whether the agent is evaluated in games that start with different stack sizes each time.
        """
        return self._is_multi_stack

    def evaluate(self, iter_nr):
        """ Evaluate an agent and send the results as logs to the Chief. """
        raise NotImplementedError

    def update_weights(self):
        """ Update the local weights on the master, for instance by calling .pull_current_strat_from_chief()  """
        raise NotImplementedError

    def pull_current_strat_from_chief(self):
        """
        Pulls and Returns weights or any other changing algorithm info of any format from the Chief.
        """
        w, self._chief_info = self._ray.get(self._ray.remote(self._chief_handle.pull_current_eval_strategy,
                                                             self._chief_info))
        return w

    def _create_experiments(self, self_name, ):
        """
        Registers a new experiment either for each player and their average or just for their average.
        """

        if self._log_conf_interval:
            exp_names_conf = {
                eval_mode:
                    [
                        self._ray.get(
                            [
                                self._ray.remote(self._chief_handle.create_experiment,
                                                 self._t_prof.name
                                                 + " " + eval_mode
                                                 + "_stack_" + str(stack_size[0])
                                                 + ": " + self_name
                                                 + " Conf_" + bound_end)
                                for bound_end in ["lower95", "upper95"]
                            ]
                        )
                        for stack_size in self._t_prof.eval_stack_sizes
                    ]
                for eval_mode in self._t_prof.eval_modes_of_algo
            }
        else:
            exp_names_conf = None

        exp_name_total = {
            eval_mode:
                [
                    self._ray.get(
                        self._ray.remote(self._chief_handle.create_experiment,
                                         self._t_prof.name
                                         + " " + eval_mode
                                         + "_stack_" + str(stack_size[0])
                                         + ": " + self_name
                                         + " Total"))
                    for stack_size in self._t_prof.eval_stack_sizes
                ]
            for eval_mode in self._t_prof.eval_modes_of_algo
        }

        return exp_name_total, exp_names_conf

    def _get_95confidence(self, scores):
        mean = np.mean(scores).item()
        std = np.std(scores).item()

        _d = 1.96 * std / np.sqrt(scores.shape[0])
        return float(mean), float(_d)

    def _log_results(self, agent_mode, stack_size_idx, iter_nr, score, upper_conf95=None, lower_conf95=None):
        """
        Log evaluation results by sending these results to the Chief, who will later send them to the Crayon log server.

        Args:
            agent_mode:             Evaluation mode of the agent whose performance is logged
            stack_size_idx:         If evaluating multiple starting stack sizes, this is an index describing which one
                                    this data is from.
            iter_nr:                Algorithm Iteration of this data
            score:                  Score in this evaluation (e.g. exploitability)
        """
        graph_name = "Evaluation/" + self._eval_env_bldr.env_cls.WIN_METRIC

        self._ray.remote(self._chief_handle.add_scalar,
                         self._exp_name_total[agent_mode][stack_size_idx], graph_name, iter_nr, score)

        if self._log_conf_interval:
            assert upper_conf95 is not None
            assert lower_conf95 is not None
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_names_conf[agent_mode][stack_size_idx][0], graph_name, iter_nr, lower_conf95)
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_names_conf[agent_mode][stack_size_idx][1], graph_name, iter_nr, upper_conf95)

    def _log_multi_stack(self, agent_mode, iter_nr, score_total, upper_conf95=None, lower_conf95=None):
        """
        Additional logging for multistack evaluations
        """
        graph_name = "Evaluation/" + self._eval_env_bldr.env_cls.WIN_METRIC
        self._ray.remote(self._chief_handle.add_scalar,
                         self._exp_name_multi_stack[agent_mode], graph_name, iter_nr, score_total)

        if self._log_conf_interval:
            assert upper_conf95 is not None
            assert lower_conf95 is not None
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_names_multi_stack_conf[agent_mode][0], graph_name, iter_nr, lower_conf95)
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_names_multi_stack_conf[agent_mode][1], graph_name, iter_nr, upper_conf95)
