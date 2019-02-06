# Copyright (c) 2019 Eric Steinberger


import copy

import numpy as np

from PokerRL.game._.tree.PublicTree import PublicTree
from PokerRL.game.wrappers import HistoryEnvBuilder
from PokerRL.rl.rl_util import get_env_cls_from_str


class CFRBase:
    """
    base class to all full-width (i.e. non MC) tabular CFR methods
    """

    def __init__(self,
                 name,
                 chief_handle,
                 game_cls,
                 agent_bet_set,
                 algo_name,
                 starting_stack_sizes=None,
                 ):
        """
        Args:
            name (str):                             Under this name all logs, data, and checkpoints will appear.
            chief_handle (ChiefBase):               Reference to chief worker
            game_cls (PokerEnv subclass):           Class (not instance) to be trained in.
            agent_bet_set (iterable):               Choosing a bet-set from bet_sets.py is recommended. If solving a
                                                    Limit poker game, this value will not be considered, but must still
                                                    be passed. Just set this to any list of floats (e.g. [0.0])
            starting_stack_sizes (list of ints):    For each stack size in this list, a CFR strategy will be computed.
                                                    Results are logged individually and averaged (uniform).
                                                    If None, takes the default for the game.
        """

        self._name = name
        self._n_seats = 2

        self._chief_handle = chief_handle

        if starting_stack_sizes is None:
            self._starting_stack_sizes = [game_cls.DEFAULT_STACK_SIZE]
        else:
            self._starting_stack_sizes = copy.deepcopy(starting_stack_sizes)
        self._game_cls_str = game_cls.__name__

        self._env_args = [
            game_cls.ARGS_CLS(n_seats=self._n_seats,
                              starting_stack_sizes_list=[start_chips for _ in range(self._n_seats)],
                              bet_sizes_list_as_frac_of_pot=agent_bet_set,
                              )
            for start_chips in self._starting_stack_sizes
        ]
        self._env_bldrs = [
            HistoryEnvBuilder(env_cls=get_env_cls_from_str(self._game_cls_str),
                              env_args=self._env_args[s])

            for s in range(len(self._starting_stack_sizes))
        ]

        self._trees = [
            PublicTree(env_bldr=self._env_bldrs[idx],
                       stack_size=self._env_args[idx].starting_stack_sizes_list,
                       stop_at_street=None)
            for idx in range(len(self._env_bldrs))
        ]

        for tree in self._trees:
            tree.build_tree()
            print("Tree with stack size", tree.stack_size, "has", tree.n_nodes, "nodes out of which", tree.n_nonterm,
                  "are non-terminal.")

        self._algo_name = algo_name

        self._exps_curr_total = [
            self._chief_handle.create_experiment(
                self._name + "_Curr_S" + str(self._starting_stack_sizes[s]) + "_total_" + self._algo_name)
            for s in range(len(self._starting_stack_sizes))
        ]

        self._exps_avg_total = [
            self._chief_handle.create_experiment(
                self._name + "_Avg_total_S" + str(self._starting_stack_sizes[s]) + "_" + self._algo_name)
            for s in range(len(self._starting_stack_sizes))
        ]

        self._exp_all_averaged_curr_total = self._chief_handle.create_experiment(
            self._name + "_Curr_total_averaged_" + self._algo_name)

        self._exp_all_averaged_avg_total = self._chief_handle.create_experiment(
            self._name + "_Avg_total_averaged_" + self._algo_name)

        self._iter_counter = None

    @property
    def name(self):
        return self._name

    @property
    def algo_name(self):
        return self._algo_name

    @property
    def iter_counter(self):
        return self._iter_counter

    def reset(self):
        self._iter_counter = 0

        for p in range(self._n_seats):
            self._reset_player(p_id=p)

        for t_idx in range(len(self._trees)):
            self._trees[t_idx].fill_uniform_random()

        self._compute_cfv()
        self._log_curr_strat_expl()

    def iteration(self):
        for p in range(self._n_seats):
            self._compute_cfv()
            self._compute_regrets(p_id=p)
            self._compute_new_strategy(p_id=p)
            self._update_reach_probs()
            self._add_strategy_to_average(p_id=p)

        self._iter_counter += 1

        self._compute_cfv()
        self._log_curr_strat_expl()
        self._evaluate_avg_strats()

    def _compute_cfv(self):
        for t_idx in range(len(self._trees)):
            self._trees[t_idx].compute_ev()

    def _regret_formula_first_it(self, ev_all_actions, strat_ev):
        raise NotImplementedError

    def _regret_formula_after_first_it(self, ev_all_actions, strat_ev, last_regrets):
        raise NotImplementedError

    def _compute_regrets(self, p_id):

        for t_idx in range(len(self._trees)):
            def __compute_evs(_node):
                # EV of each action
                N_ACTIONS = len(_node.children)
                ev_all_actions = np.zeros(shape=(self._env_bldrs[t_idx].rules.RANGE_SIZE, N_ACTIONS), dtype=np.float32)
                for i, child in enumerate(_node.children):
                    ev_all_actions[:, i] = child.ev[p_id]

                # EV if playing by curr strat
                strat_ev = _node.ev[p_id]
                strat_ev = np.expand_dims(strat_ev, axis=-1).repeat(N_ACTIONS, axis=-1)

                return strat_ev, ev_all_actions

            def _fill_after_first(_node):
                if _node.p_id_acting_next == p_id:
                    strat_ev, ev_all_actions = __compute_evs(_node=_node)
                    _node.data["regret"] = self._regret_formula_after_first_it(ev_all_actions=ev_all_actions,
                                                                               strat_ev=strat_ev,
                                                                               last_regrets=_node.data["regret"])

                for c in _node.children:
                    _fill_after_first(c)

            def _fill_first(_node):
                if _node.p_id_acting_next == p_id:
                    strat_ev, ev_all_actions = __compute_evs(_node=_node)

                    _node.data["regret"] = self._regret_formula_first_it(ev_all_actions=ev_all_actions,
                                                                         strat_ev=strat_ev)

                for c in _node.children:
                    _fill_first(c)

            if self._iter_counter == 0:
                _fill_first(self._trees[t_idx].root)
            else:
                _fill_after_first(self._trees[t_idx].root)

    def _compute_new_strategy(self, p_id):
        """ Assumes regrets have been computed for player ""p_id"" already! """
        raise NotImplementedError

    def _update_reach_probs(self):
        for t_idx in range(len(self._trees)):
            self._trees[t_idx].update_reach_probs()

    def _add_strategy_to_average(self, p_id):
        raise NotImplementedError

    def _log_curr_strat_expl(self):
        expl_totals = []
        for t_idx in range(len(self._trees)):
            METRIC = self._env_bldrs[t_idx].env_cls.WIN_METRIC
            expl_p = [
                float(self._trees[t_idx].root.exploitability[p]) * self._env_bldrs[t_idx].env_cls.EV_NORMALIZER
                for p in range(self._n_seats)
            ]
            expl_total = sum(expl_p) / self._n_seats
            expl_totals.append(expl_total)

            self._chief_handle.add_scalar(self._exps_curr_total[t_idx],
                                          "Evaluation/" + METRIC, self._iter_counter, expl_total)

            self._trees[t_idx].export_to_file(name=self._name + "_Curr_" + str(self._iter_counter))

        expl_total_averaged = sum(expl_totals) / float(len(expl_totals))
        self._chief_handle.add_scalar(self._exp_all_averaged_curr_total,
                                      "Evaluation/" + METRIC, self._iter_counter, expl_total_averaged)

    def _evaluate_avg_strats(self):
        expl_totals = []
        for t_idx in range(len(self._trees)):
            METRIC = self._env_bldrs[t_idx].env_cls.WIN_METRIC
            eval_tree = PublicTree(env_bldr=self._env_bldrs[t_idx],
                                   stack_size=self._env_args[t_idx].starting_stack_sizes_list,
                                   stop_at_street=None,
                                   is_debugging=False,
                                   )
            eval_tree.build_tree()

            def _fill(_node_eval, _node_train):
                if _node_eval.p_id_acting_next != eval_tree.CHANCE_ID and (not _node_eval.is_terminal):
                    _node_eval.strategy = np.copy(_node_train.data["avg_strat"])
                    assert np.allclose(np.sum(_node_eval.strategy, axis=1), 1, atol=0.0001)

                for c_eval, c_train in zip(_node_eval.children, _node_train.children):
                    _fill(_node_eval=c_eval, _node_train=c_train)

            # sets up some stuff; we overwrite strategy afterwards
            eval_tree.fill_uniform_random()

            # fill with strat
            _fill(_node_eval=eval_tree.root, _node_train=self._trees[t_idx].root)
            eval_tree.update_reach_probs()

            # compute EVs
            eval_tree.compute_ev()

            eval_tree.export_to_file(name=self._name + "_Avg_" + str(self._iter_counter))

            # log
            expl_p = [
                float(eval_tree.root.exploitability[p]) * self._env_bldrs[t_idx].env_cls.EV_NORMALIZER
                for p in range(eval_tree.n_seats)
            ]
            expl_total = sum(expl_p) / eval_tree.n_seats
            expl_totals.append(expl_total)

            self._chief_handle.add_scalar(self._exps_avg_total[t_idx],
                                          "Evaluation/" + METRIC, self._iter_counter, expl_total)

        expl_total_averaged = sum(expl_totals) / float(len(expl_totals))
        self._chief_handle.add_scalar(self._exp_all_averaged_avg_total,
                                      "Evaluation/" + METRIC, self._iter_counter, expl_total_averaged)

    def _reset_player(self, p_id):
        def __reset(_node, _p_id):
            if _node.p_id_acting_next == _p_id:
                # regrets and strategies only need to be stored for one player at each node
                _node.data = {
                    "regret": None,
                    "avg_strat": None
                }
                _node.strategy = None

            for c in _node.children:
                __reset(c, _p_id=_p_id)

        for t_idx in range(len(self._trees)):
            __reset(self._trees[t_idx].root, _p_id=p_id)
