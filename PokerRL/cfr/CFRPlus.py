# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.cfr._CFRBase import CFRBase as _CFRBase


class CFRPlus(_CFRBase):

    def __init__(self,
                 name,
                 chief_handle,
                 game_cls,
                 agent_bet_set,
                 starting_stack_sizes=None,
                 delay=0, ):
        """
        delay (int):                            Linear Averaging delay of CFR+ (only applicable if ""cfr_plus"" is
                                                True)
        """
        super().__init__(name=name,
                         chief_handle=chief_handle,
                         game_cls=game_cls,
                         starting_stack_sizes=starting_stack_sizes,
                         agent_bet_set=agent_bet_set,
                         algo_name="CFRp_delay" + str(delay)
                         )

        self.delay = delay
        self.reset()

    def _evaluate_avg_strats(self):
        if self._iter_counter > self.delay:
            return super()._evaluate_avg_strats()

    def _regret_formula_after_first_it(self, ev_all_actions, strat_ev, last_regrets):
        return np.maximum(ev_all_actions - strat_ev + last_regrets, 0)

    def _regret_formula_first_it(self, ev_all_actions, strat_ev):
        return np.maximum(ev_all_actions - strat_ev, 0)  # not max of axis; this is like relu

    def _compute_new_strategy(self, p_id):
        for t_idx in range(len(self._trees)):
            def _fill(_node):
                if _node.p_id_acting_next == p_id:
                    N = len(_node.children)

                    _reg = _node.data["regret"]
                    _reg_sum = np.expand_dims(np.sum(_reg, axis=1), axis=1).repeat(N, axis=1)

                    with np.errstate(divide='ignore', invalid='ignore'):
                        _node.strategy = np.where(
                            _reg_sum > 0.0,
                            _reg / _reg_sum,
                            np.full(shape=(self._env_bldrs[t_idx].rules.RANGE_SIZE, N,), fill_value=1.0 / N,
                                    dtype=np.float32)
                        )

                for c in _node.children:
                    _fill(c)

            _fill(self._trees[t_idx].root)

    def _add_strategy_to_average(self, p_id):
        def _fill(_node):
            if _node.p_id_acting_next == p_id:
                if self._iter_counter > self.delay:
                    current_weight = np.sum(np.arange(self.delay + 1, self._iter_counter + 1))
                    new_weight = self._iter_counter - self.delay + 1

                    m_old = current_weight / (current_weight + new_weight)
                    m_new = new_weight / (current_weight + new_weight)
                    _node.data["avg_strat"] = m_old * _node.data["avg_strat"] + m_new * _node.strategy

                    assert np.allclose(np.sum(_node.data["avg_strat"], axis=1), 1, atol=0.0001)

                elif self._iter_counter == self.delay:
                    _node.data["avg_strat"] = np.copy(_node.strategy)

                    assert np.allclose(np.sum(_node.data["avg_strat"], axis=1), 1, atol=0.0001)

            for c in _node.children:
                _fill(c)

        for t_idx in range(len(self._trees)):
            _fill(self._trees[t_idx].root)
