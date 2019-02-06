# Copyright (c) 2019 Eric Steinberger


import os

import psutil

from PokerRL.eval.rl_br import _util
from PokerRL.rl import rl_util
from PokerRL.rl.agent_modules.DDQN import DDQN
from PokerRL.rl.base_cls.workers.ParameterServerBase import ParameterServerBase as _ParameterServerBase
from PokerRL.rl.neural.DuelingQNet import DuelingQNet


class Local_RLBR_ParameterServer(_ParameterServerBase):

    def __init__(self, t_prof, chief_handle):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle)
        self._args = t_prof.module_args["rlbr"]
        self._eval_env_bldr = _util.get_env_builder_rlbr(t_prof=t_prof)
        self._env_bldr = self._eval_env_bldr  # Override base class variable

        if self._t_prof.log_verbose:
            self._exp_mem_usage = self._ray.get(
                self._ray.remote(self._chief_handle.create_experiment,
                                 self._t_prof.name + "_RL-BR_PS_Memory_Usage")
            )

        self._nets = [None for _ in range(self._env_bldr.N_SEATS)]
        self._eps = [None for _ in range(self._env_bldr.N_SEATS)]
        self._optims = [None for _ in range(self._env_bldr.N_SEATS)]

    def reset(self, p_id, global_iteration):
        self._nets[p_id] = self._get_new_net()
        self._optims[p_id] = self._get_new_optim(p_id=p_id)
        self.update_eps(p_id=p_id, update_nr=0)

        if self._t_prof.log_verbose:
            # Logs
            process = psutil.Process(os.getpid())
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_mem_usage, "Debug/MemoryUsage/PS", global_iteration, process.memory_info().rss)

    def get_weights(self, p_id):
        self._nets[p_id].zero_grad()
        return self._ray.state_dict_to_numpy(self._nets[p_id].state_dict())

    def apply_grads(self, p_id, list_of_grads):
        self._apply_grads(list_of_grads=list_of_grads, optimizer=self._optims[p_id], net=self._nets[p_id],
                          grad_norm_clip=self._args.ddqn_args.grad_norm_clipping)

    def update_eps(self, p_id, update_nr):
        self._eps[p_id] = rl_util.polynomial_decay(base=self._args.ddqn_args.eps_start,
                                                   minimum=self._args.ddqn_args.eps_min,
                                                   const=self._args.ddqn_args.eps_const,
                                                   exponent=self._args.ddqn_args.eps_exponent,
                                                   counter=update_nr)

    def get_eps(self, p_id):
        return self._eps[p_id]

    def _get_new_net(self):
        return DuelingQNet(q_args=self._args.ddqn_args.q_args, env_bldr=self._eval_env_bldr, device=self._device)

    def _get_new_optim(self, p_id):
        return rl_util.str_to_optim_cls(self._args.ddqn_args.optim_str)(self._nets[p_id].parameters(),
                                                                        lr=self._args.ddqn_args.lr)

    def get_eval_ddqn_state_dicts(self):
        ddqns = []
        for p in range(self._eval_env_bldr.N_SEATS):
            ddqn = DDQN(owner=p, ddqn_args=self._args.ddqn_args, env_bldr=self._eval_env_bldr)
            ddqn.load_net_state_dict(self._nets[p].state_dict())
            ddqn.update_target_net()
            ddqn.eps = None
            ddqn.buf = None
            ddqns.append(ddqn.state_dict())
        return ddqns
