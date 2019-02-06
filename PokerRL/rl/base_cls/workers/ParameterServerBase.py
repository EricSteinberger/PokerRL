# Copyright (c) 2019 Eric Steinberger


import torch

from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase


class ParameterServerBase(WorkerBase):
    """
    You do NOT have to use a ParameterServer for your algorithm, but if you do, you can subclass this Base class to
    make use of pre-written functions. You can, however, also create a PS architecture without inheriting from this
    class.
    """

    def __init__(self, t_prof, chief_handle):
        super().__init__(t_prof=t_prof)

        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)
        self._chief_handle = chief_handle

        self._device = torch.device(t_prof.device_parameter_server)

    def _apply_grads(self, list_of_grads, net, optimizer, grad_norm_clip=None):
        optimizer.zero_grad()
        n_grad_sources = len(list_of_grads)

        # _______________________________________ pull the grads from workers __________________________________________
        # if the code is run distributed, only Object IDs pointing to specific workers arrive, so we have to pull them
        # we batch the .get() calls for stability. Here we create batches
        grad_batches = []
        c = 0
        while n_grad_sources > c:
            s = min(n_grad_sources, c + self._t_prof.max_n_las_sync_simultaneously)
            grad_batches.append(list_of_grads[c:s])
            if type(grad_batches[-1]) is not list:
                grad_batches[-1] = [grad_batches[-1]]
            c = s

        # final grads
        list_of_grads = []
        for batch in grad_batches:
            for g in self._ray.get(batch):
                if g is not None:
                    list_of_grads.append(self._ray.grads_to_torch(g, device=self._device))

        # ________________________________________________ apply grads _________________________________________________
        n_grad_sources = len(list_of_grads)  # have to recompute because the ones that were None are removed now
        if n_grad_sources > 0:
            # reshape and average
            grads = {}
            for name, _ in net.named_parameters():
                grads[name] = []
                for i in range(n_grad_sources):
                    grads[name].append(list_of_grads[i][name])
                grads[name] = torch.mean(torch.stack(grads[name], dim=0), dim=0)

            for name, param in net.named_parameters():
                param.grad = grads[name]

            if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=grad_norm_clip)

            optimizer.step()
