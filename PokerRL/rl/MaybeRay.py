# Copyright (c) 2019 Eric Steinberger
import numpy as np
import psutil
import torch

# not being able to import ray is okay if you don't run distributed!
try:
    import ray
    import ray.utils
except ModuleNotFoundError:
    pass


class MaybeRay:
    """
    These ray wrapper functions serve to unify the distributed and local function call syntax. If you call them,
    but do not run distributed, everything will just be tunneled through and ray will not even be imported,
    but the same code can be run distributed just by enabling a single flag ""runs_distributed"".
    """

    def __init__(self, runs_distributed=True, runs_cluster=False):
        """
        Args:k
            runs_distributed (bool): all ray calls are skipped if this is false
        """
        if runs_cluster:
            assert runs_distributed
        self.runs_distributed = runs_distributed
        self.runs_cluster = runs_cluster

    # __________________________________________________ ray wrapper ___________________________________________________
    def init_cluster(self, redis_address):
        assert self.runs_cluster
        ray.init(
            redis_address=redis_address,
            redis_max_memory=min(10 ** 10, int(psutil.virtual_memory().total * 0.1)),
            object_store_memory=min(2 * (10 ** 10), int(psutil.virtual_memory().total * 0.4)),
        )

    def init_local(self):
        if self.runs_distributed:
            ray.init(
                redis_max_memory=min(10 ** 10, int(psutil.virtual_memory().total * 0.1)),
                object_store_memory=min(2 * (10 ** 10), int(psutil.virtual_memory().total * 0.4)),
            )

    def get(self, obj):
        if self.runs_distributed:
            return ray.get(obj)
        return obj

    def put(self, obj):
        if self.runs_distributed:
            return ray.put(obj)
        return obj

    def remote(self, fn, *args):
        if self.runs_distributed:
            return fn.remote(*args)
        return fn(*args)

    def create_worker(self, cls, *args):
        if self.runs_distributed:
            return cls.remote(*args)
        return cls(*args)

    def wait(self, _list, num_returns=None, timeout=None, return_not_ready=False):
        """
        Args:
            _list (list:                    list of object ids to wait for
            num_returns (int):              if None: wait for all; if any number waits for that number
            timeout (int):                  Optional. If specified, waits only for ""timeout"" milliseconds
            return_not_ready (bool):        if True: returns tuple (rdy, not_rdy). If false: returns rdy
        """
        if self.runs_distributed:
            num_returns = len(_list) if num_returns is None else num_returns
            rdy, not_rdy = ray.wait(_list, num_returns=num_returns, timeout=timeout)
            if return_not_ready:
                return rdy, not_rdy
            return rdy

        if return_not_ready:
            return _list, []
        return _list

    # __________________________________________________ util tools ____________________________________________________
    def state_dict_to_numpy(self, _dict):
        """ if local, skip torch <--> numpy """
        if self.runs_distributed:
            np_dict = {}
            for k in list(_dict.keys()):
                if isinstance(_dict[k], np.ndarray):
                    np_dict[k] = _dict[k]
                else:
                    np_dict[k] = _dict[k].cpu().numpy()

            return np_dict

        return _dict

    def state_dict_to_torch(self, _dict, device):
        """ if local, skip torch <--> numpy """

        new_dict = {}
        if self.runs_distributed:
            for k in list(_dict.keys()):
                if isinstance(_dict[k], np.ndarray):
                    new_dict[k] = torch.from_numpy(_dict[k])
                else:
                    new_dict[k] = _dict[k]

                new_dict[k] = new_dict[k].to(device)
        else:
            for k in list(_dict.keys()):
                new_dict[k] = _dict[k].to(device)

        return new_dict

    def grads_to_numpy(self, g):
        if self.runs_distributed:  # if local, skip torch <--> numpy
            if g is None:
                return None
            for name in list(g.keys()):
                if g[name] is not None:
                    g[name] = g[name].cpu().numpy()
        return g

    def grads_to_torch(self, g, device):
        if self.runs_distributed:  # if local, skip torch <--> numpy
            if g is None:
                return None
            for k in list(g.keys()):
                if g[k] is not None:
                    g[k] = torch.from_numpy(g[k])

        for k in list(g.keys()):
            if g[k] is not None:
                g[k] = g[k].to(device)

        return g
