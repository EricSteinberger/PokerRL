# Copyright (c) 2019 Eric Steinberger


from os.path import join as ospj

from PokerRL.rl.MaybeRay import MaybeRay
from PokerRL.util import file_util


class WorkerBase:
    """
    The base-class for every worker that can run locally and on a cluster with the same code.
    """

    def __init__(self, t_prof):
        self._t_prof = t_prof

        if t_prof is None:
            self._ray = MaybeRay(runs_distributed=False, runs_cluster=False)
        else:
            self._ray = MaybeRay(runs_distributed=t_prof.DISTRIBUTED, runs_cluster=t_prof.CLUSTER)

    def checkpoint(self, curr_step):
        """
        Override if worker has state to store/load for checkpoints
        """
        pass

    def load_checkpoint(self, name_to_load, step):
        """
        Override if worker has state to store/load for checkpoints
        """
        pass

    def _get_checkpoint_file_path(self, name, step, cls, worker_id):
        path = ospj(self._t_prof.path_checkpoint, str(name), str(step))
        file_util.create_dir_if_not_exist(path)
        return ospj(path, cls.__name__ + "_" + str(worker_id) + ".pkl")
