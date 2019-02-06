# Copyright (c) 2019 Eric Steinberger


"""
Wraps the local BR master in a ray actor to be placed on any (one) machine in the cluster. The BR computation itself
is not distributed.
"""

import ray
import torch

from PokerRL.eval.br.LocalBRMaster import LocalBRMaster as LocalEvalBRMaster


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class DistBRMaster(LocalEvalBRMaster):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        LocalEvalBRMaster.__init__(self, t_prof=t_prof, chief_handle=chief_handle, eval_agent_cls=eval_agent_cls)
