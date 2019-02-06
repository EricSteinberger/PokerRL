# Copyright (c) 2019 Eric Steinberger


"""
Wraps the local H2H master in a ray actor (i.e. worker) to be placed on any (one) machine in the cluster.
The H2H computation itself is not distributed.
"""

import ray
import torch

from PokerRL.eval.head_to_head.LocalHead2HeadMaster import LocalHead2HeadMaster as LocalEvalHead2HeadMaster


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class DistHead2HeadMaster(LocalEvalHead2HeadMaster):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        LocalEvalHead2HeadMaster.__init__(self, t_prof=t_prof, chief_handle=chief_handle, eval_agent_cls=eval_agent_cls)
