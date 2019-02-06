# Copyright (c) 2019 Eric Steinberger


"""
Wraps the local LBR worker in a ray actor to be placed on any machine in the cluster. You can spawn as many of these
as you want to accelerate the LBR computation; the EvalLBRMaster will manage them all.
"""

import ray
import torch

from PokerRL.eval.lbr.LocalLBRWorker import LocalLBRWorker as LocalEvalLBRWorker


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class DistLBRWorker(LocalEvalLBRWorker):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        LocalEvalLBRWorker.__init__(self, t_prof=t_prof, chief_handle=chief_handle, eval_agent_cls=eval_agent_cls)
