# Copyright (c) 2019 Eric Steinberger


"""
Wraps the local LBR master in a ray actor to be placed on any (one) machine in the cluster. This worker will then
manage distributed LBRWorkers to compute the local best response approximation faster in parallel.
"""
import ray

from PokerRL.eval.lbr.LocalLBRMaster import LocalLBRMaster as _LocalEvalLBRMaster


@ray.remote(num_cpus=1)
class DistLBRMaster(_LocalEvalLBRMaster):

    def __init__(self, t_prof, chief_handle):
        _LocalEvalLBRMaster.__init__(self, t_prof=t_prof, chief_handle=chief_handle)
