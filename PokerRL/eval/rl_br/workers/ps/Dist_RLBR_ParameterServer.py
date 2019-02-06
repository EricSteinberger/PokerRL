# Copyright (c) 2019 Eric Steinberger


import ray

from PokerRL.eval.rl_br.workers.ps.Local_RLBR_ParameterServer import Local_RLBR_ParameterServer as _LocalPS


@ray.remote(num_cpus=1)
class Dist_RLBR_ParameterServer(_LocalPS):

    def __init__(self, t_prof, chief_handle):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle)
