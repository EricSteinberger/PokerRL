# Copyright (c) 2019 Eric Steinberger


import ray

from PokerRL.eval.rl_br.workers.la.Local_RLBR_LearnerActor import Local_RLBR_LearnerActor as _LocalLA


@ray.remote(num_cpus=1)
class Dist_RLBR_LearnerActor(_LocalLA):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle, eval_agent_cls=eval_agent_cls)
