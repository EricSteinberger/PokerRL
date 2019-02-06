# Copyright (c) 2019 Eric Steinberger


from PokerRL.rl.MaybeRay import MaybeRay


class HighLevelAlgoBase:
    """
    A HighLevelAlgo should encapsulate the high-level functionality of an algorithm and should use a
    MaybeRay instance to be compatible with distributed as well as local runs.
    """

    def __init__(self, t_prof, la_handles, chief_handle):
        """

        Args:
            t_prof (TrainingProfile)
            la_handles (list)
            chief_handle (class or ray worker handle)

        """
        self._t_prof = t_prof
        self._ray = MaybeRay(runs_distributed=t_prof.DISTRIBUTED, runs_cluster=t_prof.CLUSTER)
        self._la_handles = la_handles
        self._chief_handle = chief_handle
