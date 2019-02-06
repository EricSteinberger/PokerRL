# Copyright (c) 2019 Eric Steinberger


class _PokerEnvArgs:

    def __init__(self,
                 n_seats,
                 starting_stack_sizes_list=None,
                 stack_randomization_range=(0, 0),
                 scale_rewards=True,
                 use_simplified_headsup_obs=True,
                 return_pre_transition_state_in_info=False,
                 *args,
                 **kwargs,
                 ):
        """
        Args:
            n_seats (int):                      The number of players in the game

            starting_stack_sizes_list (list):   An integer for each player, specifying the starting stack size.

            stack_randomization_range (tuple): (min_delta, max_delta). This will be added to the specified stack sizes
                                                Stack sizes are going to be subject to random offsets in the set range
                                                each episode. This only applies in evaluation mode of the environment.
                                                To not use this feature, pass (0,0).

            scale_rewards (bool):               Whether to scale rewards or not

            use_simplified_headsup_obs (bool):  Whether HU (i.e. 1v1) envs should have an obs without game aspects only
                                                relevant in 3+ player games (e.g. side-pots).

            return_pre_transition_state_in_info (bool):
                                                Whether the environment shall return certain additional information
        """
        self.n_seats = n_seats

        if starting_stack_sizes_list is None:
            self.starting_stack_sizes_list = [None for _ in range(n_seats)]
        else:
            self.starting_stack_sizes_list = starting_stack_sizes_list
        self.stack_randomization_range = stack_randomization_range
        self.scale_rewards = scale_rewards
        self.use_simplified_headsup_obs = use_simplified_headsup_obs
        self.RETURN_PRE_TRANSITION_STATE_IN_INFO = return_pre_transition_state_in_info


class NoLimitPokerEnvArgs(_PokerEnvArgs):
    """
    Args to any game that inherits from NoLimitPokerEnv
    """

    def __init__(self,
                 n_seats,
                 starting_stack_sizes_list=None,
                 stack_randomization_range=(0, 0),
                 scale_rewards=True,
                 use_simplified_headsup_obs=True,
                 return_pre_transition_state_in_info=False,
                 *args,
                 **kwargs):
        super().__init__(n_seats=n_seats,
                         starting_stack_sizes_list=starting_stack_sizes_list,
                         stack_randomization_range=stack_randomization_range,
                         scale_rewards=scale_rewards,
                         use_simplified_headsup_obs=use_simplified_headsup_obs,
                         return_pre_transition_state_in_info=return_pre_transition_state_in_info,
                         *args, **kwargs)
        self.N_ACTIONS = 3


class DiscretizedPokerEnvArgs(_PokerEnvArgs):
    """
    Args to any game that inherits from DiscretizedPokerEnv.
    """

    def __init__(self,
                 n_seats,
                 bet_sizes_list_as_frac_of_pot,
                 starting_stack_sizes_list=None,
                 stack_randomization_range=(0, 0),
                 uniform_action_interpolation=False,
                 use_simplified_headsup_obs=True,
                 scale_rewards=True,
                 return_pre_transition_state_in_info=False,
                 *args, **kwargs):
        """
        Args:
            bet_sizes_list_as_frac_of_pot (list):   list of allowed bet sizes in fractions of current pot.
                                                    e.g. [0.1, 0.3, 0.7, 1, 1.5]

            uniform_action_interpolation (bool):    In discrete poker envs, a finite number of raise sizes is defined.
                                                    If, for instance, the bet sizes are [0.1, 0.3, 0.7, 1, 1.5],
                                                    the agent doesn't bet these exact amounts, but uniformly sampled
                                                    amounts between the selected bet size, and the next bigger and
                                                    smaller one, if this argument is set to True.

            for info on all other arguments refer to docs in PokerEnvArgs' init function
        """
        super().__init__(n_seats=n_seats,
                         starting_stack_sizes_list=starting_stack_sizes_list,
                         stack_randomization_range=stack_randomization_range,
                         scale_rewards=scale_rewards,
                         use_simplified_headsup_obs=use_simplified_headsup_obs,
                         return_pre_transition_state_in_info=return_pre_transition_state_in_info,
                         *args, **kwargs)
        self.bet_sizes_list_as_frac_of_pot = bet_sizes_list_as_frac_of_pot
        self.uniform_action_interpolation = uniform_action_interpolation
        self.N_ACTIONS = len(self.bet_sizes_list_as_frac_of_pot) + 2  # +2 is for FOLD and CHECK/CALL.


class LimitPokerEnvArgs(_PokerEnvArgs):
    """
    Args to any game that inherits from LimitPokerEnv
    """

    def __init__(self,
                 n_seats,
                 starting_stack_sizes_list=None,
                 stack_randomization_range=(0, 0),
                 use_simplified_headsup_obs=True,
                 scale_rewards=True,
                 return_pre_transition_state_in_info=False,
                 *args, **kwargs):
        super().__init__(n_seats=n_seats,
                         starting_stack_sizes_list=starting_stack_sizes_list,
                         stack_randomization_range=stack_randomization_range,
                         scale_rewards=scale_rewards,
                         use_simplified_headsup_obs=use_simplified_headsup_obs,
                         return_pre_transition_state_in_info=return_pre_transition_state_in_info,
                         *args, **kwargs)
        self.N_ACTIONS = 3
