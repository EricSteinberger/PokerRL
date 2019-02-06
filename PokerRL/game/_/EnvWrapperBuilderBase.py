# Copyright (c) 2019 Eric Steinberger


import copy


class EnvWrapperBuilderBase:
    WRAPPER_CLS = NotImplementedError

    def __init__(self,
                 env_cls,
                 env_args,
                 ):
        """
        Args:
            env_cls (PokerEnv subclass):    Any PokerEnv subclass (not an instance)

            env_args:                       An instance of either PokerEnvArgs or DiscretePokerEnv, depending on what
                                            type env_cls is
        """
        self.env_cls = env_cls
        self.rules = env_cls.RULES
        self.env_args = env_args

        self.lut_holder = env_cls.get_lut_holder()

        # Info about the env
        self.N_SEATS = env_args.n_seats
        self.N_ACTIONS = env_args.N_ACTIONS
        self.pub_obs_size = self._get_num_public_observation_features()
        self.priv_obs_size = self._get_num_private_observation_features()
        self.complete_obs_size = self.pub_obs_size + self.priv_obs_size
        self.obs_board_idxs, self.obs_players_idxs, self.obs_table_state_idxs = self._get_obs_parts_idxs()
        self.obs_size_board = len(self.obs_board_idxs)
        self.obs_size_player_info_each = len(self.obs_players_idxs[0])
        self.obs_size_table_state = len(self.obs_table_state_idxs)

    def get_new_env(self, is_evaluating, stack_size=None):
        """
        Args:
            is_evaluating (bool):       if True, no stack-size randomization is applied to the environment.

            stack_size (list):          Optional.
                                        list of n_seats ints. if provided: the environment will use these stack-
                                        sizes as the default starting point. If not provided, the env will use
                                        the standard stack sizes given by the EnvBuilder instance this is called in.

        Returns:
            PokerEnv subclass instance
        """

        args = copy.deepcopy(self.env_args)
        if stack_size is not None:
            assert isinstance(stack_size, list)
            args.starting_stack_sizes_list = copy.deepcopy(stack_size)

        return self.env_cls(env_args=args, lut_holder=self.lut_holder, is_evaluating=is_evaluating)

    def get_new_wrapper(self, is_evaluating, init_from_env=None, stack_size=None):
        """
        Args:
            init_from_env:              Optional.
                                        An instance of the same PokerEnv subclass to be created with this wrapper.
                                        Its deck, etc. is used to initialize the one to the exact same state and
                                        future randomness.

            is_evaluating (bool):       if True, no stack-size randomization is applied to the environment.

            stack_size (int):           Optional.
                                        list of n_seats ints. if provided: the environment will use these stack-
                                        sizes as the default starting point. If not provided, the env will use
                                        the standard stack sizes given by the EnvBuilder instance this is called in.

        Returns:
            EnvWrapper subclass instance
        """
        if init_from_env is None:
            env = self.get_new_env(is_evaluating=is_evaluating, stack_size=stack_size)
        else:
            env = init_from_env
        return self.WRAPPER_CLS(env=env, env_bldr_that_built_me=self)

    def _get_num_public_observation_features(self):
        """ Can be overridden if needed.

        Return the number of features (potentially per timestep) of the wrapped observation.
        This is only run once, so it is ok, if it is not efficient to evaluate this.
        """
        _env = self.env_cls(env_args=self.env_args, lut_holder=self.lut_holder, is_evaluating=True)
        return _env.observation_space.shape[0]

    def _get_num_private_observation_features(self):
        """
        Can be overridden if needed.
        Return the number of features (potentially per timestep) of the vector that will be appended to the public obs
        to resemble the private obs.
        This is only run once, so it is ok, if it is not efficient to evaluate this.
        """
        return (self.rules.N_SUITS + self.rules.N_RANKS) * self.rules.N_HOLE_CARDS

    def _get_obs_parts_idxs(self):
        """
        Override if your wrapper appends something in front of the env's original obs.
        Returns:

        """
        _env = self.env_cls(env_args=self.env_args, lut_holder=self.lut_holder, is_evaluating=True)
        return \
            _env.obs_parts_idxs_dict["board"], \
            _env.obs_parts_idxs_dict["players"], \
            _env.obs_parts_idxs_dict["table_state"]
