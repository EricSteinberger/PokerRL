# Copyright (c) 2019 Eric Steinberger


import copy
import os
from os.path import join as ospj

import torch


class TrainingProfileBase:
    """
    A TrainingProfile should hold hyperparameters and an for a run of an algorithm.
    """

    def __init__(self,

                 # --- general
                 name,
                 log_verbose,
                 log_export_freq,
                 checkpoint_freq,
                 eval_agent_export_freq,

                 # --- env
                 game_cls,
                 env_bldr_cls,
                 start_chips,

                 # --- Evaluation
                 eval_modes_of_algo,
                 eval_stack_sizes,

                 module_args,

                 # --- Computing
                 path_data=None,
                 local_crayon_server_docker_address="localhost",
                 device_inference="cpu",
                 DISTRIBUTED=False,
                 CLUSTER=False,
                 DEBUGGING=False,

                 # --- Only relevant if running distributed
                 redis_head_adr=None,  # (str) IP under which the ray redis server can be reached
                 ):
        """
        Args:
            name (str):                             Under this name all logs, data, and checkpoints will appear.
            log_verbose (bool):                     Whether the program shall log detailed in Tensorboard.
            log_export_freq:                        Every X iterations, logs are pushed to the Crayon Docker container.
            checkpoint_freq (int):                  Every X iterations, make a recoverable copy of state of training.
            eval_agent_export_freq (int):        Every X iterations, an EvalAgent instance of the algo is exported.
            game_cls (PokerEnv subclass):           Class (not instance) to be trained in.
            env_bldr_cls (EnvBuilder subclass)      Class (not instance) to wrap the environment.
            start_chips (int):                      Standard stack size to initialize all players with.
            eval_modes_of_algo (tuple):             Tuple of algo-specific EvalAgent's eval modes
            eval_stack_sizes (tuple):               Tuple of lists of ints. if None, defaults to what's used in
                                                    training_profile.env_bldr.
            module_args (dict):                     All modules or parts of algorithms may have their own args. These
                                                    are stored in seperate objects and accessible under a certain string
                                                    key in the ""module_args"" dict.
            path_data:                              path to store data (e.g. checkpoints) the algorithm generates in.
                                                    If None, we will store data in a folder we create in your home dir.
            local_crayon_server_docker_address:     ip of crayon docker container (default: localhost)
            device_inference:                       "cpu" or "cuda". This device will be used for batched NN inference
            DISTRIBUTED (bool):                     Whether ray should be used at all.
            CLUSTER:                                requires "DISTRIBUTED==True".
                                                    If True, runs on many machines, if False, runs on local CPUs/GPUs.
            DEBUGGING (bool):                       Whether to use assert statements for debugging
            redis_head_adr:                         Only applicable if "CLUSTER==True". IP address under which the ray
                                                    head can be found.

        """

        # Assert basic modules were passed
        assert "env" in module_args

        # t_prof
        self.name = name
        self.log_verbose = log_verbose
        self.log_export_freq = log_export_freq
        self.checkpoint_freq = checkpoint_freq
        self.eval_agent_export_freq = eval_agent_export_freq

        self.module_args = module_args

        if CLUSTER:
            if redis_head_adr:
                self.redis_head_adr = redis_head_adr
            else:
                from ray import services

                self.redis_head_adr = services.get_node_ip_address() + ":6379"
        self.local_crayon_server_docker_address = local_crayon_server_docker_address

        self.DISTRIBUTED = DISTRIBUTED or CLUSTER
        self.CLUSTER = CLUSTER
        self.DEBUGGING = DEBUGGING
        self.HAVE_GPU = torch.cuda.is_available()

        self.n_seats = self.module_args["env"].n_seats

        # Eval
        self.eval_modes_of_algo = eval_modes_of_algo
        if eval_stack_sizes is None:
            if start_chips is None:
                self.eval_stack_sizes = [[game_cls.DEFAULT_STACK_SIZE for _ in range(self.n_seats)]]
            else:
                self.eval_stack_sizes = [copy.deepcopy(self.module_args["env"].starting_stack_sizes_list)]
        else:
            assert isinstance(eval_stack_sizes, tuple)
            assert isinstance(eval_stack_sizes[0], list)
            self.eval_stack_sizes = list(eval_stack_sizes)

        self.game_cls_str = game_cls.__name__
        self.env_builder_cls_str = env_bldr_cls.__name__

        assert isinstance(device_inference, str), "Please pass a string (either 'cpu' or 'cuda')!"
        self.device_inference = torch.device(device_inference)

        # Paths
        def get_root_path():
            return "C:\\" if os.name == 'nt' else os.path.expanduser('~/')

        self._data_path = path_data if path_data is not None else os.path.join(get_root_path(), "poker_ai_data")
        self.path_agent_export_storage = ospj(self._data_path, "eval_agent")
        self.path_log_storage = ospj(self._data_path, "logs")
        self.path_checkpoint = ospj(self._data_path, "checkpoint")
        self.path_trainingprofiles = ospj(self._data_path, "TrainingProfiles")

        for p in [self._data_path,
                  self.path_agent_export_storage,
                  self.path_log_storage,
                  self.path_checkpoint,
                  self.path_trainingprofiles,
                  ]:
            if (not os.path.exists(p)) and (not os.path.isfile(p)):
                os.makedirs(p)
