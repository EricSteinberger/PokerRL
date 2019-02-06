# Copyright (c) 2019 Eric Steinberger


import copy

import numpy as np


class CircularBufferBase:
    """
    self.games stores references to Game subclass objects. One Game instance might be referenced multiple times,
    depending on the number of steps that it contains. This is to keep equally likely sampling.
    """

    def __init__(self, env_bldr, max_size):
        self._env_bldr = env_bldr
        self._max_size = max_size
        self._size = None
        self._top = None

    def sample(self, device, batch_size):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError

    def reset(self):
        self._size = 0
        self._top = 0


class BRMemorySaverBase:
    """ Interface for correct BR reward storing """

    def __init__(self, env_bldr, buffer):
        self._env_bldr = env_bldr
        self._buffer = buffer
        self._intermediate_memory = _PlayerTimeStepMemory()

    def add_terminal(self,
                     reward_p,
                     terminal_obs,
                     ):
        raise NotImplementedError

    def add_non_terminal_experience(self,
                                    obs_t_before_acted,
                                    a_selected_t,
                                    legal_actions_list_t):
        raise NotImplementedError

    def _add_step_to_memory(self):
        raise NotImplementedError

    def reset(self, range_idx):
        raise NotImplementedError


class _PlayerTimeStepMemory:

    def __init__(self):
        self.obs_t = None
        self.obs_tp1 = None
        self.action = None
        self.action_tp1 = None
        self.legal_actions_list_t = None
        self.legal_actions_list_tp1 = None

    def add_experience(self, obs_t, action_t, legal_actions_list_t):
        """
        called when it is player's turn.
        """
        if self.obs_t is None:
            self.obs_t = np.copy(obs_t)
            self.action = action_t
            self.legal_actions_list_t = np.copy(legal_actions_list_t)
        elif self.obs_tp1 is None:
            self.obs_tp1 = np.copy(obs_t)
            self.action_tp1 = action_t
            self.legal_actions_list_tp1 = np.copy(legal_actions_list_t)
        else:
            raise BufferError("This should not happen.")

    def is_level_1(self):
        return self.obs_t is not None and self.obs_tp1 is None

    def is_complete(self):
        return self.obs_tp1 is not None

    def step(self):
        self.obs_t = np.copy(self.obs_tp1)
        self.obs_tp1 = None
        self.action = self.action_tp1
        self.action_tp1 = None
        self.legal_actions_list_t = copy.deepcopy(self.legal_actions_list_tp1)
        self.legal_actions_list_tp1 = None

    def reset(self):
        self.obs_t = None
        self.obs_tp1 = None
        self.action = None
        self.action_tp1 = None
        self.legal_actions_list_t = None
        self.legal_actions_list_tp1 = None
