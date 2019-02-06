# Copyright (c) 2019 Eric Steinberger


import torch

from PokerRL.rl.buffers._circular_base import CircularBufferBase


class CircularBufferFLAT(CircularBufferBase):

    def __init__(self, env_bldr, max_size):
        super().__init__(env_bldr=env_bldr, max_size=max_size)

        self.storage_device = torch.device("cpu")

        self._pub_obs_t_buffer = None
        self._action_t_buffer = None
        self._range_idx_buffer = None
        self._reward_buffer = None
        self._pub_obs_tp1_buffer = None
        self._legal_action_mask_t_buffer = None
        self._legal_action_mask_tp1_buffer = None
        self._done_buffer = None

        self.reset()

    @property
    def max_size(self):
        return self._max_size

    @property
    def size(self):
        return self._size

    def add_step(self, pub_obs_t, a_t, range_idx, r_t, legal_action_mask_t, pub_obs_tp1, done_tp1,
                 legal_action_mask_tp1):
        self._pub_obs_t_buffer[self._top] = torch.from_numpy(pub_obs_t).to(device=self.storage_device)
        self._action_t_buffer[self._top] = a_t
        self._range_idx_buffer[self._top] = range_idx
        self._reward_buffer[self._top] = r_t
        self._pub_obs_tp1_buffer[self._top] = torch.from_numpy(pub_obs_tp1).to(device=self.storage_device)
        self._legal_action_mask_tp1_buffer[self._top] = torch.from_numpy(legal_action_mask_tp1).to(
            device=self.storage_device)
        self._legal_action_mask_t_buffer[self._top] = torch.from_numpy(legal_action_mask_t).to(
            device=self.storage_device)
        self._done_buffer[self._top] = float(done_tp1)

        if self._size < self._max_size:
            self._size += 1

        self._top = (self._top + 1) % self._max_size

    def sample(self, device, batch_size):
        indices = torch.randint(0, self._size, (batch_size,), dtype=torch.long, device=device)

        return self._pub_obs_t_buffer[indices].to(device), \
               self._action_t_buffer[indices].to(device), \
               self._range_idx_buffer[indices].to(device), \
               self._legal_action_mask_t_buffer[indices].to(device), \
               self._reward_buffer[indices].to(device), \
               self._pub_obs_tp1_buffer[indices].to(device), \
               self._legal_action_mask_tp1_buffer[indices].to(device), \
               self._done_buffer[indices].to(device)

    def state_dict(self):
        return {
            "pub_obs_t_buffer": self._pub_obs_t_buffer.cpu().clone(),
            "action_t_buffer": self._action_t_buffer.cpu().clone(),
            "range_idx_buffer": self._range_idx_buffer.cpu().clone(),
            "reward_buffer": self._reward_buffer.cpu().clone(),
            "pub_obs_tp1_buffer": self._pub_obs_tp1_buffer.cpu().clone(),
            "legal_action_mask_t_buffer": self._legal_action_mask_t_buffer.cpu().clone(),
            "legal_action_mask_tp1_buffer": self._legal_action_mask_tp1_buffer.cpu().clone(),
            "done_buffer": self._done_buffer.cpu().clone(),
            "size": self._size,
            "top": self._top
        }

    def load_state_dict(self, state):
        self._pub_obs_t_buffer = state["pub_obs_t_buffer"]
        self._action_t_buffer = state["action_t_buffer"]
        self._range_idx_buffer = state["range_idx_buffer"]
        self._reward_buffer = state["reward_buffer"]
        self._pub_obs_tp1_buffer = state["pub_obs_tp1_buffer"]
        self._legal_action_mask_t_buffer = state["legal_action_mask_t_buffer"]
        self._legal_action_mask_tp1_buffer = state["legal_action_mask_tp1_buffer"]
        self._done_buffer = state["done_buffer"]
        self._size = state["size"]
        self._top = state["top"]

    def reset(self):
        super().reset()
        self._pub_obs_t_buffer = torch.empty(size=(self._max_size, self._env_bldr.pub_obs_size), dtype=torch.float32,
                                             device=self.storage_device)
        self._action_t_buffer = torch.empty(size=(self._max_size,), dtype=torch.long, device=self.storage_device)
        self._range_idx_buffer = torch.empty(size=(self._max_size,), dtype=torch.long, device=self.storage_device)
        self._reward_buffer = torch.empty(size=(self._max_size,), dtype=torch.float32, device=self.storage_device)
        self._pub_obs_tp1_buffer = torch.empty(size=(self._max_size, self._env_bldr.pub_obs_size),
                                               dtype=torch.float32, device=self.storage_device)
        self._legal_action_mask_t_buffer = torch.ByteTensor(size=(self._max_size, self._env_bldr.N_ACTIONS),
                                                            device=self.storage_device)
        self._legal_action_mask_tp1_buffer = torch.ByteTensor(size=(self._max_size, self._env_bldr.N_ACTIONS),
                                                              device=self.storage_device)
        self._done_buffer = torch.empty(size=(self._max_size,), dtype=torch.float32, device=self.storage_device)
