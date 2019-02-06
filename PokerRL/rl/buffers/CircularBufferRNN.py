# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch

from PokerRL.rl.buffers._circular_base import CircularBufferBase


class CircularBufferRNN(CircularBufferBase):
    """
    self.games stores references to Game subclass objects. One Game instance might be referenced multiple times,
    depending on the number of steps that it contains. This is to keep equally likely sampling.
    """

    def __init__(self, env_bldr, max_size):
        super().__init__(env_bldr=env_bldr, max_size=max_size)

        self._games = None
        self.reset()

    @property
    def max_size(self):
        return self._max_size

    @property
    def size(self):
        return self._size

    def add_game(self, game):
        for _ in range(game.n_steps_in_game_memory):
            self._games[self._top] = game

            if self._size < self._max_size:
                self._size += 1

            self._top = (self._top + 1) % self._max_size

    def sample(self, device, batch_size):
        """
        Args:
            batch_size (int)
            device (torch.device)

        Returns:
            tuple
        """
        indices = np.random.randint(low=0, high=self._size, size=batch_size)

        samples = [self._games[i].sample() for i in indices]

        batch_legal_action_mask_tp1 = [sample["mask_tp1"] for sample in samples]
        batch_legal_action_mask_tp1 = torch.from_numpy(np.array(batch_legal_action_mask_tp1)).to(device=device)

        batch_legal_action_mask_t = [sample["mask_t"] for sample in samples]
        batch_legal_action_mask_t = torch.from_numpy(np.array(batch_legal_action_mask_t)).to(device=device)

        batch_action_t = [sample["a"] for sample in samples]
        batch_action_t = torch.tensor(batch_action_t, dtype=torch.long, device=device)

        batch_range_idx = [sample["range_idx"] for sample in samples]
        batch_range_idx = torch.from_numpy(np.array(batch_range_idx)).to(dtype=torch.long, device=device)

        batch_reward = [sample["rew"] for sample in samples]
        batch_reward = torch.from_numpy(np.array(batch_reward)).to(dtype=torch.float32, device=device)

        batch_done = [sample["done"] for sample in samples]
        batch_done = torch.tensor(batch_done, dtype=torch.float32, device=device)

        # obs will be further processed into a PackedSequence in the net.
        batch_pub_obs_t = [sample["o_t"] for sample in samples]
        batch_pub_obs_tp1 = [sample["o_tp1"] for sample in samples]

        return batch_pub_obs_t, \
               batch_action_t, \
               batch_range_idx, \
               batch_legal_action_mask_t, \
               batch_reward, \
               batch_pub_obs_tp1, \
               batch_legal_action_mask_tp1, \
               batch_done,

    def state_dict(self):
        return {
            "games": self._games,
            "size": self._size,
            "top": self._top
        }

    def load_state_dict(self, state):
        self._games = state["games"]
        self._size = state["size"]
        self._top = state["top"]

    def reset(self):
        super().reset()
        self._games = np.array([None for _ in range(self._max_size)], dtype=object)
