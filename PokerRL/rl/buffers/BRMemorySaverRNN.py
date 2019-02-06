# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.rl import rl_util
from PokerRL.rl.buffers._circular_base import BRMemorySaverBase


class _GameForBR:

    def __init__(self, range_idx):
        self._n_steps_in_game_memory = 0

        self._range_idx = range_idx

        self._obs_sequence = []

        self._obs_t_idxs_per_step = []
        self._obs_tp1_idxs_per_step = []

        self._action_buffer = []
        self._reward_buffer = []
        self._done_buffer = []
        self._legal_actions_mask_tp1_buffer = []
        self._legal_actions_mask_t_buffer = []

    @property
    def n_steps_in_game_memory(self):
        return self._n_steps_in_game_memory

    def add(self, o_t, a, legal_action_mask_t, rew, done, o_tp1, legal_action_mask_tp1):
        self._obs_t_idxs_per_step.append(o_t.shape[0])
        self._obs_tp1_idxs_per_step.append(o_tp1.shape[0])

        self._obs_sequence = np.copy(o_tp1)

        self._legal_actions_mask_tp1_buffer.append(np.copy(legal_action_mask_tp1))
        self._legal_actions_mask_t_buffer.append(np.copy(legal_action_mask_t))
        self._action_buffer.append(a)
        self._reward_buffer.append(rew)
        self._done_buffer.append(done)

        self._n_steps_in_game_memory += 1

    def sample(self):
        idx = np.random.randint(low=0, high=self._n_steps_in_game_memory)
        return {
            "o_t": self._obs_sequence[:self._obs_t_idxs_per_step[idx]],
            "o_tp1": self._obs_sequence[:self._obs_tp1_idxs_per_step[idx]],
            "mask_t": self._legal_actions_mask_t_buffer[idx],
            "mask_tp1": self._legal_actions_mask_tp1_buffer[idx],
            "a": self._action_buffer[idx],
            "rew": self._reward_buffer[idx],
            "done": self._done_buffer[idx],
            "range_idx": self._range_idx
        }


class BRMemorySaverRNN(BRMemorySaverBase):
    """ Interface for correct BR reward storing """

    def __init__(self, env_bldr, buffer):
        super().__init__(env_bldr=env_bldr, buffer=buffer)
        self._game_memory = None

    def add_terminal(self,
                     reward_p,
                     terminal_obs,
                     ):

        if self._intermediate_memory.is_level_1():
            self._game_memory.add(o_t=np.copy(self._intermediate_memory.obs_t),
                                  a=self._intermediate_memory.action,
                                  rew=reward_p,
                                  done=True,
                                  o_tp1=np.copy(terminal_obs),

                                  legal_action_mask_t=rl_util.get_legal_action_mask_np(
                                      n_actions=self._env_bldr.N_ACTIONS,
                                      legal_actions_list=self._intermediate_memory.legal_actions_list_t),

                                  legal_action_mask_tp1=rl_util.get_legal_action_mask_np(
                                      n_actions=self._env_bldr.N_ACTIONS,
                                      legal_actions_list=self._intermediate_memory.legal_actions_list_tp1)
                                  )

        if self._game_memory is not None:
            self._buffer.add_game(game=self._game_memory)
        self._intermediate_memory.reset()

    def add_non_terminal_experience(self,
                                    obs_t_before_acted,
                                    a_selected_t,
                                    legal_actions_list_t):

        self._intermediate_memory.add_experience(obs_t=obs_t_before_acted,
                                                 action_t=a_selected_t,
                                                 legal_actions_list_t=legal_actions_list_t)

        if self._intermediate_memory.is_complete():
            self._add_step_to_memory()
            self._intermediate_memory.step()

    def _add_step_to_memory(self):
        self._game_memory.add(o_t=np.copy(self._intermediate_memory.obs_t),
                              a=self._intermediate_memory.action,
                              rew=0.0,
                              done=False,
                              o_tp1=np.copy(self._intermediate_memory.obs_tp1),

                              legal_action_mask_t=rl_util.get_legal_action_mask_np(
                                  n_actions=self._env_bldr.N_ACTIONS,
                                  legal_actions_list=self._intermediate_memory.legal_actions_list_t),

                              legal_action_mask_tp1=rl_util.get_legal_action_mask_np(
                                  n_actions=self._env_bldr.N_ACTIONS,
                                  legal_actions_list=self._intermediate_memory.legal_actions_list_tp1))

    def reset(self, range_idx):
        """ Call with env reset """
        self._intermediate_memory.reset()
        self._game_memory = _GameForBR(range_idx=range_idx)