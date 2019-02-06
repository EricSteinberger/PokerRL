# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.rl import rl_util
from PokerRL.rl.buffers._circular_base import BRMemorySaverBase


class BRMemorySaverFLAT(BRMemorySaverBase):
    """ Interface for correct BR reward storing """

    def __init__(self, env_bldr, buffer):
        super().__init__(env_bldr=env_bldr, buffer=buffer)
        self._range_idx = None

    def add_terminal(self,
                     reward_p,
                     terminal_obs,
                     ):

        if self._intermediate_memory.is_level_1():
            self._buffer.add_step(pub_obs_t=np.copy(self._intermediate_memory.obs_t),
                                  a_t=self._intermediate_memory.action,
                                  range_idx=self._range_idx,
                                  r_t=reward_p,
                                  legal_action_mask_t=rl_util.get_legal_action_mask_np(
                                      n_actions=self._env_bldr.N_ACTIONS,
                                      legal_actions_list=self._intermediate_memory.legal_actions_list_t),

                                  pub_obs_tp1=np.copy(terminal_obs),
                                  done_tp1=True,
                                  legal_action_mask_tp1=rl_util.get_legal_action_mask_np(
                                      n_actions=self._env_bldr.N_ACTIONS,
                                      legal_actions_list=self._intermediate_memory.legal_actions_list_tp1)
                                  )

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
        self._buffer.add_step(pub_obs_t=np.copy(self._intermediate_memory.obs_t),
                              a_t=self._intermediate_memory.action,
                              range_idx=self._range_idx,
                              r_t=0.0,
                              legal_action_mask_t=rl_util.get_legal_action_mask_np(
                                  n_actions=self._env_bldr.N_ACTIONS,
                                  legal_actions_list=self._intermediate_memory.legal_actions_list_t),

                              pub_obs_tp1=np.copy(self._intermediate_memory.obs_tp1),
                              done_tp1=False,
                              legal_action_mask_tp1=rl_util.get_legal_action_mask_np(
                                  n_actions=self._env_bldr.N_ACTIONS,
                                  legal_actions_list=self._intermediate_memory.legal_actions_list_tp1))

    def reset(self, range_idx):
        """ Call with env reset """
        self._range_idx = range_idx
        self._intermediate_memory.reset()