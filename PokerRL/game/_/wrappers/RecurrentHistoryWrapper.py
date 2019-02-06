# Copyright (c) 2019 Eric Steinberger


import copy

import numpy as np

from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
from PokerRL.game._.wrappers._Wrapper import Wrapper


class RecurrentHistoryWrapper(Wrapper):
    """
    Stores a sequence of current observations of each timestep thus having perfect recall.
    This wrapper is suitable for recurrent NN architectures.
    """

    def __init__(self, env, env_bldr_that_built_me):
        super().__init__(env=env, env_bldr_that_built_me=env_bldr_that_built_me)
        self.invert_history_order = env_bldr_that_built_me.invert_history_order
        self._list_of_obs_this_episode = None

    def _reset_state(self, **kwargs):
        self._list_of_obs_this_episode = []

    def _pushback(self, env_obs):
        if self.invert_history_order:
            self._list_of_obs_this_episode.insert(0, np.copy(env_obs))
        else:
            self._list_of_obs_this_episode.append(np.copy(env_obs))

    def print_obs(self, wrapped_obs):
        assert isinstance(wrapped_obs, np.ndarray)
        print("*****************************************************************************************************")
        print("*****************************************************************************************************")
        print("*****************************************************************************************************")
        print()
        print("________________________________________ OBSERVATION HISTORY ________________________________________")
        print()

        for o in wrapped_obs:
            self.env.print_obs(o)

    def get_current_obs(self, env_obs=None):
        return np.array(self._list_of_obs_this_episode, dtype=np.float32)

    def state_dict(self):
        return {
            "base": super().state_dict(),
            "obs_seq": copy.deepcopy(self._list_of_obs_this_episode)
        }

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict=state_dict["base"])
        self._list_of_obs_this_episode = copy.deepcopy(state_dict["obs_seq"])

    def set_to_public_tree_node_state(self, node):
        """
        Sets the internal env wrapper to the state ""node"" is in.

        Args:
            node:                   Any node (of any type) in a PublicTree instance.

        """
        state_seq = []  # will be sorted. [0] is root.

        def add(_node):
            if _node is not None:
                if _node.p_id_acting_next != _node.tree.CHANCE_ID:
                    self.env.load_state_dict(_node.env_state)
                    state_seq.insert(0, self.env.get_current_obs(is_terminal=False))
                add(_node.parent)

        add(node)  # will step from node to parent to parent of parent... to root.
        self.reset()
        self._reset_state()  # from .reset() the first obs is in by default

        # fake step in the internal env to gain current rr state
        for obs in state_seq:
            self._pushback(env_obs=obs)

        assert len(state_seq) == len(self._list_of_obs_this_episode)

        self.env.load_state_dict(node.env_state, blank_private_info=True)
        assert np.array_equal(node.env_state[EnvDictIdxs.board_2d], self.env.board)
