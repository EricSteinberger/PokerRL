# Copyright (c) 2019 Eric Steinberger


from PokerRL.game._.wrappers._Wrapper import Wrapper


class VanillaWrapper(Wrapper):
    """
    This wrapper doesn't track any history in any way and always just shows the current observation. Thus, it does not
    have perfect recall.

    This wrapper is suitable for feedforward NN architectures.
    """

    def __init__(self, env, env_bldr_that_built_me):
        super().__init__(env=env, env_bldr_that_built_me=env_bldr_that_built_me)

    def _reset_state(self, **kwargs):
        pass

    def _pushback(self, env_obs):
        pass

    def print_obs(self, wrapped_obs):
        self.env.print_obs(wrapped_obs)

    def get_current_obs(self, env_obs=None):
        if env_obs:
            return env_obs
        return self.env.get_current_obs()

    def state_dict(self):
        return {
            "base": super().state_dict(),
        }

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict=state_dict["base"])

    def set_to_public_tree_node_state(self, node):
        pass
