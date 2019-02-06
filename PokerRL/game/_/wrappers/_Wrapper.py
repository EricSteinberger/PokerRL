# Copyright (c) 2019 Eric Steinberger


from PokerRL.game._.rl_env.base.PokerEnv import PokerEnv


class Wrapper:

    def __init__(self, env, env_bldr_that_built_me):
        """
        Args:
            env (PokerEnv subclass instance):   The environment instance to be wrapped

            env_bldr_that_built_me:          EnvWrappers should only be created by EnvBuilders. The EnvBuilder
                                                instance passes ""self"" as the value for this argument.
        """
        assert issubclass(type(env), PokerEnv)
        self.env = env
        self.env_bldr = env_bldr_that_built_me

    def _return_obs(self, rew_for_all_players, done, info, env_obs=None):
        return self.get_current_obs(env_obs=env_obs), rew_for_all_players, done, info

    # _______________________________ directly interact with the env inside the wrapper ________________________________
    def step(self, action):
        """
        Steps the environment from an action of the natural action representation to the environment.

        Returns:
            obs, reward, done, info
        """
        env_obs, rew_for_all_players, done, info = self.env.step(action)
        self._pushback(env_obs)
        return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

    def step_from_processed_tuple(self, action):
        """
        Steps the environment from a tuple (action, num_chips,).

        Returns:
            obs, reward, done, info
        """
        env_obs, rew_for_all_players, done, info = self.env.step_from_processed_tuple(action)
        self._pushback(env_obs)
        return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

    def step_raise_pot_frac(self, pot_frac):
        """
        Steps the environment from a fractional pot raise instead of an action as usually specified.

        Returns:
            obs, reward, done, info
        """
        env_obs, rew_for_all_players, done, info = self.env.step_raise_pot_frac(pot_frac=pot_frac)
        self._pushback(env_obs)
        return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

    def reset(self, deck_state_dict=None):
        env_obs, rew_for_all_players, done, info = self.env.reset(deck_state_dict=deck_state_dict)
        self._reset_state()
        self._pushback(env_obs)
        return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

    # ___________________________________________________ To Override __________________________________________________
    def state_dict(self):
        return {"env": self.env.state_dict()}

    def load_state_dict(self, state_dict):
        self.env.load_state_dict(state_dict["env"])

    def _reset_state(self):
        raise NotImplementedError

    def _pushback(self, env_obs):
        """
        Processes a transition in the wrapper. This should be called before every action by any agent.
        """
        raise NotImplementedError

    def get_current_obs(self, env_obs):
        raise NotImplementedError

    def print_obs(self, wrapped_obs):
        raise NotImplementedError

    def set_to_public_tree_node_state(self, node):
        raise NotImplementedError
