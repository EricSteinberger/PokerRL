# Copyright (c) 2019 Eric Steinberger


from PokerRL.rl import rl_util
from PokerRL.rl.MaybeRay import MaybeRay
from PokerRL.util.file_util import do_pickle, load_pickle


class EvalAgentBase:
    """
    This baseclass should be subclassed by each agent/algorithm type. It is used to wrap the agent with his own
    internal1 environment. It provides a standardized API for querying the agent for different things to the evaluators.
    If an algorithm employs different agents for each seat on the table, this class should wrap all of them in one.
    """
    ALL_MODES = NotImplementedError  # Override with list of all modes

    def __init__(self, t_prof, mode=None, device=None):
        """
        Args:
            t_prof (TrainingProfile):
            mode:                       Any mode your algorithm's eval agent can be evaluated in. Specify modes
                                        as class variables and pass one of them here. Can be changed later by calling
                                        .to_mode(new_mode) on this instance
            device (torch.device):      The device the eval agent shall live and act on.
        """
        self.t_prof = t_prof
        self.ray = MaybeRay(runs_distributed=t_prof.DISTRIBUTED, runs_cluster=t_prof.CLUSTER)
        self.env_bldr = rl_util.get_env_builder(t_prof=t_prof)

        self._internal_env_wrapper = self.env_bldr.get_new_wrapper(is_evaluating=True, stack_size=None)
        self._mode = mode

        if device is None:
            self.device = self.t_prof.device_inference
        else:
            self.device = device

    # __________________________________________________ Query agents __________________________________________________
    def get_a_probs_for_each_hand(self):
        """
        Returns:
            np.ndarray(RANGE_SIZE, N_ACTIONS): the action probabilities for each hand
        """
        raise NotImplementedError

    def get_a_probs(self):
        """
        Returns:
            np.ndarray(N_ACTIONS): action probs for hand currently held in current state
        """
        raise NotImplementedError

    def get_action(self, step_env=True, need_probs=False):
        """
        Args:
            step_env (bool):        Whether the internal env shall be stepped
            need_probs (bool):      Whether the action probabilities for all hands shall be returned too

        Returns:
            action,
            action probs for each hand (or None if not need_probs)
        """
        raise NotImplementedError

    def get_action_frac_tuple(self, step_env=True):
        """
        Args:
            step_env (bool):        Whether the internal env shall be stepped

        Returns:
            2-tuple:  ((FOLD CALL or RAISE), fraction)
        """
        raise NotImplementedError

    def state_dict(self):
        """ Override and keep base as one field! """
        return {
            "t_prof": self.t_prof,
            "mode": self._mode,
            "env": self._internal_env_wrapper.state_dict(),
            "agent": self._state_dict(),
        }

    def load_state_dict(self, state):
        self._internal_env_wrapper.load_state_dict(state["env"])
        self._mode = state["mode"]
        self._load_state_dict(state["agent"])

    def _state_dict(self):
        # Implement your agent's state_dict
        raise NotImplementedError

    def _load_state_dict(self, state):
        # Implement your agent's load_state_dict
        raise NotImplementedError

    def update_weights(self, weights_for_eval_agent):
        """
        Args:
            weights_for_eval_agent: Can be any algorithm-specific data; e.g. Neural Network parameters for the agent
        """
        raise NotImplementedError

    def can_compute_mode(self):
        """
        Returns:
            bool:                   Whether whatever condition is satisfied (e.g. for delayed CFR+ whether enough
                                    iterations have passed) to evaluate the algorithm with self._mode
        """
        raise NotImplementedError

    # _____________________________________________________ State ______________________________________________________
    def set_stack_size(self, stack_size):
        self._internal_env_wrapper.env.set_stack_size(stack_size=stack_size)

    def get_mode(self):
        return self._mode

    def set_mode(self, mode):
        assert mode in self.ALL_MODES
        self._mode = mode

    def set_env_wrapper(self, env_wrapper):
        self._internal_env_wrapper = env_wrapper

    def get_env_wrapper(self):
        return self._internal_env_wrapper

    def set_to_public_tree_node_state(self, node):
        self._internal_env_wrapper.set_to_public_tree_node_state(node=node)

    # __________________________________________________ Notifications _________________________________________________
    def notify_of_action(self, p_id_acted, action_he_did):
        assert self._internal_env_wrapper.env.current_player.seat_id == p_id_acted
        self._internal_env_wrapper.step(action=action_he_did)

    def notify_of_processed_tuple_action(self, p_id_acted, action_he_did):
        assert self._internal_env_wrapper.env.current_player.seat_id == p_id_acted
        self._internal_env_wrapper.step_from_processed_tuple(action=action_he_did)

    def notify_of_raise_frac_action(self, p_id_acted, frac):
        """ this fn is only useful to call if current_player wants to raise. Therefore it assumes that's the case. """
        assert self._internal_env_wrapper.env.current_player.seat_id == p_id_acted
        self._internal_env_wrapper.step_raise_pot_frac(pot_frac=frac)

    def notify_of_reset(self):
        self._internal_env_wrapper.reset()
        self._internal_env_wrapper._list_of_obs_this_episode = []  # from .reset() the first obs is in by default

    def reset(self, deck_state_dict=None):
        self._internal_env_wrapper.reset(deck_state_dict=deck_state_dict)

    # ___________________________________________________ Store State __________________________________________________
    def env_state_dict(self):
        return self._internal_env_wrapper.state_dict()

    def load_env_state_dict(self, state_dict):
        self._internal_env_wrapper.load_state_dict(state_dict)

    def store_to_disk(self, path, file_name):
        do_pickle(obj=self.state_dict(), path=path, file_name=file_name)

    @classmethod
    def load_from_disk(cls, path_to_eval_agent):
        state = load_pickle(path=path_to_eval_agent)

        eval_agent = cls(t_prof=state["t_prof"])
        eval_agent.load_state_dict(state=state)

        return eval_agent
