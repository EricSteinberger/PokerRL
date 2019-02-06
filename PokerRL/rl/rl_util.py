# Copyright (c) 2019 Eric Steinberger


"""
Utility functions for RL
"""
import numpy as np
import torch

from PokerRL.game.games import ALL_ENVS
from PokerRL.game.wrappers import ALL_BUILDERS


def polynomial_decay(base, const, counter, exponent, minimum=0):
    return minimum + ((base - minimum) / (1 + const * np.power(counter, exponent)))


def str_to_optim_cls(optim_string):
    if optim_string.lower() == "sgd":
        return torch.optim.SGD

    elif optim_string.lower() == "adam":
        def fn(parameters, lr):
            return torch.optim.Adam(parameters, lr=lr)

        return fn

    elif optim_string.lower() == "rms":
        def fn(parameters, lr):
            return torch.optim.RMSprop(parameters, lr=lr)

        return fn

    elif optim_string.lower() == "sgdmom":
        def fn(parameters, lr):
            return torch.optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True)

        return fn

    else:
        raise ValueError(optim_string)


def str_to_loss_cls(loss_str):
    if loss_str.lower() == "mse":
        return torch.nn.MSELoss()

    elif loss_str.lower() == "weighted_mse":
        return lambda y, trgt, w: torch.mean(w * ((y - trgt) ** 2))

    elif loss_str.lower() == "ce":
        return torch.nn.CrossEntropyLoss()

    elif loss_str.lower() == "smoothl1":
        return torch.nn.SmoothL1Loss()

    else:
        raise ValueError(loss_str)


def str_to_rnn_cls(rnn_str):
    if rnn_str.lower() == "lstm":
        return torch.nn.LSTM

    elif rnn_str.lower() == "gru":
        return torch.nn.GRU

    elif rnn_str.lower() == "vanilla":
        return torch.nn.RNN

    else:
        raise ValueError(rnn_str)


def get_env_cls_from_str(env_str):
    for e in ALL_ENVS:
        if env_str == e.__name__:
            return e
    raise ValueError(env_str, "is not registered or does not exist.")


def get_env_builder(t_prof):
    ENV_BUILDER = get_builder_from_str(t_prof.env_builder_cls_str)
    return ENV_BUILDER(env_cls=get_env_cls_from_str(t_prof.game_cls_str), env_args=t_prof.module_args["env"])


def get_builder_from_str(wrapper_str):
    for b in ALL_BUILDERS:
        if wrapper_str == b.__name__:
            return b
    raise ValueError(wrapper_str, "is not registered or does not exist.")


def get_legal_action_mask_torch(n_actions, legal_actions_list, device, dtype=torch.uint8):
    """
    Args:
        legal_actions_list (list):  List of legal actions as integers, where 0 is always FOLD, 1 is CHECK/CALL.
                                    2 is BET/RAISE for continuous PokerEnvs, and for DiscretePokerEnv subclasses,
                                    numbers greater than 1 are all the raise sizes.

        device (torch.device):      device the mask shall be put on

        dtype:                      dtype the mask shall have

    Returns:
        torch.Tensor:               a many-hot representation of the list of legal actions.
    """
    idxs = torch.LongTensor(legal_actions_list, device=device)
    mask = torch.zeros((n_actions), device=device, dtype=dtype)
    mask[idxs] = 1
    return mask


def batch_get_legal_action_mask_torch(n_actions, legal_actions_lists, device, dtype=torch.uint8):
    """

    Args:
        legal_actions_lists (list): List of lists. Each of the 2nd level lists contains legal actions as integers,
                                    where 0 is always FOLD, 1 is CHECK/CALL. 2 is BET/RAISE for continuous
                                    PokerEnvs, and for DiscretePokerEnv subclasses, numbers greater than 1 are all
                                    the raise sizes.

        device (torch.device):      device the mask shall be put on

        dtype:                      dtype the mask shall have

    Returns:
        torch.Tensor:               a many-hot representation of the list of legal actions.

    """
    assert isinstance(legal_actions_lists[0], list), "need list of lists of legal actions (as ints)!"

    mask = torch.zeros((len(legal_actions_lists), n_actions,), device=device, dtype=dtype)
    for i, legal_action_list in enumerate(legal_actions_lists):
        mask[i, torch.LongTensor(legal_action_list, device=device)] = 1
    return mask


def get_legal_action_mask_np(n_actions, legal_actions_list, dtype=np.uint8):
    """

    Args:
        legal_actions_list (list):  List of legal actions as integers, where 0 is always FOLD, 1 is CHECK/CALL.
                                    2 is BET/RAISE for continuous PokerEnvs, and for DiscretePokerEnv subclasses,
                                    numbers greater than 1 are all the raise sizes.

        dtype:                      dtype the mask shall have

    Returns:
        np.ndarray:                 a many-hot representation of the list of legal actions.

    """
    mask = np.zeros(shape=n_actions, dtype=dtype)
    mask[legal_actions_list] = 1
    return mask
