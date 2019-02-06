# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn


class QNet(nn.Module):

    def __init__(self, env_bldr, q_args, device):
        super().__init__()

        self._env_bldr = env_bldr
        self._args = q_args
        self._n_actions = env_bldr.N_ACTIONS

        self._relu = nn.ReLU(inplace=False)

        MPM = q_args.mpm_args.get_mpm_cls()
        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=q_args.mpm_args)

        self._final_layer = nn.Linear(in_features=self._mpm.output_units,
                                      out_features=q_args.n_units_final)

        self._val = nn.Linear(in_features=q_args.n_units_final, out_features=self._n_actions)

        self.to(device)

    def forward(self, pub_obses, range_idxs, legal_action_masks):
        """
        Args:
            pub_obses (list):                       list of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (torch.Tensor):              integer representation of hand
        """
        y = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)
        y = self._relu(self._final_layer(y))
        y = self._val(y)

        # The mask is important because the strategy inference assumes illegals to have exactly zero
        return y * legal_action_masks


class QNetArgs:

    def __init__(self, n_units_final, mpm_args):
        self.n_units_final = n_units_final
        self.mpm_args = mpm_args
