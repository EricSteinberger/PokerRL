# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn


class AdvantageNet(nn.Module):

    def __init__(self, env_bldr, adv_net_args, device):
        super().__init__()

        self._env_bldr = env_bldr
        self._args = adv_net_args
        self._n_actions = env_bldr.N_ACTIONS

        self._relu = nn.ReLU(inplace=False)

        MPM = adv_net_args.mpm_args.get_mpm_cls()
        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=adv_net_args.mpm_args)

        self._final_layer = nn.Linear(in_features=self._mpm.output_units,
                                      out_features=adv_net_args.n_units_final)
        self._adv = nn.Linear(in_features=adv_net_args.n_units_final, out_features=self._n_actions)

        self.to(device)

    def forward(self, pub_obses, range_idxs, legal_action_masks):
        y = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)
        y = self._relu(self._final_layer(y))
        y = self._adv(y)

        # sets all illegal actions to 0 for mean computation
        y *= legal_action_masks

        # can't directly compute mean cause illegal actions are still in there. Computing sum and dividing by ||A(I)||
        mean = (y.sum(dim=1) / legal_action_masks.sum(dim=1)).unsqueeze(1).expand(-1, self._n_actions)

        # subtracting mean also subtracts from illegal actions; have to mask again
        return (y - mean) * legal_action_masks


class AdvNetArgs:

    def __init__(self, n_units_final, mpm_args):
        self.n_units_final = n_units_final
        self.mpm_args = mpm_args
