# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn


class DuelingQNet(nn.Module):

    def __init__(self, env_bldr, q_args, device):
        super().__init__()

        self._env_bldr = env_bldr
        self._q_args = q_args
        self._n_actions = env_bldr.N_ACTIONS

        self._relu = nn.ReLU(inplace=False)

        MPM = q_args.mpm_args.get_mpm_cls()
        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=q_args.mpm_args)

        # ____________________ advantage net & v net layers _______________________
        self._adv_layer = nn.Linear(in_features=self._mpm.output_units, out_features=q_args.n_units_final)
        self._state_v_layer = nn.Linear(in_features=self._mpm.output_units, out_features=q_args.n_units_final)

        self._adv = nn.Linear(in_features=q_args.n_units_final, out_features=self._n_actions)
        self._v = nn.Linear(in_features=q_args.n_units_final, out_features=1)

        self.to(device)

    def forward(self, pub_obses, range_idxs, legal_action_masks):
        shared_out = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)

        adv = self._get_adv(shared_out=shared_out, legal_action_masks=legal_action_masks)
        val_layer = self._relu(self._state_v_layer(shared_out))
        val = self._v(val_layer).expand_as(adv)

        # The mask is important because the strategy inference assumes illegals to have exactly zero
        return (val + adv) * legal_action_masks

    def get_adv(self, pub_obses, range_idxs, legal_action_masks):
        shared_out = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)
        return self._get_adv(shared_out=shared_out, legal_action_masks=legal_action_masks)

    def _get_adv(self, shared_out, legal_action_masks):
        y = self._relu(self._adv_layer(shared_out))
        y = self._adv(y)

        # sets all illegal actions to 0 for mean computation
        y *= legal_action_masks

        # can't directly compute mean cause illegal actions are still in there. Computing sum and dividing by ||A(I)||
        mean = (y.sum(dim=1) / legal_action_masks.sum(dim=1)).unsqueeze(1).expand(-1, self._n_actions)

        # subtracting mean also subtracts from illegal actions; have to mask again
        return (y - mean) * legal_action_masks


class DuelingQArgs:

    def __init__(self, n_units_final, mpm_args):
        self.n_units_final = n_units_final
        self.mpm_args = mpm_args
