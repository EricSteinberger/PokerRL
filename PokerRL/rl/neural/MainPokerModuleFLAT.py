# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch
import torch.nn as nn


class MainPokerModuleFLAT(nn.Module):
    """
    Feeds parts of the observation through different fc layers before the RNN

    Structure (each branch merge is a concat):

    Table & Player state --> FC -> RE -> FCS -> RE ----------------------------.
    Board Cards ---> FC -> RE --> cat -> FC -> RE -> FCS -> RE -> FC -> RE --> cat --> FC -> RE -> FCS-> RE -> Normalize
    Private Cards -> FC -> RE -'


    where FCS refers to FC+Skip and RE refers to ReLU
    """

    def __init__(self,
                 env_bldr,
                 device,
                 mpm_args,
                 ):
        super().__init__()
        self.args = mpm_args

        self.env_bldr = env_bldr

        self.N_SEATS = self.env_bldr.N_SEATS
        self.device = device

        self.board_start = self.env_bldr.obs_board_idxs[0]
        self.board_stop = self.board_start + len(self.env_bldr.obs_board_idxs)

        self.pub_obs_size = self.env_bldr.pub_obs_size
        self.priv_obs_size = self.env_bldr.priv_obs_size

        self._relu = nn.ReLU(inplace=False)

        if mpm_args.use_pre_layers:
            self._priv_cards = nn.Linear(in_features=self.env_bldr.priv_obs_size,
                                         out_features=mpm_args.other_units)
            self._board_cards = nn.Linear(in_features=self.env_bldr.obs_size_board,
                                          out_features=mpm_args.other_units)

            self.cards_fc_1 = nn.Linear(in_features=2 * mpm_args.other_units, out_features=mpm_args.card_block_units)
            self.cards_fc_2 = nn.Linear(in_features=mpm_args.card_block_units, out_features=mpm_args.card_block_units)
            self.cards_fc_3 = nn.Linear(in_features=mpm_args.card_block_units, out_features=mpm_args.other_units)

            self.hist_and_state_1 = nn.Linear(in_features=self.env_bldr.pub_obs_size - self.env_bldr.obs_size_board,
                                              out_features=mpm_args.other_units)
            self.hist_and_state_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

            self.final_fc_1 = nn.Linear(in_features=2 * mpm_args.other_units, out_features=mpm_args.other_units)
            self.final_fc_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

        else:
            self.final_fc_1 = nn.Linear(in_features=self.env_bldr.complete_obs_size, out_features=mpm_args.other_units)
            self.final_fc_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

        self.lut_range_idx_2_priv_o = torch.from_numpy(self.env_bldr.lut_holder.LUT_RANGE_IDX_TO_PRIVATE_OBS)
        self.lut_range_idx_2_priv_o = self.lut_range_idx_2_priv_o.to(device=self.device, dtype=torch.float32)

        self.to(device)

    @property
    def output_units(self):
        return self.args.other_units

    def forward(self, pub_obses, range_idxs):
        """
        1. do list -> padded
        2. feed through pre-processing fc layers
        3. PackedSequence (sort, pack)
        4. rnn
        5. unpack (unpack re-sort)
        6. cut output to only last entry in sequence

        Args:
            pub_obses (list):                 list of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (LongTensor):        range_idxs (one for each pub_obs) tensor([2, 421, 58, 912, ...])
        """

        # ____________________________________________ Packed Sequence _____________________________________________
        priv_obses = self.lut_range_idx_2_priv_o[range_idxs]

        if isinstance(pub_obses, list):
            pub_obses = torch.from_numpy(np.array(pub_obses)).to(self.device, torch.float32)

        if self.args.use_pre_layers:
            _board_obs = pub_obses[:, self.board_start:self.board_stop]
            _hist_and_state_obs = torch.cat([
                pub_obses[:, :self.board_start],
                pub_obses[:, self.board_stop:]
            ],
                dim=-1
            )
            y = self._feed_through_pre_layers(board_obs=_board_obs, priv_obs=priv_obses,
                                              hist_and_state_obs=_hist_and_state_obs)

        else:
            y = torch.cat((priv_obses, pub_obses,), dim=-1)

        final = self._relu(self.final_fc_1(y))
        final = self._relu(self.final_fc_2(final) + final)

        # Normalize last layer
        if self.args.normalize:
            final = final - final.mean(dim=-1).unsqueeze(-1)
            final = final / final.std(dim=-1).unsqueeze(-1)

        return final

    def _feed_through_pre_layers(self, priv_obs, board_obs, hist_and_state_obs):

        # """""""""""""""
        # Cards Body
        # """""""""""""""
        _priv_1 = self._relu(self._priv_cards(priv_obs))
        _board_1 = self._relu(self._board_cards(board_obs))

        cards_out = self._relu(self.cards_fc_1(torch.cat([_priv_1, _board_1], dim=-1)))
        cards_out = self._relu(self.cards_fc_2(cards_out) + cards_out)
        cards_out = self.cards_fc_3(cards_out)

        hist_and_state_out = self._relu(self.hist_and_state_1(hist_and_state_obs))
        hist_and_state_out = self.hist_and_state_2(hist_and_state_out) + hist_and_state_out

        return self._relu(torch.cat([cards_out, hist_and_state_out], dim=-1))


class MPMArgsFLAT:

    def __init__(self,
                 use_pre_layers=True,
                 card_block_units=192,
                 other_units=64,
                 normalize=True,
                 ):
        self.use_pre_layers = use_pre_layers
        self.other_units = other_units
        self.card_block_units = card_block_units
        self.normalize = normalize

    def get_mpm_cls(self):
        return MainPokerModuleFLAT
