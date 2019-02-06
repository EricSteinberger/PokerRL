# Copyright (c) 2019 Eric Steinberger


import torch
import torch.nn as nn

from PokerRL.rl import rl_util


class MainPokerModuleRNN(nn.Module):
    """
    Feeds parts of the observation through different fc layers before the RNN

    Structure (each branch merge is a concat):

    Table & Player state --> FC -> ReLU -------------------------------------------.
    Board, private info  --> FC -> ReLU -> FC+Skip -> ReLU -> FC+Skip -> ReLU ---- cat --> FC -> ReLU -> RNN ->

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
        self.board_len = len(self.env_bldr.obs_board_idxs)

        self.table_start = self.env_bldr.obs_table_state_idxs[0]
        self.table_len = len(self.env_bldr.obs_table_state_idxs)

        self.players_info_starts = [player_i_idxs[0] for player_i_idxs in self.env_bldr.obs_players_idxs]
        self.players_info_lens = [len(player_i_idxs) for player_i_idxs in self.env_bldr.obs_players_idxs]

        self.pub_obs_size = self.env_bldr.pub_obs_size
        self.priv_obs_size = self.env_bldr.priv_obs_size

        self._relu = nn.ReLU(inplace=False)

        if mpm_args.use_pre_layers:
            self.cards_fc_1 = nn.Linear(in_features=self.env_bldr.obs_size_board + self.env_bldr.priv_obs_size,
                                        out_features=mpm_args.n_cards_state_units)

            self.cards_fc_2 = nn.Linear(in_features=mpm_args.n_cards_state_units,
                                        out_features=mpm_args.n_cards_state_units)

            self.cards_fc_3 = nn.Linear(in_features=mpm_args.n_cards_state_units,
                                        out_features=mpm_args.n_cards_state_units)

            self.table_state_fc = nn.Linear(in_features=self.env_bldr.obs_size_table_state
                                                        + self.env_bldr.obs_size_player_info_each * self.N_SEATS,
                                            out_features=mpm_args.n_merge_and_table_layer_units)

            self.merge_fc = nn.Linear(in_features=mpm_args.n_cards_state_units + mpm_args.n_merge_and_table_layer_units,
                                      out_features=mpm_args.n_merge_and_table_layer_units)

            self.rnn = rl_util.str_to_rnn_cls(mpm_args.rnn_cls_str)(input_size=mpm_args.n_merge_and_table_layer_units,
                                                                    hidden_size=mpm_args.rnn_units,
                                                                    num_layers=mpm_args.rnn_stack,
                                                                    dropout=mpm_args.rnn_dropout,
                                                                    bidirectional=False,
                                                                    batch_first=False)

        else:
            """ Inputs all data directly into the rnn. """
            self.rnn = rl_util.str_to_rnn_cls(mpm_args.rnn_cls_str)(
                input_size=self.env_bldr.complete_obs_size,
                hidden_size=mpm_args.rnn_units,
                num_layers=mpm_args.rnn_stack,
                dropout=mpm_args.rnn_dropout,
                bidirectional=False,
                batch_first=False
            )
        self.lut_range_idx_2_priv_o = torch.from_numpy(self.env_bldr.lut_holder.LUT_RANGE_IDX_TO_PRIVATE_OBS)
        self.lut_range_idx_2_priv_o = self.lut_range_idx_2_priv_o.to(device=self.device, dtype=torch.float32)

        self.to(device)

    @property
    def output_units(self):
        return self.args.rnn_units

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
        bs = len(pub_obses)

        # this is just for one history step! has shape [bs, n_priv_features] now
        priv_obs = self.lut_range_idx_2_priv_o[range_idxs]

        if bs > 1:

            seq_lens = torch.tensor([sample.shape[0] for sample in pub_obses], device=self.device, dtype=torch.int32)
            max_len = seq_lens.max().item()

            _pub_obs = pub_obses
            pub_obses = torch.zeros((max_len, bs, self.pub_obs_size), dtype=torch.float32, device=self.device)
            for i, pub in zip(range(bs), _pub_obs):
                pub_obses[:seq_lens[i], i] = torch.from_numpy(pub).to(self.device)

            # extend private obs to whole time series. Technically goes over the seq_len (to max len)
            priv_obs = priv_obs.unsqueeze(0).repeat(max_len, 1, 1)

            if self.args.use_pre_layers:
                # slice and feed through the fc layers. the result is a tensor: [max_len, bs, some_n_of_units]
                y = self._feed_through_pre_layers(pub_o=pub_obses, priv_o=priv_obs)
            else:
                # concat pub and priv obs
                y = torch.cat((priv_obs, pub_obses), dim=-1)

            # sort descending for packing
            seq_lens, idx_shifts = torch.sort(seq_lens, descending=True)
            y = y[:, idx_shifts, :]

            # pack
            y = torch.nn.utils.rnn.pack_padded_sequence(y, lengths=seq_lens, batch_first=False)

            # Feed through RNN
            y, _ = self.rnn(y)

            # Unpack Sequence
            y, seq_lens = nn.utils.rnn.pad_packed_sequence(y, batch_first=False)

            if self.args.sum_step_outputs:
                y = y.sum(0) * (1.0 / seq_lens.float()).unsqueeze(-1)
            else:  # use last element of each sequence in the batch, because we only need the last action
                y = y[seq_lens - 1, torch.arange(bs, device=self.device, dtype=torch.long), :].squeeze(dim=0)

            # sort the samples back to their original order
            idx_unsort_obs_t = torch.arange(bs, device=self.device, dtype=torch.long)
            idx_unsort_obs_t.scatter_(src=idx_unsort_obs_t.clone(), dim=0, index=idx_shifts)

            return y[idx_unsort_obs_t]

        else:
            seq_len = pub_obses[0].shape[0]
            pub_obses = torch.from_numpy(pub_obses[0]).to(self.device).view(seq_len, bs, self.pub_obs_size)
            priv_obs = priv_obs.unsqueeze(0).expand(seq_len, bs, self.priv_obs_size)

            if self.args.use_pre_layers:
                y = self._feed_through_pre_layers(pub_o=pub_obses, priv_o=priv_obs)
            else:
                # concat pub and priv obs
                y = torch.cat((priv_obs, pub_obses), dim=-1)

            y, _ = self.rnn(y)

            if self.args.sum_step_outputs:
                return y.sum(0) * (1.0 / seq_len)
            else:  # use last element of each sequence in the batch, because we only need the last action
                return y[seq_len - 1].view(bs, -1)

    def _feed_through_pre_layers(self, pub_o, priv_o):

        # """""""""""""""
        # Cards Body
        # """""""""""""""
        _cards_obs = torch.cat((priv_o, pub_o.narrow(dim=-1, start=self.board_start, length=self.board_len)), dim=-1)
        cards_out = self._relu(self.cards_fc_1(_cards_obs))
        cards_out = self._relu(self.cards_fc_2(cards_out) + cards_out)
        cards_out = self._relu(self.cards_fc_3(cards_out) + cards_out)

        # """""""""""""""
        # Table Body
        # """""""""""""""
        _table_obs = torch.cat(
            [
                pub_o.narrow(dim=-1, start=self.table_start, length=self.table_len)
            ]
            +
            [
                pub_o.narrow(dim=-1, start=self.players_info_starts[i],
                             length=self.players_info_lens[i])
                for i in range(self.N_SEATS)
            ]
            , dim=-1
        )
        table_out = self._relu(self.table_state_fc(_table_obs))

        # """""""""""""""
        # Merge Layer
        # """""""""""""""
        return self._relu(self.merge_fc(torch.cat([cards_out, table_out], dim=-1)))


class MPMArgsRNN:

    def __init__(self,
                 rnn_units,
                 rnn_stack,
                 rnn_dropout,
                 rnn_cls_str="lstm",
                 use_pre_layers=True,
                 n_cards_state_units=96,
                 n_merge_and_table_layer_units=32,
                 sum_step_outputs=False,
                 ):
        self.rnn_units = rnn_units
        self.rnn_stack = rnn_stack
        self.rnn_dropout = rnn_dropout
        self.rnn_cls_str = rnn_cls_str
        self.use_pre_layers = use_pre_layers
        self.n_cards_state_units = n_cards_state_units
        self.n_merge_and_table_layer_units = n_merge_and_table_layer_units
        self.sum_step_outputs = sum_step_outputs

    def get_mpm_cls(self):
        return MainPokerModuleRNN
