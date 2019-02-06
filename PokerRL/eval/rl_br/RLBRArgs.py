# Copyright (c) 2019 Eric Steinberger


import copy

from PokerRL.game.poker_env_args import DiscretizedPokerEnvArgs, LimitPokerEnvArgs, NoLimitPokerEnvArgs
from PokerRL.rl.agent_modules.DDQN import DDQNArgs
from PokerRL.rl.neural.DuelingQNet import DuelingQArgs


class RLBRArgs:

    def __init__(self,
                 rlbr_bet_set,
                 n_hands_each_seat=20000,
                 n_workers=5,

                 # Training
                 DISTRIBUTED=False,
                 n_iterations=10000,
                 play_n_games_per_iter=50,
                 pretrain_n_games=5120,
                 device_training="cpu",

                 # the DDQN
                 nn_type="feedforward",
                 target_net_update_freq=300,
                 batch_size=512,
                 buffer_size=5e4,
                 optim_str="adam",
                 loss_str="mse",
                 lr=0.001,
                 eps_start=0.3,
                 eps_min=0.02,
                 eps_const=0.02,
                 eps_exponent=0.7,

                 # the QNet
                 rnn_cls_str="lstm",
                 n_units_final=64,
                 n_merge_and_table_layer_units=64,
                 n_cards_state_units=192,
                 use_pre_layers=False,
                 rnn_units=32,
                 rnn_stack=1,
                 dropout=0.0,
                 ):

        if nn_type == "recurrent":
            from PokerRL.rl.neural.MainPokerModuleRNN import MPMArgsRNN

            mpm_args = MPMArgsRNN(rnn_cls_str=rnn_cls_str,
                                  rnn_units=rnn_units,
                                  rnn_stack=rnn_stack,
                                  rnn_dropout=dropout,
                                  use_pre_layers=use_pre_layers,
                                  n_cards_state_units=n_cards_state_units,
                                  n_merge_and_table_layer_units=n_merge_and_table_layer_units)

        elif nn_type == "feedforward":
            from PokerRL.rl.neural.MainPokerModuleFLAT import MPMArgsFLAT

            mpm_args = MPMArgsFLAT(use_pre_layers=use_pre_layers,
                                   card_block_units=n_cards_state_units,
                                   other_units=n_merge_and_table_layer_units)

        else:
            raise ValueError(nn_type)
        if DISTRIBUTED and n_workers < 2:
            raise RuntimeError("RL-BR needs at least 2 workers, when running distributed. This is for 1 ParameterServer"
                               "and at least one LearnerActor")

        self.n_las = (n_workers - 1) if DISTRIBUTED else 1

        self.n_hands_each_seat = int(n_hands_each_seat)
        self.n_iterations = int(n_iterations)
        self.play_n_games_per_iter = int(play_n_games_per_iter)
        self.pretrain_n_games = int(pretrain_n_games)
        self.rlbr_bet_set = rlbr_bet_set

        self.ddqn_args = DDQNArgs(
            q_args=DuelingQArgs(
                n_units_final=n_units_final,
                mpm_args=mpm_args),
            cir_buf_size=int(buffer_size),
            batch_size=int(batch_size),
            n_mini_batches_per_update=1,
            target_net_update_freq=target_net_update_freq,
            optim_str=optim_str,
            loss_str=loss_str,
            lr=lr,
            eps_start=eps_start,
            eps_const=eps_const,
            eps_exponent=eps_exponent,
            eps_min=eps_min,
            grad_norm_clipping=1.0,
            device_training=device_training,
        )

    def get_rlbr_env_args(self, agents_env_args, randomization_range=None):
        arg_cls = type(agents_env_args)

        if arg_cls is DiscretizedPokerEnvArgs:
            return DiscretizedPokerEnvArgs(
                n_seats=agents_env_args.n_seats,
                starting_stack_sizes_list=copy.deepcopy(agents_env_args.starting_stack_sizes_list),
                bet_sizes_list_as_frac_of_pot=copy.deepcopy(self.rlbr_bet_set),
                stack_randomization_range=randomization_range if randomization_range else (0, 0),
                use_simplified_headsup_obs=agents_env_args.use_simplified_headsup_obs,
                uniform_action_interpolation=False
            )

        elif arg_cls is LimitPokerEnvArgs:
            return LimitPokerEnvArgs(
                n_seats=agents_env_args.n_seats,
                starting_stack_sizes_list=copy.deepcopy(agents_env_args.starting_stack_sizes_list),
                stack_randomization_range=randomization_range if randomization_range else (0, 0),
                use_simplified_headsup_obs=agents_env_args.use_simplified_headsup_obs,
                uniform_action_interpolation=False
            )

        elif arg_cls is NoLimitPokerEnvArgs:
            raise NotImplementedError("Currently not supported")

        else:
            raise NotImplementedError(arg_cls)
