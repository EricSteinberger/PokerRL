# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch

from PokerRL.rl import rl_util
from PokerRL.rl.neural.DuelingQNet import DuelingQNet
from PokerRL.rl.neural.NetWrapperBase import NetWrapperArgsBase as _NetWrapperArgsBase
from PokerRL.rl.neural.NetWrapperBase import NetWrapperBase as _NetWrapperBase


class DDQN(_NetWrapperBase):

    def __init__(self,
                 env_bldr,
                 ddqn_args,
                 owner,
                 ):
        super().__init__(
            net=DuelingQNet(env_bldr=env_bldr, q_args=ddqn_args.q_args, device=ddqn_args.device_training),
            env_bldr=env_bldr,
            args=ddqn_args,
            owner=owner,
            device=ddqn_args.device_training,
        )

        self._eps = None

        self._target_net = DuelingQNet(env_bldr=env_bldr, q_args=ddqn_args.q_args, device=ddqn_args.device_training)
        self._target_net.eval()
        self.update_target_net()

        self._batch_arranged = torch.arange(ddqn_args.batch_size, dtype=torch.long, device=self.device)
        self._minus_e20 = torch.full((ddqn_args.batch_size, self._env_bldr.N_ACTIONS,),
                                     fill_value=-10e20,
                                     device=self.device,
                                     dtype=torch.float32,
                                     requires_grad=False)

        self._n_actions_arranged = np.arange(self._env_bldr.N_ACTIONS, dtype=np.int32).tolist()

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, value):
        self._eps = value

    def select_br_a(self, pub_obses, range_idxs, legal_actions_lists, explore=False):
        if explore and (np.random.random() < self._eps):
            return np.array(
                [legal_actions[np.random.randint(len(legal_actions))] for legal_actions in legal_actions_lists]
            )

        with torch.no_grad():
            self.eval()
            range_idxs = torch.tensor(range_idxs, dtype=torch.long, device=self.device)
            q = self._net(pub_obses=pub_obses, range_idxs=range_idxs,
                          legal_action_masks=rl_util.batch_get_legal_action_mask_torch(
                              n_actions=self._env_bldr.N_ACTIONS,
                              legal_actions_lists=legal_actions_lists,
                              device=self.device,
                              dtype=torch.float32)).cpu().numpy()
            for b in range(q.shape[0]):
                illegal_actions = [i for i in self._n_actions_arranged if i not in legal_actions_lists[b]]
                if len(illegal_actions) > 0:
                    illegal_actions = np.array(illegal_actions)
                    q[b, illegal_actions] = -1e20

            return np.argmax(q, axis=1)

    def update_target_net(self):
        self._target_net.load_state_dict(self._net.state_dict())
        self._target_net.eval()

    def _mini_batch_loop(self, buffer, grad_mngr):
        batch_pub_obs_t, \
        batch_a_t, \
        batch_range_idx, \
        batch_legal_action_mask_t, \
        batch_r_t, \
        batch_pub_obs_tp1, \
        batch_legal_action_mask_tp1, \
        batch_done = \
            buffer.sample(device=self.device, batch_size=self._args.batch_size)

        # [batch_size, n_actions]
        q1_t = self._net(pub_obses=batch_pub_obs_t, range_idxs=batch_range_idx,
                         legal_action_masks=batch_legal_action_mask_t.to(torch.float32))
        q1_tp1 = self._net(pub_obses=batch_pub_obs_tp1, range_idxs=batch_range_idx,
                           legal_action_masks=batch_legal_action_mask_tp1.to(torch.float32)).detach()
        q2_tp1 = self._target_net(pub_obses=batch_pub_obs_tp1, range_idxs=batch_range_idx,
                                  legal_action_masks=batch_legal_action_mask_tp1.to(torch.float32)).detach()

        # ______________________________________________ TD Learning _______________________________________________
        # [batch_size]
        q1_t_of_a_selected = q1_t[self._batch_arranged, batch_a_t]

        # only consider allowed actions for tp1
        q1_tp1 = torch.where(batch_legal_action_mask_tp1,
                             q1_tp1,
                             self._minus_e20)

        # [batch_size]
        _, best_a_tp1 = q1_tp1.max(dim=-1, keepdim=False)
        q2_best_a_tp1 = q2_tp1[self._batch_arranged, best_a_tp1]

        q2_best_a_tp1 = q2_best_a_tp1 * (1.0 - batch_done)
        target = batch_r_t + q2_best_a_tp1

        grad_mngr.backprop(pred=q1_t_of_a_selected, target=target)

    def state_dict(self):
        return {
            "q_net": self._net.state_dict(),
            "target_net": self._target_net.state_dict(),
            "eps": self._eps,
            "owner": self.owner,
            "args": self._args,
        }

    def load_state_dict(self, state):
        assert self.owner == state["owner"]
        # Not loading args by design

        self._net.load_state_dict(state["q_net"])
        self._target_net.load_state_dict(state["target_net"])
        self._eps = state["eps"]

    @staticmethod
    def from_state_dict(state_dict, env_bldr):
        ddqn = DDQN(owner=state_dict["owner"],
                    ddqn_args=state_dict["args"],
                    env_bldr=env_bldr)
        ddqn.load_state_dict(state_dict)
        ddqn.update_target_net()
        return ddqn

    @staticmethod
    def inference_version_from_state_dict(state_dict, env_bldr):
        ddqn = DDQN.from_state_dict(state_dict=state_dict, env_bldr=env_bldr)
        ddqn.buf = None
        ddqn.eps = None
        return ddqn


class DDQNArgs(_NetWrapperArgsBase):

    def __init__(self,
                 q_args,
                 cir_buf_size=1e5,
                 batch_size=512,
                 n_mini_batches_per_update=1,
                 target_net_update_freq=300,
                 optim_str="adam",
                 loss_str="mse",
                 lr=0.005,
                 eps_start=0.065,
                 eps_const=0.007,
                 eps_exponent=0.475,
                 eps_min=0.02,
                 grad_norm_clipping=10.0,
                 device_training="cpu",
                 ):
        assert isinstance(device_training, str), "Please pass a string (either 'cpu' or 'cuda')!"

        super().__init__(batch_size=batch_size,
                         n_mini_batches_per_update=n_mini_batches_per_update,
                         optim_str=optim_str,
                         loss_str=loss_str,
                         lr=lr,
                         grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)

        self.q_args = q_args
        self.cir_buf_size = int(cir_buf_size)
        self.target_net_update_freq = int(target_net_update_freq)
        self.eps_start = eps_start
        self.eps_const = eps_const
        self.eps_exponent = eps_exponent
        self.eps_min = eps_min
