# Copyright (c) 2019 Eric Steinberger
# Inspiration of architecture from DeepStack-Leduc (https://github.com/lifrordi/DeepStack-Leduc/tree/master/Source)

import copy
import json
import os

import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs, PlayerDictIdxs
from PokerRL.game._.tree._.StrategyFiller import StrategyFiller
from PokerRL.game._.tree._.ValueFiller import ValueFiller
from PokerRL.game._.tree._.nodes import PlayerActionNode, ChanceNode
from PokerRL.util import file_util


class PublicTree:
    """
    Builds a full public game tree to compute the counterfactual values and the best response against strategy profiles.
    You can also visualize game trees in the browser as described in README.md.

    Warning: This part of PokerRL is INEFFICIENT and SLOW! Only works in Leduc games (no limit Leduc too though)
    The reason why we don't really care though is that there are great codebases for CFR and alike in large games.
    Deep Learning methods spend most of the time evaluating neural nets for strategy queries. This is no different even
    with this horribly slow implementation of a Public Tree; its just that the tree has 5% instead of 0.001% overhead.
    """
    CHANCE_ID = "Ch"

    def __init__(self,
                 env_bldr,
                 stack_size,
                 stop_at_street,
                 put_out_new_round_after_limit=False,
                 is_debugging=False,
                 ):
        """
        To start the tree from a given scenario, set ""env"" to that scenario and it will be treated
        as the root.
        """
        self._env_bldr = env_bldr
        self._is_debugging = is_debugging
        self._stack_size = stack_size

        # set up optional graphical tree export
        _dir_tree_vis = os.path.join("C:\\" if os.name == 'nt' else os.path.expanduser('~/'), "PokerRL_Viz")

        # This is just me being over-cautious to not overwrite someones files...
        _tree_vis_installed = os.path.isdir(_dir_tree_vis) \
                              and os.path.isfile(os.path.join(_dir_tree_vis, "ALLOWED_TO_WRITE_HERE.dontdelete"))

        if _tree_vis_installed:
            self.dir_tree_vis_data = os.path.join(_dir_tree_vis, "data")
            file_util.create_dir_if_not_exist(self.dir_tree_vis_data)
        else:
            self.dir_tree_vis_data = None

        # _________________________________ tree vars __________________________________
        self.root = None
        self._n_nodes = 0
        self._n_nonterm = 0

        self._env = self._env_bldr.get_new_env(is_evaluating=True, stack_size=stack_size)
        _ar = self._env.get_args()
        _ar.RETURN_PRE_TRANSITION_STATE_IN_INFO = True
        self._env.set_args(_ar)
        self._env.reset()

        self._n_seats = self._env_bldr.N_SEATS

        # _____________________ stops building tree at given round _____________________
        self._stop_at_street = max(self._env.ALL_ROUNDS_LIST) + 1 if stop_at_street is None else stop_at_street
        self._put_out_new_round_after_limit = put_out_new_round_after_limit

        self._value_filler = ValueFiller(tree=self)
        self._strategy_filler = StrategyFiller(tree=self, env_bldr=env_bldr)

    # _____________________________________________________ API ________________________________________________________
    @property
    def stack_size(self):
        return self._stack_size

    @property
    def is_debugging(self):
        return self._is_debugging

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def n_nonterm(self):
        return self._n_nonterm

    @property
    def n_seats(self):
        return self._n_seats

    @property
    def stop_at_street(self):
        return self._stop_at_street

    @property
    def put_out_new_round_after_limit(self):
        return self._put_out_new_round_after_limit

    @property
    def env_bldr(self):
        return self._env_bldr

    def build_tree(self):
        """
        Builds from the current state of the environment
        """
        self.root = ChanceNode(env_state=self._env.state_dict(),
                               tree=self,
                               parent=None,
                               is_terminal=False,
                               p_id_acted_last=None,
                               depth=0)
        self.root.allowed_actions = self._env.get_legal_actions()
        self.root.reach_probs = np.full(shape=(self._env_bldr.N_SEATS, self._env_bldr.rules.RANGE_SIZE),
                                        fill_value=1.0 / float(self._env_bldr.rules.RANGE_SIZE),
                                        dtype=np.float32)

        self._build_tree(current_node=self.root)

    def compute_ev(self):
        self._value_filler.compute_cf_values_heads_up(self.root)

    def fill_uniform_random(self):
        self._strategy_filler.fill_uniform_random()

    def fill_random_random(self):
        self._strategy_filler.fill_random_random()

    def fill_with_agent_policy(self, agent):
        self._strategy_filler.fill_with_agent_policy(agent=agent)

    def update_reach_probs(self):
        self._strategy_filler.update_reach_probs()

    def get_tree_as_dict(self):
        return self._export_for_node_strategy_tree(self.root)

    def export_to_file(self, name="data"):
        if self.dir_tree_vis_data is not None:
            file_util.write_dict_to_file_js(_dir=self.dir_tree_vis_data, file_name=name,
                                            dictionary=self.get_tree_as_dict())

    def copy(self):
        _tree = PublicTree(env_bldr=self._env_bldr,
                           stack_size=copy.deepcopy(self._stack_size),
                           stop_at_street=self._stop_at_street,
                           put_out_new_round_after_limit=self._put_out_new_round_after_limit)
        _tree.root = copy.deepcopy(self.root)

        return _tree

    # __________________________________________________ INTERNAL ______________________________________________________
    def _build_tree(self, current_node):
        current_node.children = self._get_children_nodes(node=current_node)
        self._n_nodes += len(current_node.children)
        self._n_nonterm += len([c for c in current_node.children if (not c.is_terminal)])
        for child in current_node.children:
            self._build_tree(current_node=child)

    def _get_children_of_action_node(self, parent):
        """
        Creates the children nodes after an action node (a node where a player acted).
        """
        self._env.load_state_dict(parent.env_state)
        if self._stop_at_street <= self._env.current_round:
            return []

        if parent.p_id_acting_next == self.CHANCE_ID:
            """
            When we get here, the parent node has a special attribute that carries the state after the new round is
            started. Here we adjust that state to not have the newly dealt cards dealt and cards drawn from the deck,
            but all other updates stay.
            """

            # ______________________________________ break if too far ______________________________________
            if not self._put_out_new_round_after_limit:
                if self._stop_at_street <= parent.env_state[EnvDictIdxs.current_round]:
                    return []

            # _____________________ prepare state to new round without new cards dealt _____________________
            _base_env_state = copy.deepcopy(parent.new_round_state)

            # ________________________________ generate all possible boards ________________________________

            def _make_boards(_new_boards, _board_1d, _idx_to_insert_in, _n_todo):
                if _n_todo > 0:
                    for _c in [_ for _ in range(self._env.N_CARDS_IN_DECK) if _ not in _board_1d]:
                        _b = np.copy(_board_1d)
                        _b[_idx_to_insert_in] = _c

                        for i in range(_n_todo):
                            _make_boards(_new_boards=_new_boards, _board_1d=_b, _idx_to_insert_in=_idx_to_insert_in + 1,
                                         _n_todo=_n_todo - 1)

                        _new_boards.append(self._env_bldr.lut_holder.get_2d_cards(cards_1d=_b))

            new_boards = []
            board_1d = self._env_bldr.lut_holder.get_1d_cards(cards_2d=_base_env_state[EnvDictIdxs.board_2d])
            _make_boards(_new_boards=new_boards,
                         _board_1d=board_1d,
                         _idx_to_insert_in=self._env_bldr.lut_holder.DICT_LUT_N_CARDS_OUT[
                             self._env_bldr.rules.ROUND_BEFORE[_base_env_state[EnvDictIdxs.current_round]]],
                         _n_todo=self._env_bldr.lut_holder.DICT_LUT_CARDS_DEALT_IN_TRANSITION_TO[
                             _base_env_state[EnvDictIdxs.current_round]])

            # ____________________________________ build children nodes ____________________________________
            children = []
            for board in new_boards:
                child_env_state = copy.deepcopy(_base_env_state)
                child_env_state[EnvDictIdxs.board_2d] = board
                _deck_remaining_before_deal = np.copy(child_env_state[EnvDictIdxs.deck]["deck_remaining"])
                for c in board:
                    child_env_state[EnvDictIdxs.deck]["deck_remaining"] = \
                        np.delete(_deck_remaining_before_deal, np.where(_deck_remaining_before_deal == c), axis=0)

                c = ChanceNode(env_state=child_env_state,
                               tree=self,
                               parent=parent,
                               is_terminal=False,
                               p_id_acted_last=self.CHANCE_ID,
                               depth=parent.depth + 1)

                self._env.load_state_dict(child_env_state)
                c.allowed_actions = self._env.get_legal_actions()
                children.append(c)

            return children

        else:
            children = []
            for action in parent.allowed_actions:
                self._env.load_state_dict(parent.env_state)
                _, __, is_terminal, info = self._env.step(action)

                # after action:  Terminal
                if is_terminal:
                    # state before payouts or rundown
                    env_state = info["state_dict_before_money_move"]
                    env_state[EnvDictIdxs.current_round] = parent.env_state[EnvDictIdxs.current_round]
                    env_state[EnvDictIdxs.board_2d] = parent.env_state[EnvDictIdxs.board_2d]
                    env_state[EnvDictIdxs.deck] = copy.deepcopy(parent.env_state[EnvDictIdxs.deck])
                    is_terminal = True
                    new_round_state = None

                # after action:  New round
                elif info["chance_acts"]:
                    # state before round transition
                    env_state = info["state_dict_before_money_move"]
                    env_state[EnvDictIdxs.current_round] = parent.env_state[EnvDictIdxs.current_round]
                    if self._is_debugging:
                        assert env_state[EnvDictIdxs.current_round] == parent.env_state[EnvDictIdxs.current_round]
                        assert np.array_equal(env_state[EnvDictIdxs.board_2d],
                                              np.copy(parent.env_state[EnvDictIdxs.board_2d]))
                        assert np.array_equal(env_state[EnvDictIdxs.deck]["deck_remaining"],
                                              np.copy(parent.env_state[EnvDictIdxs.deck]["deck_remaining"]))
                    is_terminal = False
                    new_round_state = self._env.state_dict()
                    new_round_state[EnvDictIdxs.board_2d] = np.copy(parent.env_state[EnvDictIdxs.board_2d])
                    new_round_state[EnvDictIdxs.deck] = copy.deepcopy(parent.env_state[EnvDictIdxs.deck])

                else:
                    env_state = self._env.state_dict()
                    is_terminal = False
                    new_round_state = None

                node = PlayerActionNode(env_state=env_state,
                                        action=action,
                                        is_terminal=is_terminal,
                                        parent=parent,
                                        p_id_acted_last=parent.p_id_acting_next,
                                        tree=self,
                                        depth=parent.depth + 1,
                                        new_round_state=new_round_state)

                if is_terminal:
                    node.p_id_acting_next = None
                    node.allowed_actions = []
                elif info["chance_acts"]:
                    node.p_id_acting_next = self.CHANCE_ID
                    node.allowed_actions = []
                else:
                    node.allowed_actions = self._env.get_legal_actions()
                children.append(node)

            return children

    def _get_children_nodes(self, node):
        if node.is_terminal:
            return []
        else:
            return self._get_children_of_action_node(parent=node)  # -> action or chance or terminal

    def _get_action_as_str(self, action):
        if action is None:
            return "None"
        mapping = {
            Poker.FOLD: "FOLD",
            Poker.CHECK_CALL: "CHECK",
            "CHANCE": "CHANCE"
        }

        # Returns RX where X is the index of the bet-size option selected, if the player raises/bets
        return mapping.get(action, "R" + str(action - 2))

    def _export_for_node_strategy_tree(self, node):

        def _2darr_to_str(arr):
            _str = ""
            if arr is None:
                return "Not Computed"

            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    _str += str("{:10.4f}".format(arr[i, j])) + " "
                if i < arr.shape[0] - 1:
                    _str += " || "

            return _str

        if node.parent is None:
            title = "ROOT"
        elif isinstance(node, ChanceNode):
            title = self._env.cards2str(cards_2d=node.env_state[EnvDictIdxs.board_2d])
        else:
            title = "Player acted last " + str(node.p_id_acted_last) + " :: Action: " + self._get_action_as_str(
                node.action) + " :: Board: " + self._env.cards2str(cards_2d=node.env_state[EnvDictIdxs.board_2d])

        """
        Tree print format for this field is (note, in each state only one player acts): 
        pi_p(a1, h1) pi_p(a2, h1) ... pi_p(an, h1) ... || pi_p(a1, h2) pi_p(a2, h2) ... pi_p(an, h2) || ...
        """
        strat_str = _2darr_to_str(node.strategy)

        """
        Tree print format for this field is: 
        reach(p1, h1) reach(p1, h2) ... reach(p1, hn) || reach(p2, h1) reach(p2, h2) ... reach(p2, hn)
        """
        reach_probs_str = _2darr_to_str(node.reach_probs)

        """
        Tree print format for this field is: 
        ev(p1, h1) ev(p1, h2) ... ev(p1, hn) || ev(p2, h1) ev(p2, h2) ... ev(p2, hn)
        """
        ev_str = _2darr_to_str(node.ev)

        """
        Tree print format for this field is: 
        ev_br(p1, h1) ev_br(p1, h2) ... ev_br(p1, hn) || ev_br(p2, h1) ev_br(p2, h2) ... ev_br(p2, hn)
        """
        ev_br_str = _2darr_to_str(node.ev_br)

        if isinstance(node, np.ndarray) and len(node.shape) == 2:
            """ Since the data field is general, it is impossible to know how/what to format. The code below is capable
                of processing 2d numpy arrays. For anything else, we will set an empty string. """
            data_str = _2darr_to_str(node.data)
        else:
            data_str = ""

        json_table = {
            'text': {
                # for header
                'title': title,
                'round': "Round : " + Poker.INT2STRING_ROUND[node.env_state[EnvDictIdxs.current_round]],

                # for body
                'main_pot': "Pot : " + json.dumps(int(node.env_state[EnvDictIdxs.main_pot])),

                'terminal': ("TERM" if str(node.is_terminal) else "") + " " + str(node.allowed_actions),

                'side_pots': "SP: " + json.dumps(
                    [int(i) for i in node.env_state[EnvDictIdxs.side_pots]]),

                'stack_sizes': "Stacks: " + json.dumps(
                    [int(s[PlayerDictIdxs.stack]) for s in node.env_state[EnvDictIdxs.seats]]),

                'current_bets': "Bets: " + json.dumps(
                    [int(s[PlayerDictIdxs.current_bet]) for s in node.env_state[EnvDictIdxs.seats]]),

                'not_folded': "Playing: " + json.dumps(
                    [not s[PlayerDictIdxs.folded_this_episode] for s in node.env_state[EnvDictIdxs.seats]]).replace(
                    "true", "1").replace("false", "0") + "  Next: " + str(node.p_id_acting_next),

                "exploitability": "Exploitability: " + (str(
                    node.exploitability) if node.exploitability is not None else "Exploitability not computed")
                                  + "   ||   BR Action per hand "
                                  + ("" if node.br_a_idx_in_child_arr_for_each_hand is None else
                                     str([self._get_action_as_str(a)
                                          for a in
                                          np.array(node.allowed_actions)[node.br_a_idx_in_child_arr_for_each_hand]])),

                'strategy': "STRAT: " + strat_str,

                'reach_probs': "REACH: " + reach_probs_str,

                'ev': "EV: " + ev_str,

                'ev_br': "EV-BR: " + ev_br_str,

                'data': "DATA: " + data_str
            },

            # tree nodes are all collapsed by default
            'collapsed': True,

            'children': []
        }

        # recurse over all children
        for child in node.children:
            json_table['children'].append(self._export_for_node_strategy_tree(child))

        return json_table
