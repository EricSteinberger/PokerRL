# Copyright (c) 2019 Eric Steinberger
# Inspiration of architecture from DeepStack-Leduc (https://github.com/lifrordi/DeepStack-Leduc/tree/master/Source)


from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs


class NodeBase:
    """Base node from which all nodes extend"""

    def __init__(self, env_state, tree, parent, p_id_acted_last, is_terminal, depth):
        """
        A NodeBase stores the STATE it had BEFORE someone acted in it. all actions trigger a child.
        The strategy in a node refers to the prob distr that the player that acts in the node
        acts with to PRODUCE THE NODE'S CHILDREN.
        """
        self.env_state = env_state
        self.parent = parent
        self.p_id_acting_next = self.env_state[EnvDictIdxs.current_player]
        self.p_id_acted_last = p_id_acted_last

        self.is_terminal = is_terminal
        self.depth = depth
        self.tree = tree

        # built in recursion
        self.allowed_actions = []
        self.children = []

        self.strategy = None  # p_id_acting_next' strategy: np.arr((range_size, n_actions), np.float32)
        self.reach_probs = None  # reach probs of all players: np.arr((n_seats, range_size), np.float32)
        self.ev = None  # EVs for all players: np.arr((n_seats, range_size), np.float32)
        self.ev_br = None  # EV vs BR:  np.arr((n_seats, range_size), np.float32)
        self.ev_weighted = None
        self.ev_br_weighted = None
        self.br_a_idx_in_child_arr_for_each_hand = None
        self.epsilon = None  # float
        self.exploitability = None  # float

        # Any algorithm using a PublicTree to run on (e.g. CFR, CFR+) can use this to hold a dict of vals
        self.data = None


class PlayerActionNode(NodeBase):

    def __init__(self, env_state, tree, parent, is_terminal, p_id_acted_last, action, depth, new_round_state=None):
        super().__init__(env_state=env_state, tree=tree, parent=parent, p_id_acted_last=p_id_acted_last,
                         is_terminal=is_terminal, depth=depth)
        if action is not None:
            self.action = action
        else:
            self.action = None
        self.new_round_state = new_round_state  # Hack to work with PokerEnv


class ChanceNode(NodeBase):

    def __init__(self, env_state, tree, parent, p_id_acted_last, is_terminal, depth):
        super().__init__(env_state=env_state, tree=tree, parent=parent, p_id_acted_last=p_id_acted_last,
                         is_terminal=is_terminal, depth=depth)

        self.action = "CHANCE"
