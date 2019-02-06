# Copyright (c) 2019 Eric Steinberger


import copy
import time

import numpy as np
from gym import spaces

from PokerRL.game.Poker import Poker
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs, PlayerDictIdxs
from PokerRL.game._.rl_env.base._Deck import DeckOfCards
from PokerRL.game._.rl_env.base._PokerPlayer import PokerPlayer


class PokerEnv:
    """
    This abstract class implements general functions and attributes of a Poker Environment. Subclasses can define
    exact rules and hyperparameters of the environment. This base-engine supports HeadsUp and N-player poker.
    It is sub-base-classed by a Limit- and a Discrete- wrapper, as this base-class would default to the No-Limit
    ruleset.

    Functionality & Dynamics:
        reward:
            if terminal state:
                                numpy.array(shape=n_seats) that contains ""stack_after_episode - starting_stack""
                                for every player. This reward is either scaled or unscaled, depending on the setting
                                in the args that were passed when creating an instance of a PokerEnv subclass.

            else:
                                numpy.zeros(shape=n_seats)

        obs:
            if terminal state:
                                numpy array of zeros. The shape is the same as in non terminal states

            else:
                                covering PUBLIC state of the env (stacks, pot, board, etc.) No hole cards (or any
                                private information) from any player are included! To get a player's private
                                information, checkout .get_hole_cards_of_player()


        actions:
            For DiscretePokerEnv subclasses:
                                    0 = FOLD, 1 = CHECK/CALL
                                    2 = RAISE_SIZE_1, ... N = RAISE_SIZE_N-1

            For direct subclasses (i.e. "approximately continuous bet size" poker games like No Limit Holdem:
                                    Tuple: (action, raise size), where action is 0 for FOLD, 1 for CHECK/CALL, 2 for
                                    BET/RAISE. the raise_size is always to be passed, but only matters if the action
                                    is 2 (i.e. BET/RAISE). It is expressed as the total number of chips the player
                                    wants to have bet after placing the bet. So if the current bet is 30, and the agent
                                    wants to bet 60 chips more, the tuple should be (2, 90).

        card representations:
            Single cards can be referenced in 2 forms: 1D and 2D.
            1D refers to a card mapped to a single unique integer.
            2D refers to an array/tuple/list that contains the rank at the 1st index and the suit at the 2nd.

            A poker hand or the public board can be seen as an array of these cards. board_2d, for example refers to a
            2D array [[card1_rank, card1_suit], [card2_rank, card2_suit], ...].

            Alternatively, a hand of multiple cards can be referenced to by its index in an array sorted by the 1D
            representation of the first card, then by the 2nd, and so on. This format is called ""range_idx"" in this
            framework.
    """

    # _____________________ Variables to be defined by the subclassing game according to its rules _____________________

    SMALL_BLIND = NotImplementedError
    BIG_BLIND = NotImplementedError
    ANTE = NotImplementedError
    SMALL_BET = NotImplementedError
    BIG_BET = NotImplementedError
    DEFAULT_STACK_SIZE = NotImplementedError

    EV_NORMALIZER = NotImplementedError  # Divisor for chip numbers to compute e.g. MBB/H from raw chip numbers in eval.
    WIN_METRIC = NotImplementedError  # To plot graphs in relation to a fixed quantity (e.g. Poker.MeasureAnte)

    N_HOLE_CARDS = NotImplementedError  # number of private cards
    N_RANKS = NotImplementedError  # number of card-ranks
    N_SUITS = NotImplementedError  # number of card-suits
    N_CARDS_IN_DECK = NotImplementedError  # N_RANKS * N_SUITS
    RANGE_SIZE = NotImplementedError  # number of possible unique combinations of holecards

    BTN_IS_FIRST_POSTFLOP = NotImplementedError
    FIRST_ACTION_NO_CALL = False

    IS_FIXED_LIMIT_GAME = NotImplementedError
    IS_POT_LIMIT_GAME = NotImplementedError

    # Only relevant if Limit game!
    MAX_N_RAISES_PER_ROUND = NotImplementedError
    ROUND_WHERE_BIG_BET_STARTS = NotImplementedError

    # obs modes
    SUITS_MATTER = NotImplementedError  # Whether suits matter in the ruleset played. (i.e. do flushes exist?)

    N_FLOP_CARDS = NotImplementedError
    N_TURN_CARDS = NotImplementedError
    N_RIVER_CARDS = NotImplementedError
    N_TOTAL_BOARD_CARDS = NotImplementedError

    # This MUST NOT skip rounds. Setting it to [PREFLOP, FLOP] is ok, but [PREFLOP, TURN] will not work.
    ALL_ROUNDS_LIST = NotImplementedError

    # Dicts mapping rounds to other rounds.
    ROUND_BEFORE = NotImplementedError
    ROUND_AFTER = NotImplementedError

    # Map ints to strings for printing cards
    RANK_DICT = NotImplementedError
    SUIT_DICT = NotImplementedError

    # Class that contains the base-rulset the game follows
    RULES = NotImplementedError

    # ____________________________________________________ CONSTRUCT ___________________________________________________
    def __init__(self,
                 env_args,
                 lut_holder,
                 is_evaluating,
                 ):
        """
        Args:
            env_args:               Depending on game type an instance of PokerEnvArgs or DiscretePokerEnvArgs

            lut_holder:             Depending on game type. An instance of a subclass of LutHolder. It is not checked
                                    whether the correct one is passed, so be sure to pass the right one!
                                    lut_holder could theoretically be created encapsulated by an instance of this class,
                                    but for optimization (i.e. only one per machine, not per env), we pass it.

            is_evaluating (bool):   Whether the environment shall be spawned in evaluation mode (i.e. no randomization)
                                    or not.
        """
        assert env_args.n_seats >= 2

        self._args = copy.deepcopy(env_args)
        self.lut_holder = lut_holder
        self.IS_EVALUATING = is_evaluating

        # deck of cards
        self.deck = DeckOfCards(num_suits=self.N_SUITS, num_ranks=self.N_RANKS)

        # Initialize args
        self.BTN_POS = NotImplementedError
        self.SB_POS = NotImplementedError
        self.BB_POS = NotImplementedError
        self._USE_SIMPLE_HU_OBS = NotImplementedError
        self.RETURN_PRE_TRANSITION_STATE_IN_INFO = NotImplementedError
        self.N_SEATS = NotImplementedError
        self.MAX_CHIPS = NotImplementedError
        self.STACK_RANDOMIZATION_RANGE = NotImplementedError
        self.REWARD_SCALAR = NotImplementedError
        self.seats = NotImplementedError
        self._init_from_args(env_args=env_args, is_evaluating=is_evaluating)

        # ______________________________  Observation- & Action-space ______________________________
        self.observation_space, self.obs_idx_dict, self.obs_parts_idxs_dict = self._construct_obs_space()
        self.hole_card_space_shape = [self.N_HOLE_CARDS, 2]

        # __________________________________  episode state vars  __________________________________
        # all these vars should be initialized by calling env.reset() in the training script.
        self.current_round = None
        self.side_pots = None  # chip count in side pot.  list with len=n_seats
        self.main_pot = None  # chip count in main pot
        self.board = None  # np.ndarray(shape=(n_cards, 2))
        self.last_action = None  # list of 3 ints: [action_idx, _raise_amount, player.seat_id]
        self.capped_raise = CappedRaise()  # [happend_this_round, player_that_raised, player_that_cant_reopen]
        self.current_player = None  # PokerPlayer instance of player currently having to get_action
        self.last_raiser = None  # PokerPlayer instance
        self.n_actions_this_episode = None  # Number of actions performed this episode

        # only relevant in Limit games
        self.n_raises_this_round = NotImplementedError

    def _construct_obs_space(self):
        """
        The maximum all chip-values can reach is n_seats, because we normalize by dividing by the average starting stack
        """
        obs_idx_dict = {}
        obs_parts_idxs_dict = {
            "board": [],
            "players": [[] for _ in range(self.N_SEATS)],
            "table_state": [],
        }
        next_idx = [0]  # list is a mutatable object. int not.

        def get_discrete(size, name, _curr_idx):
            obs_idx_dict[name] = _curr_idx[0]
            _curr_idx[0] += 1
            return spaces.Discrete(size)

        def get_new_box(name, _curr_idx, high, low=0):
            obs_idx_dict[name] = _curr_idx[0]
            _curr_idx[0] += 1
            return spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

        if (self.N_SEATS == 2) and self._USE_SIMPLE_HU_OBS:
            # __________________________  Public Information About Game State  _________________________
            _k = next_idx[0]
            _table_space = [  # (blinds are in obs to give the agent a perspective on starting stack after normalization
                get_new_box("ante", next_idx, self.N_SEATS),  # .................................... self.ANTE
                get_new_box("small_blind", next_idx, self.N_SEATS),  # ............................. self.SMALL_BLIND
                get_new_box("big_blind", next_idx, self.N_SEATS),  # ............................... self.BIG_BLIND
                get_new_box("min_raise", next_idx, self.N_SEATS),  # ............................... min total raise
                get_new_box("pot_amt", next_idx, self.N_SEATS),  # ................................. main_pot amount
                get_new_box("total_to_call", next_idx, self.N_SEATS),  # ........................... total_to_call
                get_new_box("last_action_how_much", next_idx, self.N_SEATS),  # .................... self.last_action[1]
            ]
            for i in range(3):  # .................................................................. self.last_action[0]
                _table_space.append(get_discrete(1, "last_action_what_" + str(i), next_idx))

            for i in range(
                self.N_SEATS):  # ....................................................... self.last_action[2]
                _table_space.append(get_discrete(1, "last_action_who_" + str(i), next_idx))

            for i in range(
                self.N_SEATS):  # ....................................................... curr_player.seat_id
                _table_space.append(get_discrete(1, "p" + str(i) + "_acts_next", next_idx))

            for i in range(max(self.ALL_ROUNDS_LIST) + 1):  # ...................................... round onehot
                _table_space.append(get_discrete(1, "round_" + Poker.INT2STRING_ROUND[i], next_idx)),

            # add to parts_dict for possible slicing for agents.
            obs_parts_idxs_dict["table_state"] += list(range(_k, next_idx[0]))

            # __________________________  Public Information About Each Player  ________________________
            _player_space = []
            for i in range(self.N_SEATS):
                _k = next_idx[0]
                _player_space += [
                    get_new_box("stack_p" + str(i), next_idx, self.N_SEATS),  # ..................... stack
                    get_new_box("curr_bet_p" + str(i), next_idx, self.N_SEATS),  # .................. current_bet
                    get_discrete(1, "is_allin_p" + str(i), next_idx),  # ............................ is_allin
                ]

                # add to parts_dict for possible slicing for agents
                obs_parts_idxs_dict["players"][i] += list(range(_k, next_idx[0]))

            # _______________________________  Public cards (i.e. board)  ______________________________
            _board_space = []
            _k = next_idx[0]
            for i in range(self.N_TOTAL_BOARD_CARDS):

                x = []
                for j in range(self.N_RANKS):  # .................................................... rank
                    x.append(get_discrete(1, str(i) + "th_board_card_rank_" + str(j), next_idx))

                for j in range(self.N_SUITS):  # .................................................... suit
                    x.append(get_discrete(1, str(i) + "th_board_card_suit_" + str(j), next_idx))

                _board_space += x

            # add to parts_dict for possible slicing for agents
            obs_parts_idxs_dict["board"] += list(range(_k, next_idx[0]))

            # __________________________  Return Complete _Observation Space  __________________________
            # Tuple (lots of spaces.Discrete and spaces.Box)
            _observation_space = spaces.Tuple(_table_space + _player_space + _board_space)
            _observation_space.shape = [len(_observation_space.spaces)]

        else:
            # __________________________  Public Information About Game State  _________________________
            _k = next_idx[0]
            _table_space = [  # (blinds are in obs to give the agent a perspective on starting stack after normalization
                get_new_box("ante", next_idx, self.N_SEATS),  # .................................... self.ANTE
                get_new_box("small_blind", next_idx, self.N_SEATS),  # ............................. self.SMALL_BLIND
                get_new_box("big_blind", next_idx, self.N_SEATS),  # ............................... self.BIG_BLIND
                get_new_box("min_raise", next_idx, self.N_SEATS),  # ............................... min total raise
                get_new_box("pot_amt", next_idx, self.N_SEATS),  # ................................. main_pot amount
                get_new_box("total_to_call", next_idx, self.N_SEATS),  # ........................... total_to_call
                get_new_box("last_action_how_much", next_idx, self.N_SEATS),  # .................... self.last_action[1]
            ]
            for i in range(3):  # .................................................................. self.last_action[0]
                _table_space.append(get_discrete(1, "last_action_what_" + str(i), next_idx))

            for i in range(self.N_SEATS):  # ....................................................... self.last_action[2]
                _table_space.append(get_discrete(1, "last_action_who_" + str(i), next_idx))

            for i in range(self.N_SEATS):  # ....................................................... curr_player.seat_id
                _table_space.append(get_discrete(1, "p" + str(i) + "_acts_next", next_idx))

            for i in range(max(self.ALL_ROUNDS_LIST) + 1):  # ...................................... round onehot
                _table_space.append(get_discrete(1, "round_" + Poker.INT2STRING_ROUND[i], next_idx)),

            for i in range(self.N_SEATS):  # ....................................................... side pots
                _table_space.append(get_new_box("side_pot_" + str(i), next_idx, 1))

            # add to parts_dict for possible slicing for agents.
            obs_parts_idxs_dict["table_state"] += list(range(_k, next_idx[0]))

            # __________________________  Public Information About Each Player  ________________________
            _player_space = []
            for i in range(self.N_SEATS):
                _k = next_idx[0]
                _player_space += [
                    get_new_box("stack_p" + str(i), next_idx, self.N_SEATS),  # ..................... stack
                    get_new_box("curr_bet_p" + str(i), next_idx, self.N_SEATS),  # .................. current_bet
                    get_discrete(1, "has_folded_this_episode_p" + str(i), next_idx),  # ............. folded_this_epis
                    get_discrete(1, "is_allin_p" + str(i), next_idx),  # ............................ is_allin
                ]
                for j in range(self.N_SEATS):
                    _player_space.append(
                        get_discrete(1, "side_pot_rank_p" + str(i) + "_is_" + str(j), next_idx))  # . side_pot_rank

                # add to parts_dict for possible slicing for agents
                obs_parts_idxs_dict["players"][i] += list(range(_k, next_idx[0]))

            # _______________________________  Public cards (i.e. board)  ______________________________
            _board_space = []
            _k = next_idx[0]
            for i in range(self.N_TOTAL_BOARD_CARDS):

                x = []
                for j in range(self.N_RANKS):  # .................................................... rank
                    x.append(get_discrete(1, str(i) + "th_board_card_rank_" + str(j), next_idx))

                for j in range(self.N_SUITS):  # .................................................... suit
                    x.append(get_discrete(1, str(i) + "th_board_card_suit_" + str(j), next_idx))

                _board_space += x

            # add to parts_dict for possible slicing for agents
            obs_parts_idxs_dict["board"] += list(range(_k, next_idx[0]))

            # __________________________  Return Complete _Observation Space  __________________________
            # Tuple (lots of spaces.Discrete and spaces.Box)
            _observation_space = spaces.Tuple(_table_space + _player_space + _board_space)
            _observation_space.shape = [len(_observation_space.spaces)]
        return _observation_space, obs_idx_dict, obs_parts_idxs_dict

    def _init_from_args(self, env_args, is_evaluating):
        a = copy.deepcopy(env_args)

        # Heads-Up rules
        if a.n_seats == 2:
            self.BTN_POS = 0
            self.SB_POS = 0
            self.BB_POS = 1

        # >2 player poker rules
        else:
            self.BTN_POS = 0
            self.SB_POS = 1
            self.BB_POS = 2

        self._USE_SIMPLE_HU_OBS = a.use_simplified_headsup_obs
        self.RETURN_PRE_TRANSITION_STATE_IN_INFO = a.RETURN_PRE_TRANSITION_STATE_IN_INFO
        self.N_SEATS = int(a.n_seats)

        try:
            self.MAX_CHIPS = sum(a.starting_stack_sizes_list) \
                             + a.stack_randomization_range[1] * a.n_seats \
                             + 1
        except TypeError:  # stack size set to None -> Default
            self.MAX_CHIPS = a.n_seats * (self.DEFAULT_STACK_SIZE + a.stack_randomization_range[1]) + 1

        self.STACK_RANDOMIZATION_RANGE = a.stack_randomization_range

        if a.scale_rewards:
            try:
                self.REWARD_SCALAR = float(sum(a.starting_stack_sizes_list)) / float(a.n_seats) / 5
            except TypeError:  # stack size set to None -> Default
                self.REWARD_SCALAR = self.DEFAULT_STACK_SIZE / 5.0

        else:
            self.REWARD_SCALAR = 1.0

        # fill seats with players
        self.seats = [
            PokerPlayer(seat_id=i,
                        poker_env=self,
                        is_evaluating=is_evaluating,
                        starting_stack=
                        (a.starting_stack_sizes_list[i]
                         if a.starting_stack_sizes_list[i] is not None
                         else self.DEFAULT_STACK_SIZE),
                        stack_randomization_range=a.stack_randomization_range)
            for i in range(a.n_seats)]

    # __________________________________________________ TO OVERRIDE ___________________________________________________
    def get_hand_rank(self, hand_2d, board_2d):
        """

        Args:
            hand_2d (np.ndarray):       the hand to evaluate
            board_2d (np.ndarray):      the board to evaluate

        Returns:
            int: handrank, higher is better.

        """
        raise NotImplementedError

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        This function allows computing the hand rank for all possible hands on multiple boards at once.

        Args:
            boards_1d (np.ndarray):     an array of 1D board representations
            lut_holder:                 a LUT associated with this type of environment.
        """
        raise NotImplementedError

    def _get_env_adjusted_action_formulation(self, action):
        """
        A Discretized PokerEnv subclass has a different action representation than a Limit or a No-Limit env.
        This function should be used to convert these different action spaces to the standard one of PokerEnv:
        Tuple(Discrete(3), Discrete(n_chips))
        Or in words: (action_to_take, n_chips_to_bet_if_action_is_bet).
        n_chips_to_bet_if_action_is_bet always needs to be set but is ignored if the action is not Poker.BET_RAISE

        Args:
            action: subclass specific action representation

        Returns:
            Tuple:  (action_to_take, n_chips_to_bet_if_action_is_bet)
        """
        return action

    def _adjust_raise(self, raise_total_amount_in_chips):
        """
        Different per game type
        Args:
            raise_total_amount_in_chips: chips intended to raise

        Returns:
            Chips to raise to according to gametype and ""raise_total_amount_in_chips"".
        """
        raise NotImplementedError

    @staticmethod
    def get_lut_holder():
        """ return an instance of a lutholder specific to the game's rules"""
        raise NotImplementedError

    # _____________________________________________________ POKER ______________________________________________________
    def _deal_hole_cards(self):
        for player in self.seats:
            player.hand = self.deck.draw(self.N_HOLE_CARDS)

    def _deal_flop(self):
        self.board[:self.N_FLOP_CARDS] = self.deck.draw(self.N_FLOP_CARDS)

    def _deal_turn(self):
        self.board[self.N_FLOP_CARDS:self.N_FLOP_CARDS + self.N_TURN_CARDS] = self.deck.draw(self.N_TURN_CARDS)

    def _deal_river(self):
        d = self.N_FLOP_CARDS + self.N_TURN_CARDS
        self.board[d:d + self.N_RIVER_CARDS] = self.deck.draw(self.N_RIVER_CARDS)

    def _post_antes(self):
        for s in self.seats:
            s.bet_raise(self.ANTE)
            s.has_acted_this_round = False

    def _post_small_blind(self):
        player = self.seats[self.SB_POS]
        player.bet_raise(self.SMALL_BLIND)
        player.has_acted_this_round = False

    def _post_big_blind(self):
        player = self.seats[self.BB_POS]
        player.bet_raise(self.BIG_BLIND)
        player.has_acted_this_round = False

    def _payout_pots(self):
        self._assign_hand_ranks_to_all_players()

        if self.N_SEATS == 2:
            if self.seats[0].hand_rank > self.seats[1].hand_rank:
                self.seats[0].award(self.main_pot)
            elif self.seats[0].hand_rank < self.seats[1].hand_rank:
                self.seats[1].award(self.main_pot)
            else:
                # in HU the number of chips is always even because both players had to put the same amount in.
                self.seats[0].award(self.main_pot / 2)
                self.seats[1].award(self.main_pot / 2)

            self.main_pot = 0

        else:
            pots = np.array([self.main_pot] + self.side_pots)  # appends mainpot as idx 0 to sidepots
            pot_ranks = np.arange(start=-1, stop=len(self.side_pots))
            pot_and_pot_ranks = np.array((pots, pot_ranks)).T

            for e in pot_and_pot_ranks:
                pot = e[0]
                rank = e[1]
                eligible_players = [p for p in self.seats if p.side_pot_rank >= rank and not p.folded_this_episode]

                num_eligible = len(eligible_players)
                if num_eligible > 0:

                    # side_pot / num_winners could be non-int! if so, we decide randomly who to give the chip to.
                    winner_list = self._get_winner_list(players_to_consider=eligible_players)  # PokerPlayer objects
                    num_winners = int(len(winner_list))

                    chips_per_winner = int(pot / num_winners)  # rounds down
                    num_non_div_chips = int(pot) % num_winners  # distribute afterwards

                    for p in winner_list:
                        p.award(chips_per_winner)

                    # distribute the rest randomly.
                    shuffled_winner_idxs = np.arange(num_winners)
                    np.random.shuffle(shuffled_winner_idxs)
                    for p_idx in shuffled_winner_idxs[:num_non_div_chips]:
                        self.seats[p_idx].award(1)

            # set all 0
            self.side_pots = [0] * self.N_SEATS
            self.main_pot = 0

    def _pay_all_to_one_player(self, player_to_pay_to):
        """
        IGNORES SIDEPOT RANKS

        Args:
            player_to_pay_to (PokerPlayer)
        """
        for seat in self.seats:
            player_to_pay_to.award(seat.current_bet)
            seat.current_bet = 0

        player_to_pay_to.award(sum(self.side_pots))
        self.side_pots = [0] * self.N_SEATS

        player_to_pay_to.award(self.main_pot)
        self.main_pot = 0

    def _assign_hand_ranks_to_all_players(self):
        for player in self.seats:
            player.hand_rank = self.get_hand_rank(hand_2d=player.hand, board_2d=self.board)

    def _put_current_bets_into_main_pot_and_side_pots(self):

        if self.N_SEATS == 2:
            # If there is a difference between biggest and 2nd biggest current_bet, award that to the player.
            dif_p0_to_p1 = self.seats[0].current_bet - self.seats[1].current_bet

            if dif_p0_to_p1 > 0:
                self.seats[0].refund_from_bet(dif_p0_to_p1)
            elif dif_p0_to_p1 < 0:
                self.seats[1].refund_from_bet(-dif_p0_to_p1)

            self.main_pot += self.seats[0].current_bet
            self.main_pot += self.seats[1].current_bet
            self.seats[0].current_bet = 0
            self.seats[1].current_bet = 0

        else:
            # If there is a difference between biggest and 2nd biggest current_bet, award that to the player.
            _players_sorted_by_bet_in_front = sorted(self.seats, key=lambda x: x.current_bet, reverse=True)

            dif = _players_sorted_by_bet_in_front[0].current_bet - _players_sorted_by_bet_in_front[1].current_bet
            _players_sorted_by_bet_in_front[0].refund_from_bet(dif)

            # ________________________________________ fill main_pot ________________________________________
            players_not_folded = [p for p in self.seats if not p.folded_this_episode]

            # The smallest bet of someone who did not fold.
            # If no one is allin, all nonfold bets are equal and put into main pot
            main_pot_max_amount = min([p.current_bet for p in players_not_folded])

            for p in self.seats:
                # if bigger needs sidepot. if smaller means he folded at some point and just did not contribute as much
                amount_contributed = min(p.current_bet, main_pot_max_amount)

                self.main_pot += amount_contributed
                p.current_bet -= amount_contributed

            # _____________________________________ maybe fill side pots ______________________________________
            def _find_next_smallest_bet():
                """
                Returns:
                    if all current_bets are 0: None
                    otherwise: idx of smallest nonzero bet

                """
                current_bets = [p.current_bet for p in self.seats]
                next_bet_idx = None

                for b_idx in range(self.N_SEATS):
                    if current_bets[b_idx] > 0:

                        # and not self.seats[b_idx].folded_this_episode   is safe because a folded player can't
                        if ((next_bet_idx is None or current_bets[b_idx] < current_bets[next_bet_idx])
                            and not self.seats[b_idx].folded_this_episode):
                            next_bet_idx = b_idx

                return next_bet_idx

            idx_smallest_bet = _find_next_smallest_bet()  # get status after main pot calc

            while idx_smallest_bet is not None:
                current_max_side_pot_rank = max([p.side_pot_rank for p in self.seats])  # -1 on default.
                side_pot_idx = current_max_side_pot_rank + 1

                side_pot_amount_per_player_in_it = self.seats[idx_smallest_bet].current_bet

                players_not_all_in_after_this_cleanup = [p for p in self.seats if not (
                    p.current_bet < side_pot_amount_per_player_in_it and p.is_allin)]

                for p in players_not_all_in_after_this_cleanup:
                    p.side_pot_rank = side_pot_idx

                # fill this sidepot
                for p in self.seats:
                    # if bigger: needs another sidepot
                    # if smaller: means he folded at some point and just did not contribute as much
                    amount_contributed = min(p.current_bet, side_pot_amount_per_player_in_it)
                    self.side_pots[side_pot_idx] += amount_contributed
                    p.current_bet -= amount_contributed

                # for next iteration
                idx_smallest_bet = _find_next_smallest_bet()  # get status after main pot calc

    def _rundown(self):
        while True:
            self.current_round += 1

            if self.current_round == self.ALL_ROUNDS_LIST[-1] + 1:
                self._put_current_bets_into_main_pot_and_side_pots()
                self.current_round -= 1  # we still want to be on the River.

                if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                    state_before_transition = self.state_dict()

                self._payout_pots()

                if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                    return state_before_transition
                return

            elif self.current_round == Poker.FLOP:
                self._deal_flop()
            elif self.current_round == Poker.TURN:
                self._deal_turn()
            elif self.current_round == Poker.RIVER:
                self._deal_river()
            else:
                raise ValueError(self.current_round)

    def _deal_next_round(self):
        """
        Call this AFTER round+=1
        """
        if self.current_round == Poker.PREFLOP:
            self._deal_hole_cards()
        elif self.current_round == Poker.FLOP:
            self._deal_flop()
        elif self.current_round == Poker.TURN:
            self._deal_turn()
        elif self.current_round == Poker.RIVER:
            self._deal_river()
        else:
            raise ValueError(self.current_round)

    def _next_round(self):
        if self.IS_FIXED_LIMIT_GAME:
            self.n_raises_this_round = 0

        # refer to #ID_2 in docstring of this class for this.
        self.capped_raise.reset()

        # sort out mainpot, sidepots and p.currentbets
        self._put_current_bets_into_main_pot_and_side_pots()

        # This must be called BEFORE round += 1
        self.current_player = self._get_first_to_act_post_flop()

        # set has_acted_this_round = False and maybe others
        for p in self.seats:
            p.has_acted_this_round = False

        self.current_round += 1  # highly dependant on PokerEnv.""ROUND_NAME"" being sequential ints!
        self._deal_next_round()

    def _step(self, processed_action):
        """
        the action passed is considered to be for self.current_player and come from its respective agent (if applicable
        to your algorithm).

        actions are always of the form [action_idx, _raise_size]
        However _raise_size is only considered when the action is Poker.BET_RAISE
        raise_size is measured in total chips as current_bet. Not as an addition to current_bet

        Args:
            processed_action (tuple or list): (action_idx, raise_size)

        Returns:
            obs, rew_for_all_players, done?, info
        """

        # After this call, this fn assumes that executing the action is legal.
        processed_action = self._get_fixed_action(action=processed_action)

        if processed_action[0] == Poker.CHECK_CALL:
            self.current_player.check_call(total_to_call=processed_action[1])

        elif processed_action[0] == Poker.FOLD:
            self.current_player.fold()

        elif processed_action[0] == Poker.BET_RAISE:

            # This happens when someone has fewer chips than minraise and goes all in.
            # The last raiser, if there is one, can't reopen in this case until someone else reraises!
            if processed_action[1] < self._get_current_total_min_raise():
                self.capped_raise.happened_this_round = True
                self.capped_raise.player_that_raised = self.current_player
                self.capped_raise.player_that_cant_reopen = self.last_raiser  # might be None. Then everyone can raise.

            elif self.capped_raise.happened_this_round is True:
                # if someone shoved under minraise over someone else's raise the original raiser can't reraise again!
                # But if a 3rd plyr raises over that under-min shove, everyone can raise again. this is handled here.
                if self.capped_raise.player_that_cant_reopen is not self.current_player:
                    self.capped_raise.reset()

            self.last_raiser = self.current_player  # leave this line at the end of this function!!
            self.current_player.bet_raise(total_bet_amount=processed_action[1])

            self.n_actions_this_episode += 1

            # If this is a limit poker game, increment the raise counter
            if self.IS_FIXED_LIMIT_GAME:
                self.n_raises_this_round += 1
        else:
            raise RuntimeError(processed_action[0], " is not legal")

        self.last_action = [processed_action[0], processed_action[1], self.current_player.seat_id]

        # ______________________________________________________________________________________________________________
        # check if should deal next round, rundown or continue to next step in the episode of the poker game

        all_non_all_in_and_non_fold_p = [p for p in self.seats if not p.folded_this_episode and not p.is_allin]
        all_nonfold_p = [p for p in self.seats if not p.folded_this_episode]

        # just let next player run in this round
        info = None
        if self._should_continue_in_this_round(all_non_all_in_and_non_fold_p=all_non_all_in_and_non_fold_p,
                                               all_nonfold_p=all_nonfold_p):
            self.current_player = self._get_player_that_has_to_act_next()
            is_terminal = False

            if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                info = {"chance_acts": False, "state_dict_before_money_move": None}

        # next round
        elif len(all_non_all_in_and_non_fold_p) > 1:

            # payout if final round
            if self.current_round == len(self.ALL_ROUNDS_LIST) - 1:
                is_terminal = True
                self._put_current_bets_into_main_pot_and_side_pots()
                if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                    info = {"chance_acts": False, "state_dict_before_money_move": self.state_dict()}
                self._payout_pots()

            # deal next round
            else:
                is_terminal = False
                if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                    info = {"chance_acts": True, "state_dict_before_money_move": self.state_dict()}
                self._next_round()

        # rundown
        elif len(all_nonfold_p) > 1:  # rundown only makes sense if >0 are allin and 1 is not or >2 are allin.
            is_terminal = True
            state_before_payouts = self._rundown()

            if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                info = {"chance_acts": False, "state_dict_before_money_move": state_before_payouts}

        # only one not folded, so pay all pots to him.
        elif len(all_nonfold_p) == 1:
            is_terminal = True
            if self.RETURN_PRE_TRANSITION_STATE_IN_INFO:
                self._put_current_bets_into_main_pot_and_side_pots()
                info = {"chance_acts": False, "state_dict_before_money_move": self.state_dict()}
                self._payout_pots()
            else:  # more efficient, but doesnt give info needed.
                self._pay_all_to_one_player(all_nonfold_p[0])

        else:
            raise RuntimeError("There seems to be an edge-case not built into this")

        return self._get_current_step_returns(is_terminal=is_terminal, info=info)

    # _____________________________________________________ UTIL  ______________________________________________________
    def _get_winner_list(self, players_to_consider):
        """
        Returns:
            list: list of PokerPlayer instances that are winners

        """
        best_rank = max([p.hand_rank for p in players_to_consider])
        winners = [p for p in players_to_consider if p.hand_rank == best_rank]

        return winners

    def _get_current_total_min_raise(self):
        """
        Taking the highest and 2nd highest and subtracting them gives us the minraise amount. If all bets are equal,
        we return with a delta of 1 big blind.
        """

        if self.N_SEATS == 2:
            _sorted_ascending = sorted([p.current_bet for p in self.seats])  # 0 is small, 1 is big
            delta = max(_sorted_ascending[1] - _sorted_ascending[0], self.BIG_BLIND)
            return _sorted_ascending[1] + delta

        else:
            current_bets_sorted_descending = sorted([p.current_bet for p in self.seats], reverse=True)
            current_to_call_total = max(current_bets_sorted_descending)
            _largest_bet = current_bets_sorted_descending[0]

            for i in range(1, self.N_SEATS):
                if current_bets_sorted_descending[i] == _largest_bet:
                    continue

                delta_between_last_and_before_last = _largest_bet - current_bets_sorted_descending[i]

                delta = max(delta_between_last_and_before_last, self.BIG_BLIND)
                return current_to_call_total + delta

            # in cases where all bets are equal, the minraise delta is 1 big blind
            return current_to_call_total + self.BIG_BLIND

    def _get_new_board(self):
        return np.full((self.N_TOTAL_BOARD_CARDS, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)

    def _get_first_to_act_pre_flop(self):
        if self.N_SEATS >= 4:
            # left of BB
            return self.seats[3]
        else:
            # for n_players==3 and 2 btn starts
            return self.seats[0]

    def _get_first_to_act_post_flop(self):
        """
        Btn has index 0. He is always the last with the exception of some HU rules where ""BTN_IS_FIRST_POSTFLOP"" can
        be set to True. In multi-agent games, we search for the smalles seat id in the list, while 0 (i.e. btn) is
        treated as inf.
        """

        if self.N_SEATS == 2:
            if self.BTN_IS_FIRST_POSTFLOP:
                return self.seats[0]
            else:
                return self.seats[1]
        else:
            players_to_consider = [p for p in self.seats if not p.folded_this_episode and not p.is_allin]

            # since there will always be at least 2 ppl in the pot, the btn can NEVER be the first!
            first_p = players_to_consider[0]
            for p in players_to_consider:
                if p.seat_id < first_p.seat_id or first_p.seat_id == 0:
                    first_p = p

            return first_p

    def _get_biggest_bet_out_there_aka_total_to_call(self):
        """
        chip count of max([p.current_bet for p in self.seats])
        """
        return max([p.current_bet for p in self.seats])

    def _get_player_that_has_to_act_next(self):
        idx = self.seats.index(self.current_player) + 1

        for i in range(self.N_SEATS):
            mod_idx = idx % self.N_SEATS
            p = self.seats[mod_idx]

            if not p.is_allin and not p.folded_this_episode:
                return self.seats[mod_idx]

            idx += 1

        raise RuntimeError("There is no next player. Seems like some more debugging is needed...")

    def _get_fixed_action(self, action):
        """
        This method is responsible for asserting that an action is valid at the current state and returns the
        capped/changed action if not.

        Args:
            action (iterable):      iterable of 2 ints - [PokerEnv.ACTIONTYPE, _raise_amount_in_chips]

        Returns:            [Poker.FOLD, -1]
                        or  [Poker.CHECK_CALL, total_bet_to_be_placed_in_front_by_player]
                        or  [Poker.BET_RAISE, total_bet_to_be_placed_in_front_by_player]

        """
        _action_idx = action[0]

        total_to_call = self._get_biggest_bet_out_there_aka_total_to_call()

        if _action_idx == Poker.FOLD:
            if total_to_call <= self.current_player.current_bet:
                return self._process_check_call(total_to_call=total_to_call)
            else:
                return [Poker.FOLD, -1]

        elif _action_idx == Poker.CHECK_CALL:
            if (self.FIRST_ACTION_NO_CALL
                and (self.n_actions_this_episode == 0)
                and self.current_round == Poker.PREFLOP):
                return [Poker.FOLD, -1]

            return self._process_check_call(total_to_call=total_to_call)

        elif _action_idx == Poker.BET_RAISE:

            # Limit Poker specific rule
            if self.IS_FIXED_LIMIT_GAME:
                if self.n_raises_this_round >= self.MAX_N_RAISES_PER_ROUND[self.current_round]:
                    return self._process_check_call(total_to_call=total_to_call)

            if ((self.current_player.stack + self.current_player.current_bet <= total_to_call)
                or (self.capped_raise.player_that_cant_reopen is self.current_player)):
                return self._process_check_call(total_to_call=total_to_call)
            else:
                return self._process_raise(raise_total_amount_in_chips=action[1])
        else:
            raise RuntimeError('invalid action ({}) must be fold (0), call (1), or raise (2) '.format(_action_idx))

    def _process_check_call(self, total_to_call):
        delta_to_call = min(total_to_call - self.current_player.current_bet, self.current_player.stack)
        total_bet_to_be_placed = int(delta_to_call + self.current_player.current_bet)
        return [Poker.CHECK_CALL, total_bet_to_be_placed]

    def _process_raise(self, raise_total_amount_in_chips):
        raise_to = self._adjust_raise(raise_total_amount_in_chips=raise_total_amount_in_chips)
        # lastly, if that amount is too much, raise all in
        if self.current_player.current_bet + self.current_player.stack < raise_to:
            raise_to = self.current_player.stack + self.current_player.current_bet
        return [Poker.BET_RAISE, int(raise_to)]

    def _should_continue_in_this_round(self, all_non_all_in_and_non_fold_p, all_nonfold_p):
        """ util function used in ._step() """

        # only 1 player did not fold yet
        if len(all_nonfold_p) < 2:
            return False

        largest_bet = max([p.current_bet for p in self.seats])
        if len([p for p in all_nonfold_p if p.is_allin or p.current_bet == largest_bet]) == len(all_nonfold_p) \
            and len([p for p in all_non_all_in_and_non_fold_p if not p.has_acted_this_round]) == 0:
            return False
        return True

    # _____________________________________________________ OUTPUT _____________________________________________________
    def _get_current_step_returns(self, is_terminal, info):
        obs = self.get_current_obs(is_terminal)
        reward = self._get_step_reward(is_terminal)
        return obs, reward, is_terminal, info

    def _get_player_states_all_players(self, normalization_sum):
        """ Public Information About Each Player """
        if (self.N_SEATS == 2) and self._USE_SIMPLE_HU_OBS:
            player_states = []
            for player in self.seats:
                player_states += [
                    player.stack / normalization_sum,
                    player.current_bet / normalization_sum,
                    player.is_allin
                ]

        else:
            player_states = []
            for player in self.seats:
                player_states += [
                    player.stack / normalization_sum,
                    player.current_bet / normalization_sum,
                    player.folded_this_episode,
                    player.is_allin
                ]
                x = [0] * self.N_SEATS
                if player.side_pot_rank > 0:
                    x[int(player.side_pot_rank)] = 1
                player_states += x

        return player_states

    def _get_board_state(self, ):
        K = (self.N_RANKS + self.N_SUITS)
        _board_space = [0] * (self.N_TOTAL_BOARD_CARDS * K)
        for i, card in enumerate(self.board.tolist()):
            if card[0] == Poker.CARD_NOT_DEALT_TOKEN_1D:
                break
            D = K * i
            _board_space[card[0] + D] = 1

            # only include suit in obs if it is relevant to the game rules played
            if self.SUITS_MATTER:
                _board_space[card[1] + D + self.N_RANKS] = 1

        return _board_space

    def _get_table_state(self, normalization_sum):
        if (self.N_SEATS == 2) and self._USE_SIMPLE_HU_OBS:
            community_state = [
                self.ANTE / normalization_sum,
                self.SMALL_BLIND / normalization_sum,
                self.BIG_BLIND / normalization_sum,
                self._get_current_total_min_raise() / normalization_sum,
                self.main_pot / normalization_sum,
                self._get_biggest_bet_out_there_aka_total_to_call() / normalization_sum,
                self.last_action[1] / normalization_sum if self.last_action[0] is not None else 0,
            ]

            x_what = [0] * 3  # last action what
            x_who = [0] * self.N_SEATS  # last action who
            if self.last_action[0] is not None:
                x_who[self.last_action[2]] = 1
                x_what[self.last_action[0]] = 1
            community_state += x_what + x_who

            # who acts next?
            x = [0] * self.N_SEATS
            x[self.current_player.seat_id] = 1
            community_state += x

            # current round
            x = [0] * (self.ALL_ROUNDS_LIST[-1] + 1)
            x[self.current_round] = 1
            community_state += x

        else:
            community_state = [
                self.ANTE / normalization_sum,
                self.SMALL_BLIND / normalization_sum,
                self.BIG_BLIND / normalization_sum,
                self._get_current_total_min_raise() / normalization_sum,
                self.main_pot / normalization_sum,
                self._get_biggest_bet_out_there_aka_total_to_call() / normalization_sum,
                self.last_action[1] / normalization_sum if self.last_action[0] is not None else 0,
            ]

            x_what = [0] * 3  # last action what
            x_who = [0] * self.N_SEATS  # last action who
            if self.last_action[0] is not None:
                x_who[self.last_action[2]] = 1
                x_what[self.last_action[0]] = 1
            community_state += x_what + x_who

            # who acts next?
            x = [0] * self.N_SEATS
            x[self.current_player.seat_id] = 1
            community_state += x

            # current round
            x = [0] * (self.ALL_ROUNDS_LIST[-1] + 1)
            x[self.current_round] = 1
            community_state += x

            # side_pots
            if self.N_SEATS > 2:
                community_state += [sp / normalization_sum for sp in self.side_pots]
            else:
                community_state += [0] * self.N_SEATS

        return community_state

    def _get_step_reward(self, is_terminal):
        if not is_terminal:
            return np.zeros(shape=self.N_SEATS, dtype=np.float32)
        return [(p.stack - p.starting_stack_this_episode) / self.REWARD_SCALAR for p in self.seats]

    # ______________________________________________________ API _______________________________________________________
    def reset(self, deck_state_dict=None):
        """
        Resets the state of the game to the standard beginning of the episode. If specified in the args passed,
        stack size randomization is applied in the new episode. If deck_state_dict is not None, the cards
        and associated random variables are synchronized FROM the given environment, so that when .step() is called on
        each of them, they produce the same result.

        Args:
            deck_state_dict:       Optional.
                                            If an instance of a PokerEnv subclass is passed, the deck, holecards, and
                                            board in this instance will be synchronized from the handed env cls.
        """
        if self.IS_FIXED_LIMIT_GAME:
            if self.BIG_BLIND > 0:
                self.n_raises_this_round = 1  # big blind counts, but in ante-only games like LEDUC it doesn't count
            else:
                self.n_raises_this_round = 0

        # reset table
        self.side_pots = [0] * self.N_SEATS  # chip count in side pots
        self.main_pot = 0  # chip count in main pot
        self.board = self._get_new_board()
        self.last_action = [None, None, None]  # list of 3 ints: [action_idx, _raise_amount, player.seat_id]
        self.current_round = self.ALL_ROUNDS_LIST[0]
        self.capped_raise.reset()
        self.last_raiser = None
        self.n_actions_this_episode = 0

        # players
        for p in self.seats:
            p.reset()

        # reset deck
        self.deck.reset()

        # start new game
        self._post_antes()
        self._put_current_bets_into_main_pot_and_side_pots()  # antes don't count to current bet
        self._post_small_blind()
        self._post_big_blind()
        self.current_player = self._get_first_to_act_pre_flop()
        self._deal_next_round()

        # optionally synchronize random variables from another env
        if deck_state_dict is not None:
            self.load_cards_state_dict(cards_state_dict=deck_state_dict)

        return self._get_current_step_returns(is_terminal=False, info=[False, None])

    def step_raise_pot_frac(self, pot_frac):
        """
        This fn is only useful to call if current_player wants to raise. Therefore it assumes that's the case.

        Args:
            pot_frac: fraction of current pot to bet/raise

        Returns:
            obs, rew_for_all_players, done?, info:

        """
        processed_action = (2, self.get_fraction_of_pot_raise(fraction=pot_frac, player_that_bets=self.current_player))
        return self._step(processed_action=processed_action)

    def step_from_processed_tuple(self, action):
        """
        Args:
            action:  processed_action (tuple or list): (action_idx, raise_size)

        Returns:
            obs, rew_for_all_players, done?, info
        """
        return self._step(action)

    def step(self, action):
        """
        Args:
            action: env specific action representation as documented in PokerEnv.

        Returns:
            obs, rew_for_all_players, done?, info
        """

        # cap min/max raises and format action. after that the action is legal 100% and can be executed blindly
        processed_action = self._get_env_adjusted_action_formulation(action)
        return self._step(processed_action=processed_action)

    def state_dict(self):
        env_state_dict = {
            EnvDictIdxs.is_evaluating: self.IS_EVALUATING,
            EnvDictIdxs.current_round: self.current_round,  # int by value
            EnvDictIdxs.side_pots: copy.deepcopy(self.side_pots),  # np array
            EnvDictIdxs.main_pot: self.main_pot,  # int by value
            EnvDictIdxs.board_2d: np.copy(self.board),  # np array
            EnvDictIdxs.last_action: copy.deepcopy(self.last_action),
            EnvDictIdxs.capped_raise: [self.capped_raise.player_that_raised.seat_id,
                                       None
                                       if self.capped_raise.player_that_cant_reopen is None
                                       else
                                       self.capped_raise.player_that_cant_reopen.seat_id]
            if self.capped_raise.happened_this_round else None,
            EnvDictIdxs.current_player: self.current_player.seat_id,  # idx in _env.seats
            EnvDictIdxs.last_raiser: None if self.last_raiser is None else self.last_raiser.seat_id,
            # idx in _env.seats
            EnvDictIdxs.deck: self.deck.state_dict(),  # np array
            EnvDictIdxs.n_actions_this_episode: self.n_actions_this_episode,  # int
            EnvDictIdxs.seats:
                [
                    {
                        PlayerDictIdxs.seat_id: p.seat_id,  # int
                        PlayerDictIdxs.hand: np.copy(p.hand),  # np array
                        PlayerDictIdxs.hand_rank: p.hand_rank,  # int by value
                        PlayerDictIdxs.stack: p.stack,  # int by value
                        PlayerDictIdxs.current_bet: p.current_bet,  # int by value
                        PlayerDictIdxs.is_allin: p.is_allin,  # bool by value
                        PlayerDictIdxs.folded_this_episode: p.folded_this_episode,  # bool by value
                        PlayerDictIdxs.has_acted_this_round: p.has_acted_this_round,  # bool by value
                        PlayerDictIdxs.side_pot_rank: p.side_pot_rank  # int by value
                    }
                    for p in self.seats]
        }
        if self.IS_FIXED_LIMIT_GAME:
            env_state_dict[EnvDictIdxs.n_raises_this_round] = self.n_raises_this_round
        return env_state_dict

    def load_state_dict(self, env_state_dict, blank_private_info=False):
        """

        Args:
            env_state_dict:
            blank_private_info (bool): If true, hole cards are going to be set to None. This is useful when loading a
                                        public state

        Returns:

        """
        self.IS_EVALUATING = env_state_dict[EnvDictIdxs.is_evaluating]
        self.current_round = env_state_dict[EnvDictIdxs.current_round]
        self.side_pots = copy.deepcopy(env_state_dict[EnvDictIdxs.side_pots])
        self.main_pot = env_state_dict[EnvDictIdxs.main_pot]
        self.board = np.copy(env_state_dict[EnvDictIdxs.board_2d])
        self.last_action = copy.deepcopy(env_state_dict[EnvDictIdxs.last_action])

        x = env_state_dict[EnvDictIdxs.capped_raise]
        self.capped_raise = CappedRaise()
        if x is not None:
            self.capped_raise.happened_this_round = True
            self.capped_raise.player_that_raised = self.seats[x[0]]
            if x[1] is None:
                self.capped_raise.player_that_cant_reopen = None
            else:
                self.capped_raise.player_that_cant_reopen = self.seats[x[1]]

        self.current_player = self.seats[env_state_dict[EnvDictIdxs.current_player]]

        x = env_state_dict[EnvDictIdxs.last_raiser]
        self.last_raiser = None if x is None else self.seats[x]
        self.deck.load_state_dict(env_state_dict[EnvDictIdxs.deck])
        self.n_actions_this_episode = env_state_dict[EnvDictIdxs.n_actions_this_episode]

        if self.IS_FIXED_LIMIT_GAME:
            self.n_raises_this_round = env_state_dict[EnvDictIdxs.n_raises_this_round]

        for p in self.seats:
            p.seat_id = env_state_dict[EnvDictIdxs.seats][p.seat_id][PlayerDictIdxs.seat_id]
            p.stack = env_state_dict[EnvDictIdxs.seats][p.seat_id][PlayerDictIdxs.stack]
            p.current_bet = env_state_dict[EnvDictIdxs.seats][p.seat_id][PlayerDictIdxs.current_bet]
            p.is_allin = env_state_dict[EnvDictIdxs.seats][p.seat_id][PlayerDictIdxs.is_allin]
            p.folded_this_episode = env_state_dict[EnvDictIdxs.seats][p.seat_id][PlayerDictIdxs.folded_this_episode]
            p.has_acted_this_round = env_state_dict[EnvDictIdxs.seats][p.seat_id][PlayerDictIdxs.has_acted_this_round]
            p.side_pot_rank = env_state_dict[EnvDictIdxs.seats][p.seat_id][PlayerDictIdxs.side_pot_rank]

            if blank_private_info:
                p.hand = None
                p.hand_rank = None
            else:
                p.hand = np.copy(env_state_dict[EnvDictIdxs.seats][p.seat_id][PlayerDictIdxs.hand])
                p.hand_rank = env_state_dict[EnvDictIdxs.seats][p.seat_id][PlayerDictIdxs.hand_rank]

    def get_current_obs(self, is_terminal):
        """
        This function can be useful for manually setting the environment to a desired state and then getting the
        formatted observation from it without actually stepping it.

        Args:
            is_terminal (bool): Whether the last env.step() call returned that it is done or not

        Returns:
            np.ndarray: current observation as returned by the last env.step() call.

        """
        if is_terminal:
            return np.zeros(shape=self.observation_space.shape, dtype=np.float32)  # terminal is all zero
        normalization_sum = float(sum([s.starting_stack_this_episode for s in self.seats])) / self.N_SEATS
        return np.array(self._get_table_state(normalization_sum=normalization_sum) \
                        + self._get_player_states_all_players(normalization_sum=normalization_sum) \
                        + self._get_board_state()
                        , dtype=np.float32)

    def print_obs(self, obs):
        print("______________________________________ Printing _Observation _________________________________________")
        names = [e + ":  " for e in list(self.obs_idx_dict.keys())]
        str_len = max([len(e) for e in names])
        for name, key in zip(names, list(self.obs_idx_dict.keys())):
            name = name.rjust(str_len)
            print(name, obs[self.obs_idx_dict[key]])

    def set_to_public_tree_node_state(self, node):
        """
        Since there might be a conflict between ""node""'s {private cards, remaining deck, board} and those of
        the wrapped env, we need rules:

        - Take the board from the node
        - Private cards from the environment
        - Remaining deck doesn't matter

        This can result in cards appearing multiple times, if misused, but still represents the use-case best.
        """
        self.load_state_dict(node.env_state)

    def cards2str(self, cards_2d, seperator=", "):
        """

        Args:
            cards_2d:           2D representation of any amount of cards
            seperator (str):    token to put between cards when printing

        Returns:
            str

        """
        hand_as_str = ""
        for c in cards_2d:
            if not np.array_equal(c, Poker.CARD_NOT_DEALT_TOKEN_2D):
                hand_as_str += self.RANK_DICT[c[0]]
                hand_as_str += self.SUIT_DICT[c[1]]
                hand_as_str += seperator
        return hand_as_str

    def get_legal_actions(self):
        """

        Returns:
            list:   a list with none, one or multiple actions of the set [FOLD, CHECK/CALL, BET]

        """
        legal_actions = []
        for a in [(Poker.FOLD, -1,), (Poker.CHECK_CALL, -1,)]:
            if self._get_fixed_action(action=a)[0] == a[0]:
                legal_actions.append(a[0])

        a = self._get_env_adjusted_action_formulation(action=(Poker.BET_RAISE, 1,))
        fixed_a = self._get_fixed_action(action=a)
        if a[0] == fixed_a[0]:
            legal_actions.append(Poker.BET_RAISE)

        return legal_actions

    def get_range_idx(self, p_id):
        """
        Args:
            p_id (int):     seat_id of the PokerPlayer whose range_idx (an integer representation of private cards)
                            is requested

        Returns:
            int

        """
        return int(self.lut_holder.get_range_idx_from_hole_cards(
            hole_cards_2d=self.get_hole_cards_of_player(p_id=p_id)))

    def get_random_action(self):
        a = np.random.randint(low=0, high=3)
        pot_sum = sum(self.side_pots) + self.main_pot
        n = int(np.random.normal(loc=pot_sum / 2, scale=pot_sum / 5))

        return a, n

    def get_all_winnable_money(self):
        """

        Returns:
            int: all money in the pot, side pots, and currently bet
        """
        return self.main_pot + sum([p.current_bet for p in self.seats]) + sum(self.side_pots)

    def cards_state_dict(self):
        return {
            "deck": self.deck.state_dict(),
            "board": np.copy(self.board),
            "hand": [np.copy(self.seats[p].hand) for p in range(self.N_SEATS)]
        }

    def load_cards_state_dict(self, cards_state_dict):
        self.deck.load_state_dict(cards_state_dict["deck"])
        self.board = np.copy(cards_state_dict["board"])
        for p in range(self.N_SEATS):
            self.seats[p].hand = np.copy(cards_state_dict["hand"][p])

    def reshuffle_remaining_deck(self):
        self.deck.shuffle()

    def get_fraction_of_pot_raise(self, fraction, player_that_bets):
        """

        Turns a ""fraction_of_pot_raise_size"" into a ""number_of_chips"" raise size.
        We calculate the pot as the sum of all sidepots and the mainpot.

        Args:
            fraction:           fraction of pot to bet
            player_that_bets:   any PokerPlayer in self.seats

        Returns:
            int:                number of chips that produces a raise by the given fraction of the pot
        """
        _player = player_that_bets if isinstance(player_that_bets, PokerPlayer) else self.seats[player_that_bets]
        to_call = self._get_biggest_bet_out_there_aka_total_to_call() - _player.current_bet
        pot_after_call = self.main_pot + sum(self.side_pots) + sum([p.current_bet for p in self.seats]) + to_call

        delta = int(to_call + (pot_after_call * fraction))
        total_raise = delta + _player.current_bet

        return total_raise

    def get_frac_from_chip_amt(self, amt, player_that_bets):
        """
        reverse of self.get_fraction_of_pot_raise()

        Args:
            amt:                number of chips (total bet, not additional!)
            player_that_bets:   any PokerPlayer in self.seats

        Returns:
            float:              fraction of pot that produces a raise by the given amt of chips

        """
        _player = player_that_bets if isinstance(player_that_bets, PokerPlayer) else self.seats[player_that_bets]

        to_call = self._get_biggest_bet_out_there_aka_total_to_call() - _player.current_bet
        pot_after_call = self.main_pot + sum(self.side_pots) + sum([p.current_bet for p in self.seats]) + to_call

        delta = amt - _player.current_bet
        fraction = float(delta - to_call) / float(pot_after_call)

        return fraction

    def get_hole_cards_of_player(self, p_id):
        return self.seats[p_id].hand

    def eval(self):
        """
        Eval mode DOES NOT allow randomizations to be applied.
        """
        self.IS_EVALUATING = True
        for p in self.seats:
            p.IS_EVALUATING = True

    def training(self):
        """
        Allows randomizations as defined in the args passed when creating an env.
        """
        self.IS_EVALUATING = False
        for p in self.seats:
            p.IS_EVALUATING = False

    def render(self, mode='TEXT'):
        """
        When called, renders the current state of the environment

        Args:
            mode (str):   Currently only "TEXT" is supported.

        """

        if mode.upper() == 'TEXT':
            print('\n\n')
            print('___________________________________',
                  (Poker.INT2STRING_ROUND[self.current_round] + " - " + str(
                      self.current_player.seat_id) + " acts").center(15),
                  '___________________________________')

            print("Board: ", self.cards2str(self.board))

            print(("Last Action:   player_" + str(self.last_action[2]) + ": " + str(self.last_action[0]) + " " +
                   str(self.last_action[1]).rjust(113)), "|   Main_pot: ", str(self.main_pot).rjust(7))

            # player information
            for p in self.seats:
                f = "-" if p.folded_this_episode else "+" if p.is_allin else ""
                print((f + "Player_" + str(p.seat_id) + ":").rjust(14) +
                      "stack: ", str(p.stack).rjust(8),
                      "current_bet: ", str(p.current_bet).rjust(8),
                      "side_pot_rank: ", str(p.side_pot_rank).rjust(8),
                      "hand: ", self.cards2str(p.hand).rjust(8),

                      # sidepots
                      " " * 18,
                      '|   Side_pot' + str(p.seat_id) + ": " + str(self.side_pots[p.seat_id]).rjust(6))

            if self.IS_FIXED_LIMIT_GAME:
                print("Num raises this round: ", self.n_raises_this_round)
            print("\n")

        else:
            raise ValueError(mode + "is unsupported")

    def print_tutorial(self):
        print("0 \tFold")
        print("1 \tCall")
        print("2 \tRaise")
        print("If you enter 2 for raise, you will be asked what size you want to raise to TOTAL in case you play a "
              "Poker game with multiple size options")

    def human_api_ask_action(self):
        while True:
            try:
                action_idx = int(
                    input("What action do you want to take as player " + str(self.current_player.seat_id) + "?"))
            except ValueError:
                print("You need to enter one of the allowed actions. Refer to the tutorial please.")
                continue
            if action_idx not in [Poker.FOLD, Poker.CHECK_CALL, Poker.BET_RAISE]:
                print("Invalid action_idx! Please enter one of the allowed actions as described in the tutorial above.")
                continue
            break
        time.sleep(0.01)
        if action_idx == Poker.BET_RAISE:
            raise_size = int(input("To how many chips total do you want to raise?"))
        else:
            raise_size = 0

        action_tuple = [action_idx, raise_size]
        return action_tuple

    def set_stack_size(self, stack_size):
        args = copy.deepcopy(self._args)
        args.starting_stack_sizes_list = copy.deepcopy(stack_size)
        self._init_from_args(env_args=args, is_evaluating=self.IS_EVALUATING)

    def get_args(self):
        return copy.deepcopy(self._args)

    def set_args(self, env_args):
        self._init_from_args(env_args=env_args, is_evaluating=self.IS_EVALUATING)


class CappedRaise:
    """
    There is a special rule in poker games, that says that when player A raises, and player B reraises all-in, but has
    less chips than the minimum raise would actually be, player A cannot rereraise when it is his turn again, while all
    other players can rereraise. This lock is broken when a) someone else rereraises or b) the betting round is over.
    """

    def __init__(self):
        self.happened_this_round = None
        self.player_that_raised = None
        self.player_that_cant_reopen = None
        self.reset()

    def reset(self):
        self.happened_this_round = False
        self.player_that_raised = None
        self.player_that_cant_reopen = None
