# Copyright (c) 2019 Eric Steinberger
import copy

import numpy as np


class InteractiveGame:
    """
    This class facilitates play between a user against an EvalAgent or against himself in any poker game.
    """

    def __init__(self, env_cls, env_args, seats_human_plays_list, eval_agent=None):
        """

        Args:
            env_cls (PokerEnv subclass):    the subclass of PokerEnv (not instance!) that the EvalAgent was trained on
                                            or more generally, you want to play in.

            env_args (PokerEnvArgs, DiscretizedPokerEnvArgs, or LimitPokerEnvArgs):
                                            The arguments object corresponding to the environment

            seats_human_plays_list (list):  a list of ints that indicates which seats on the table are played by the
                                            human in the command line. If you pass an empty list, you can watch the AI
                                            play against itself, or you can pass a list with all seats to play against
                                            yourself. The most common use-case, however, is to pass some seats and
                                            play against the agent.

            eval_agent (EvalAgentBase):     The wrapped agent. You only need to pass one if ""seats_human_plays_list""
                                            doesn't cover all seats on the table.
        """
        if len(seats_human_plays_list) < env_args.n_seats:
            assert eval_agent is not None

        self._env = env_cls(env_args=env_args, is_evaluating=True, lut_holder=env_cls.get_lut_holder())
        self._env.reset()

        self._eval_agent = eval_agent

        self._seats_human_plays_list = seats_human_plays_list
        self._winnings_per_seat = [0 for _ in range(self._env.N_SEATS)]

    @property
    def seats_human_plays_list(self):
        return copy.deepcopy(self._seats_human_plays_list)

    @property
    def winnings_per_seat(self):
        return copy.deepcopy(self._winnings_per_seat)

    def start_to_play(self, render_mode="TEXT", limit_numpy_digits=True):
        if limit_numpy_digits:
            np.set_printoptions(precision=5, suppress=True)
        print("""                       
                                                _____
                    _____                _____ |6    |
                   |2    | _____        |5    || & & | 
                   |  &  ||3    | _____ | & & || & & | _____
                   |     || & & ||4    ||  &  || & & ||7    |
                   |  &  ||     || & & || & & ||____9|| & & | _____
                   |____Z||  &  ||     ||____S|       |& & &||8    | _____
                          |____E|| & & |              | & & ||& & &||9    |
                                 |____h|              |____L|| & & ||& & &|
                                                             |& & &||& & &|
                                                             |____8||& & &|
                                                                    |____6|
               """)
        self._env.print_tutorial()

        # play forever until human player manually stops
        while True:
            print()
            print("****************************")
            print("*        GAME START        *")
            print("****************************")
            print()

            # ______________________________________________ one episode _______________________________________________

            self._env.reset()
            if self._eval_agent is not None:
                # The agent has his own copy of the env within its EvalAgent. The following lines sets the decks between
                # self.env and the agent's env equal. However, the agent cannot see any private cards but his own in his
                # observations, of course!
                self._eval_agent.reset(deck_state_dict=self._env.cards_state_dict())

            self._env.render(mode=render_mode)
            while True:
                current_player_id = self._env.current_player.seat_id

                if self._eval_agent is not None:
                    assert np.array_equal(self._env.board, self._eval_agent._internal_env_wrapper.env.board)
                    assert np.array_equal(np.array(self._env.side_pots),
                                          np.array(self._eval_agent._internal_env_wrapper.env.side_pots))
                    assert self._env.current_player.seat_id == \
                           self._eval_agent._internal_env_wrapper.env.current_player.seat_id
                    assert self._env.current_round == self._eval_agent._internal_env_wrapper.env.current_round

                # Human acts
                if current_player_id in self._seats_human_plays_list:
                    action_tuple = self._env.human_api_ask_action()

                    if self._eval_agent is not None:
                        self._eval_agent.notify_of_processed_tuple_action(action_he_did=action_tuple,
                                                                          p_id_acted=current_player_id)

                # Agent acts
                else:
                    a_idx, frac = self._eval_agent.get_action_frac_tuple(step_env=True)
                    if a_idx == 2:
                        action_tuple = [2, self._env.get_fraction_of_pot_raise(fraction=frac,
                                                                               player_that_bets=current_player_id)]
                    else:
                        action_tuple = [a_idx, -1]

                obs, rews, done, info = self._env._step(processed_action=action_tuple)
                self._env.render(mode=render_mode)

                if done:
                    break

            for s in range(self._env.N_SEATS):
                self._winnings_per_seat[s] += np.rint(rews[s] * self._env.REWARD_SCALAR)

            print("")
            print("Current Winnings per player:", self._winnings_per_seat)
            input("Press Enter to go to the next round.")
