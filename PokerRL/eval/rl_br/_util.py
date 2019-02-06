from PokerRL.game._.rl_env.poker_types.DiscretizedPokerEnv import DiscretizedPokerEnv
from PokerRL.game._.rl_env.poker_types.LimitPokerEnv import LimitPokerEnv
from PokerRL.rl.rl_util import get_builder_from_str, get_env_cls_from_str


def get_env_builder_rlbr(t_prof):
    env_bldr_cls = get_builder_from_str(t_prof.env_builder_cls_str)
    return env_bldr_cls(env_cls=get_env_cls_from_str(t_prof.game_cls_str),
                        env_args=t_prof.module_args["rlbr"].get_rlbr_env_args(
                            agents_env_args=t_prof.module_args["env"]))


def reset_episode_multi_action_space(rlbr_env_wrapper, opponent_agent):
    ret = rlbr_env_wrapper.reset()
    opponent_agent.reset(deck_state_dict=rlbr_env_wrapper.env.cards_state_dict())
    return ret


def notify_agent_multi_action_space(action_int, rlbr_seat_id, rlbr_env_wrapper, opponent_agent):
    _type = type(rlbr_env_wrapper.env)

    if issubclass(_type, LimitPokerEnv):
        opponent_agent.notify_of_action(p_id_acted=rlbr_seat_id, action_he_did=action_int)

    elif issubclass(_type, DiscretizedPokerEnv):
        if action_int >= 2:
            raise_frac = rlbr_env_wrapper.env.bet_sizes_list_as_frac_of_pot[action_int - 2]
            opponent_agent.notify_of_raise_frac_action(p_id_acted=rlbr_seat_id, frac=raise_frac)
        else:
            opponent_agent.notify_of_action(p_id_acted=rlbr_seat_id, action_he_did=action_int)

    else:
        raise ValueError(_type)


def step_from_opp_action(action_int, opponent, rlbr_env_wrapper):
    _type = type(rlbr_env_wrapper.env)

    if issubclass(_type, LimitPokerEnv):
        return rlbr_env_wrapper.step(action=action_int)

    elif issubclass(_type, DiscretizedPokerEnv):
        if action_int >= 2:
            raise_frac = opponent.env_bldr.env_args.bet_sizes_list_as_frac_of_pot[action_int - 2]
            return rlbr_env_wrapper.step_raise_pot_frac(pot_frac=raise_frac)
        else:
            return rlbr_env_wrapper.step(action=action_int)

    else:
        raise ValueError(_type)
