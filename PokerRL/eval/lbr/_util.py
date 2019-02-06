# Copyright (c) 2019 Eric Steinberger


from PokerRL.rl.rl_util import get_builder_from_str, get_env_cls_from_str


def get_env_builder_lbr(t_prof):
    env_bldr_cls = get_builder_from_str(t_prof.env_builder_cls_str)
    return env_bldr_cls(env_cls=get_env_cls_from_str(t_prof.game_cls_str),
                        env_args=t_prof.module_args["lbr"].get_lbr_env_args(agents_env_args=t_prof.module_args["env"]))
