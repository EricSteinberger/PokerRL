# Copyright (c) 2019 Eric Steinberger


import unittest
from unittest import TestCase

from PokerRL.cfr.CFRPlus import CFRPlus
from PokerRL.cfr.LinearCFR import LinearCFR
from PokerRL.cfr.VanillaCFR import VanillaCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLLeduc
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase


class TestCFR(TestCase):

    def test_run_CFR(self):
        n_iterations = 3
        name = "TESTING_CFR"

        chief = ChiefBase(t_prof=None)
        cfr = VanillaCFR(name=name,
                         game_cls=DiscretizedNLLeduc,
                         agent_bet_set=bet_sets.POT_ONLY,
                         chief_handle=chief)

        cfr.reset()

        for iter_id in range(n_iterations):
            cfr.iteration()

    def test_run_CFRplus(self):
        n_iterations = 3
        name = "TESTING_CFRplus"

        chief = ChiefBase(t_prof=None)
        cfr = CFRPlus(name=name,
                      game_cls=DiscretizedNLLeduc,
                      delay=0,
                      agent_bet_set=bet_sets.POT_ONLY,
                      chief_handle=chief)

        cfr.reset()

        for iter_id in range(n_iterations):
            cfr.iteration()

    def test_run_linearCFR(self):

        n_iterations = 3
        name = "TESTING_LinearCFR"

        chief = ChiefBase(t_prof=None)
        cfr = LinearCFR(name=name,
                        game_cls=DiscretizedNLLeduc,
                        agent_bet_set=bet_sets.POT_ONLY,
                        chief_handle=chief)

        cfr.reset()

        for iter_id in range(n_iterations):
            cfr.iteration()


if __name__ == '__main__':
    unittest.main()
