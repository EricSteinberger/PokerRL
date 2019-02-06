# Copyright (c) 2019 Eric Steinberger


import unittest
from unittest import TestCase

from PokerRL.game._.rl_env.base._Deck import DeckOfCards


class Test(TestCase):

    def test_build(self):
        d = DeckOfCards()
        assert d.deck_remaining.shape == (d.n_ranks * d.n_suits, 2)

    def test_draw(self):
        for n in range(20):
            deck = DeckOfCards()
            cards = deck.draw(n)
            assert deck.deck_remaining.shape == (deck.n_ranks * deck.n_suits - n, 2)

            for card in cards:
                for _card in deck.deck_remaining:
                    assert not (card[0] == _card[0] and card[1] == _card[1])


if __name__ == '__main__':
    unittest.main()
