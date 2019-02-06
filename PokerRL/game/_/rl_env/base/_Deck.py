# Copyright (c) 2019 Eric Steinberger


import numpy as np


class DeckOfCards:
    """
    Cards are stored in 2D form [rank, suit]
    """

    def __init__(self, num_suits=4, num_ranks=13):
        self.n_suits = num_suits
        self.n_ranks = num_ranks

        self.deck_remaining = self._build_deck()
        self._starting_deck = self._build_deck()
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.deck_remaining)

    def draw(self, num_cards):
        """ draws from top. is completely deterministic from a given starting state """
        cards = self.deck_remaining[:num_cards]
        self.deck_remaining = self.deck_remaining[num_cards:]
        return cards

    def reset(self):
        self.deck_remaining = np.copy(self._starting_deck)
        self.shuffle()

    def state_dict(self):
        return {
            "deck_remaining": np.copy(self.deck_remaining)
        }

    def load_state_dict(self, state_dict):
        self.deck_remaining = np.copy(state_dict["deck_remaining"])

    def _build_deck(self):
        deck = np.empty(shape=(self.n_ranks * self.n_suits, 2), dtype=np.int8)

        for r in range(self.n_ranks):
            i = self.n_suits * r
            deck[i:i + self.n_suits, 0] = r
            for s in range(self.n_suits):
                deck[i + s, 1] = s

        return deck

    def remove_cards(self, cards_2d):
        idxs_to_del = []
        for to_rem in cards_2d:
            for i, c in enumerate(self.deck_remaining):
                if np.array_equal(c, to_rem):
                    idxs_to_del.append(i)
                    break

        self.deck_remaining = np.delete(self.deck_remaining, idxs_to_del, axis=0)
