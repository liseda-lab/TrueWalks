import abc
from typing import Any, Set

import numpy as np

from kg import KG


class Sampler(metaclass=abc.ABCMeta):
    """Base class for the sampling strategies."""
    def __init__(self):
        pass

    def initialize(self) -> None:
        """Tags vertices that appear at the max depth or of which all their children are tagged."""
        self.positive_visited: Set[Any] = set()  #
        self.negative_visited: Set[Any] = set()

    def sample_neighbor(self, kg: KG, walk, last, status, forbidden_links):
        """ Fetch the neighbors of a given node to extend the walk"""
        if status == "postive":
            not_tag_neighbors = [
                x
                for x in kg.get_hops(walk[-1], forbidden_links)
                if (x, len(walk)) not in self.positive_visited
            ]
            if len(not_tag_neighbors) == 0:
                if len(walk) > 2:
                    self.positive_visited.add(((walk[-2], walk[-1]), len(walk) - 2))
                return None

        else:
            not_tag_neighbors = [
                x
                for x in kg.get_hops(walk[-1], forbidden_links)
                if (x, len(walk)) not in self.negative_visited
            ]
            if len(not_tag_neighbors) == 0:
                if len(walk) > 2:
                    self.negative_visited.add(((walk[-2], walk[-1]), len(walk) - 2))
                return None

        weights = [1 for hop in not_tag_neighbors]
        weights = [x / sum(weights) for x in weights]

        # Sample a random neighbor and add them to visited if needed.
        rand_ix = np.random.choice(range(len(not_tag_neighbors)), p=weights)

        if last:
            if status == "postive":
                self.positive_visited.add((not_tag_neighbors[rand_ix], len(walk)))
            else:
                self.negative_visited.add((not_tag_neighbors[rand_ix], len(walk)))
        return not_tag_neighbors[rand_ix]


