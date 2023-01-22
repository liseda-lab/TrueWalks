import abc
from typing import Any, List, Set, DefaultDict, Tuple
from collections import defaultdict
from hashlib import md5

import rdflib
from rdflib.namespace import RDF, OWL, RDFS

from kg import KG, Vertex
from sampler import Sampler


class Walker(metaclass=abc.ABCMeta):
    """Base class for the walking strategies."""

    def __init__(
        self,
        depth: int,
        walks_per_graph: float,
        sampler: Sampler,
        wl_iterations: int = 4,
    ):
        self.depth = depth
        self.walks_per_graph = walks_per_graph
        self.sampler = sampler
        self.wl_iterations = wl_iterations

    def extract(self, kg: KG, instances: List[rdflib.URIRef]) -> Set[Tuple[Any, ...]]:
        """Fits the provided sampling strategy and then calls the
        _extract method """
        return self._extract(kg, instances)

    def _create_label(self, kg: KG, vertex: Vertex, n: int):
        """Creates a label."""
        neighbor_names = [
            self._label_map[neighbor][n - 1]
            for neighbor in kg.get_inv_neighbors(vertex)
        ]
        suffix = "-".join(sorted(set(map(str, neighbor_names))))
        return self._label_map[vertex][n - 1] + "-" + suffix

    def _weisfeiler_lehman(self, kg: KG) -> None:
        """Performs Weisfeiler-Lehman relabeling of the vertices."""
        self._label_map: DefaultDict[Any, Any] = defaultdict(dict)
        self._inv_label_map: DefaultDict[Any, Any] = defaultdict(dict)

        for v in kg._vertices:
            self._label_map[v][0] = str(v)
            self._inv_label_map[str(v)][0] = v

        for n in range(1, self.wl_iterations + 1):
            for vertex in kg._vertices:
                s_n = self._create_label(kg, vertex, n)
                self._label_map[vertex][n] = str(md5(s_n.encode()).digest())

        for vertex in kg._vertices:
            for key, val in self._label_map[vertex].items():
                self._inv_label_map[vertex][val] = key

    def _extract(self, kg: KG, instances: List[rdflib.URIRef]) -> Set[Tuple[Any, ...]]:
        """Extracts walks rooted at the provided instances which are then each transformed into a numerical representation. The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size."""
        self._weisfeiler_lehman(kg)
        positive_canonical_walks, negative_canonical_walks = set(), set()
        for instance in instances:

            positive_walks = self.extract_random_walks_dfs(kg, str(instance), "positive", ["http://hasNegativeAnnotation", "https://www.w3.org/2000/01/rdf-schema#superClassOf"])
            for n in range(self.wl_iterations + 1):
                for walk in positive_walks:
                    pos_canonical_walk = []
                    for i, hop in enumerate(walk):  # type: ignore
                        if i == 0 or i % 2 == 1:
                            pos_canonical_walk.append(str(hop))
                        else:
                            pos_canonical_walk.append(self._label_map[hop][n])
                    positive_canonical_walks.add(tuple(pos_canonical_walk))

            negative_walks = self.extract_random_walks_dfs(kg, str(instance), "negative", ["http://hasPositiveAnnotation", str(RDFS.subClassOf)])
            for n in range(self.wl_iterations + 1):
                for walk in negative_walks:
                    neg_canonical_walk = []
                    for i, hop in enumerate(walk):  # type: ignore
                        if i == 0 or i % 2 == 1:
                            neg_canonical_walk.append(str(hop))
                        else:
                            neg_canonical_walk.append(self._label_map[hop][n])
                    negative_canonical_walks.add(tuple(neg_canonical_walk))

        return positive_canonical_walks, negative_canonical_walks

    def extract_random_walks_dfs(self, graph, root, status, forbidden_links):
        """Depth-first search to extract a limited number of walks."""
        self.sampler.initialize()

        walks = []
        while len(walks) < self.walks_per_graph:
            new = (root,)
            d = 1
            while d // 2 < self.depth:
                last = d // 2 == self.depth - 1
                hop = self.sampler.sample_neighbor(graph, new, last, status, forbidden_links)
                if hop is None:
                    break
                new = new + (hop[0], hop[1])
                d = len(new) - 1
            walks.append(new)
        return list(set(walks))


