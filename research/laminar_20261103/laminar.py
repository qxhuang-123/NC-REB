"""Exact fixed-plan service certificate for laminar supply neighborhoods."""
from collections import Counter


class LaminarTree:
    def __init__(self, supports, m):
        if m < 1:
            raise ValueError('at least one demand required')
        self.supports = [tuple(sorted(set(s))) for s in supports]
        if any(i < 0 or i >= m for s in self.supports for i in s):
            raise ValueError('invalid demand index')
        sets = set(self.supports) - {()}
        sets.update((i,) for i in range(m))
        sets.add(tuple(range(m)))
        # Total membership work is linear after sorting by cardinality.
        self.nodes = [(i,) for i in range(m)]
        self.children = [[] for _ in range(m)]
        owner = list(range(m))
        self.laminar = True
        for s in sorted((s for s in sets if len(s) > 1), key=len):
            counts = Counter(owner[i] for i in s)
            bad = next((v for v, count in counts.items() if count != len(self.nodes[v])), None)
            if bad is not None:
                self.laminar = False
                self.crossing_witness = (self.nodes[bad], s)
                return
            v = len(self.nodes)
            self.nodes.append(s)
            self.children.append(list(counts))
            for i in s:
                owner[i] = v
        self.root = owner[0]
        index = {s: v for v, s in enumerate(self.nodes)}
        self.assignment = [index.get(s, -1) for s in self.supports]
        self.m = m
        assert len(self.nodes) <= 2 * m - 1

    def capacities(self, cap):
        assert self.laminar and len(cap) == len(self.assignment)
        c = [0] * len(self.nodes)
        for v, value in zip(self.assignment, cap):
            if v >= 0:
                c[v] += int(value)
        return c

    def evaluate(self, c, demand):
        """Return exact deficit and an original demand-side Hall witness."""
        r = [0] * len(self.nodes)
        for v in range(self.m):
            r[v] = max(0, int(demand[v]) - c[v])
        for v in range(self.m, len(self.nodes)):
            r[v] = max(0, sum(r[u] for u in self.children[v]) - c[v])
        loss = r[self.root]
        witness = []
        if loss:
            stack = [self.root]
            while stack:
                v = stack.pop()
                if v < self.m:
                    witness.append(v)
                else:
                    stack.extend(u for u in self.children[v] if r[u] > 0)
        return loss, witness
