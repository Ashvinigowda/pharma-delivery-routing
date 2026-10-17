import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrptw.baselines import nearest_neighbour, savings, tour_from_routes
from vrptw.checker import verify
from vrptw.ea import EA, order_crossover, mutate
from vrptw.instance import load_solomon, make_pharma
from vrptw.localsearch import local_search
from vrptw.route import route_dist, split, total_cost
from vrptw.stats import mann_whitney_p
import random

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "solomon")


def small(n=8):
    """First n customers of the synthetic pharma instance, tight capacity."""
    full = make_pharma(n=n, seed=7, cap=60)
    return full


def brute_force_split(inst, tour):
    """Try every way of cutting the tour into consecutive routes."""
    n = len(tour)
    best = float("inf")
    for mask in range(1 << (n - 1)):
        routes, cur = [], [tour[0]]
        for i in range(1, n):
            if mask >> (i - 1) & 1:
                routes.append(cur)
                cur = [tour[i]]
            else:
                cur.append(tour[i])
        routes.append(cur)
        ds = [route_dist(inst, r) for r in routes]
        if None not in ds:
            best = min(best, sum(ds))
    return best


class TestCore(unittest.TestCase):
    def test_loader(self):
        inst = load_solomon(os.path.join(DATA, "C101.txt"))
        self.assertEqual((inst.n, inst.cap, inst.vehicles), (100, 200, 25))
        self.assertEqual(inst.demand[0], 0)

    def test_split_optimal_vs_bruteforce(self):
        inst = small(8)
        rng = random.Random(3)
        for _ in range(15):
            tour = list(range(1, inst.n + 1))
            rng.shuffle(tour)
            c, routes = split(inst, tour)
            bf = brute_force_split(inst, tour)
            self.assertAlmostEqual(c, bf, places=6)
            self.assertTrue(verify(inst, routes)[0])

    def test_route_dist_detects_violations(self):
        inst = small(8)
        self.assertIsNone(route_dist(inst, list(range(1, 9))))  # capacity 60 < total demand

    def test_checker_catches_errors(self):
        inst = small(8)
        self.assertFalse(verify(inst, [[1, 2, 3]])[0])  # missing customers
        self.assertFalse(verify(inst, [[c] for c in range(1, 9)] + [[1]])[0])  # duplicate

    def test_baselines_feasible(self):
        for name in ("C101", "R101", "RC101"):
            inst = load_solomon(os.path.join(DATA, name + ".txt"))
            for fn in (nearest_neighbour, savings):
                ok, d, msg = verify(inst, fn(inst))
                self.assertTrue(ok, msg)

    def test_local_search_never_worse_and_feasible(self):
        inst = load_solomon(os.path.join(DATA, "R101.txt"))
        base = savings(inst)
        better = local_search(inst, base, rng=random.Random(1))
        self.assertTrue(verify(inst, better)[0])
        self.assertLessEqual(total_cost(inst, better), total_cost(inst, base) + 1e-9)

    def test_operators_keep_permutation(self):
        rng = random.Random(0)
        a, b = list(range(1, 21)), list(range(20, 0, -1))
        for _ in range(100):
            self.assertEqual(sorted(order_crossover(a, b, rng)), a)
            self.assertEqual(sorted(mutate(a, rng)), a)

    def test_ea_valid_and_reproducible(self):
        inst = make_pharma(n=25, seed=1)
        r1 = EA(inst, pop_size=10, seed=5).run(max_gen=15)
        r2 = EA(inst, pop_size=10, seed=5).run(max_gen=15)
        self.assertEqual(r1["cost"], r2["cost"])
        self.assertTrue(verify(inst, r1["routes"])[0])

    def test_ea_reaches_known_optimum_c101(self):
        inst = load_solomon(os.path.join(DATA, "C101.txt"))
        r = EA(inst, seed=1).run(time_limit=25)
        self.assertTrue(verify(inst, r["routes"])[0])
        self.assertLess(r["cost"], 840.0)  # best known 828.94

    def test_mann_whitney(self):
        self.assertLess(mann_whitney_p([1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]), 0.01)
        self.assertGreater(mann_whitney_p([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 0.9)


if __name__ == "__main__":
    unittest.main()
