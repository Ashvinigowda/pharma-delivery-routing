"""Evolutionary algorithm (plain) and Memetic EA (EA + local search) for VRPTW.

Chromosome  : giant tour = permutation of customers 1..n
Fitness     : cost of the optimal Split of the tour (always feasible)
Selection   : tournament           Crossover: order crossover (OX)
Mutation    : swap / inversion / relocate, with adaptive rate
Replacement : elitist generational
Memetic     : Lamarckian local search on the best offspring (+ random others)
"""
import random
import time

from .baselines import nearest_neighbour, savings, tour_from_routes
from .localsearch import local_search
from .route import split


def order_crossover(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    used = set(p1[a:b + 1])
    fill = [g for g in p2[b + 1:] + p2[:b + 1] if g not in used]
    idx = list(range(b + 1, n)) + list(range(0, a))
    for i, g in zip(idx, fill):
        child[i] = g
    return child


def mutate(t, rng):
    t = t[:]
    n = len(t)
    op = rng.random()
    if op < 1 / 3:  # swap
        i, j = rng.sample(range(n), 2)
        t[i], t[j] = t[j], t[i]
    elif op < 2 / 3:  # inversion
        i, j = sorted(rng.sample(range(n), 2))
        t[i:j + 1] = t[i:j + 1][::-1]
    else:  # relocate
        i = rng.randrange(n)
        g = t.pop(i)
        t.insert(rng.randrange(n), g)
    return t


class EA:
    def __init__(self, inst, pop_size=30, crossover_rate=0.9, mutation_rate=0.2,
                 tournament=3, elite=2, memetic=True, ls_prob=0.1, adaptive=True,
                 seed_heuristics=True, fixed=0.0, ls_k=12, seed=0):
        self.inst, self.pop_size = inst, pop_size
        self.pc, self.pm0 = crossover_rate, mutation_rate
        self.tournament, self.elite = tournament, elite
        self.memetic, self.ls_prob = memetic, ls_prob
        self.adaptive, self.seed_h = adaptive, seed_heuristics
        self.fixed, self.ls_k = fixed, ls_k
        self.rng = random.Random(seed)
        self._cache = {}

    # -- evaluation ---------------------------------------------------------------
    def evaluate(self, tour):
        key = tuple(tour)
        hit = self._cache.get(key)
        if hit is None:
            hit = split(self.inst, tour, self.fixed)
            if len(self._cache) > 200000:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def improve(self, tour):
        """Lamarckian local search: decode, improve routes, re-encode."""
        cost, routes = self.evaluate(tour)
        routes = local_search(self.inst, routes, self.fixed, self.ls_k, self.rng)
        new = tour_from_routes(routes)
        c2, _ = self.evaluate(new)
        return (new, c2) if c2 < cost else (tour, cost)

    # -- initial population ---------------------------------------------------------
    def init_population(self):
        inst, rng = self.inst, self.rng
        pop = []
        if self.seed_h:
            pop.append(tour_from_routes(nearest_neighbour(inst)))
            pop.append(tour_from_routes(savings(inst)))
            pop.append(sorted(range(1, inst.n + 1), key=lambda c: (inst.due[c], inst.ready[c])))
        base = list(range(1, inst.n + 1))
        while len(pop) < self.pop_size:
            t = base[:]
            rng.shuffle(t)
            pop.append(t)
        return pop[:self.pop_size]

    def select(self, scored):
        return min(self.rng.sample(scored, self.tournament), key=lambda s: s[0])[1]

    # -- main loop ----------------------------------------------------------------
    def run(self, max_gen=10**9, time_limit=None, stall_limit=None):
        t0 = time.time()
        rng = self.rng
        pop = self.init_population()
        scored = [(self.evaluate(t)[0], t) for t in pop]
        if self.memetic:  # polish the seeds so the search starts from a strong base
            scored = [(c, t) if i >= 3 else self._ls_pair(t) for i, (c, t) in enumerate(scored)]
        scored.sort(key=lambda s: s[0])
        best_cost, best_tour = scored[0]
        history = [(0, time.time() - t0, best_cost)]
        pm, stall, gen = self.pm0, 0, 0
        while gen < max_gen:
            gen += 1
            if time_limit and time.time() - t0 >= time_limit:
                break
            if stall_limit and stall >= stall_limit:
                break
            offspring = []
            while len(offspring) < self.pop_size - self.elite:
                p1, p2 = self.select(scored), self.select(scored)
                child = order_crossover(p1, p2, rng) if rng.random() < self.pc else p1[:]
                if rng.random() < pm:
                    child = mutate(child, rng)
                offspring.append((self.evaluate(child)[0], child))
            if self.memetic:
                offspring.sort(key=lambda s: s[0])
                offspring[0] = self._ls_pair(offspring[0][1])
                for i in range(1, len(offspring)):
                    if rng.random() < self.ls_prob:
                        offspring[i] = self._ls_pair(offspring[i][1])
            scored = sorted(scored[:self.elite] + offspring, key=lambda s: s[0])
            if scored[0][0] < best_cost - 1e-9:
                best_cost, best_tour = scored[0]
                stall, pm = 0, self.pm0
                history.append((gen, time.time() - t0, best_cost))
            else:
                stall += 1
                if self.adaptive and stall % 10 == 0:  # raise exploration when stuck
                    pm = min(0.9, pm * 1.5)
        cost, routes = self.evaluate(best_tour)
        history.append((gen, time.time() - t0, best_cost))
        return {"cost": cost, "routes": routes, "tour": best_tour,
                "history": history, "generations": gen, "time": time.time() - t0}

    def _ls_pair(self, tour):
        t, c = self.improve(tour)
        return (c, t)
