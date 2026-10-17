"""Constructive baselines coded from scratch: Nearest Neighbour and Clarke-Wright savings."""
from .instance import EPS
from .route import route_dist, total_cost


def nearest_neighbour(inst):
    """Sequential NN: extend the current route with the closest feasible customer."""
    d = inst.dist
    unvisited = set(range(1, inst.n + 1))
    routes = []
    while unvisited:
        route = []
        while True:
            last = route[-1] if route else 0
            best, bd = None, float("inf")
            for c in unvisited:
                if d[last][c] < bd and route_dist(inst, route + [c]) is not None:
                    best, bd = c, d[last][c]
            if best is None:
                break
            route.append(best)
            unvisited.remove(best)
        if not route:  # cannot happen for valid instances (single customer is feasible)
            raise ValueError("infeasible customer")
        routes.append(route)
    return routes


def savings(inst):
    """Clarke-Wright parallel savings with time-window feasibility checks."""
    d = inst.dist
    n = inst.n
    routes = {i: [i] for i in range(1, n + 1)}
    where = {i: i for i in range(1, n + 1)}  # customer -> route id
    sav = sorted(((d[0][i] + d[0][j] - d[i][j], i, j)
                  for i in range(1, n + 1) for j in range(1, n + 1) if i != j),
                 reverse=True)
    for s, i, j in sav:
        if s <= 0:
            break
        ri, rj = where[i], where[j]
        if ri == rj:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] != i or b[0] != j:  # i must end its route, j must start its route
            continue
        merged = a + b
        if route_dist(inst, merged) is None:
            continue
        routes[ri] = merged
        del routes[rj]
        for c in b:
            where[c] = ri
    return list(routes.values())


def tour_from_routes(routes):
    return [c for r in routes for c in r]
