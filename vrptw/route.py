"""Route evaluation and the Split decoder."""
from .instance import EPS

INF = float("inf")


def route_dist(inst, route):
    """Distance of depot->route->depot, or None if capacity/time windows violated."""
    d, dem, rdy, due, srv = inst.dist, inst.demand, inst.ready, inst.due, inst.service
    t = 0.0
    load = 0
    prev = 0
    dist = 0.0
    for c in route:
        load += dem[c]
        if load > inst.cap:
            return None
        step = d[prev][c]
        dist += step
        t += step
        if t < rdy[c]:
            t = rdy[c]
        elif t > due[c] + EPS:
            return None
        t += srv[c]
        prev = c
    back = d[prev][0]
    if t + back > due[0] + EPS:
        return None
    return dist + back


def split(inst, tour, fixed=0.0):
    """Optimal Split (Bellman DP) of a giant tour into feasible routes.

    Minimises sum(route distance + fixed) for the given customer order.
    Because arrival times only grow along a route, extension stops at the first
    capacity / time-window violation. Returns (cost, routes).
    """
    n = len(tour)
    d, dem, rdy, due, srv = inst.dist, inst.demand, inst.ready, inst.due, inst.service
    cap, depot_due = inst.cap, inst.due[0]
    V = [INF] * (n + 1)
    pred = [0] * (n + 1)
    V[0] = 0.0
    for i in range(n):
        base = V[i]
        if base == INF:
            continue
        load = 0
        t = 0.0
        prev = 0
        dist = 0.0
        for j in range(i, n):
            c = tour[j]
            load += dem[c]
            if load > cap:
                break
            step = d[prev][c]
            dist += step
            t += step
            if t < rdy[c]:
                t = rdy[c]
            elif t > due[c] + EPS:
                break
            t += srv[c]
            back = d[c][0]
            prev = c
            if t + back > depot_due + EPS:
                continue
            cost = base + dist + back + fixed
            if cost < V[j + 1]:
                V[j + 1] = cost
                pred[j + 1] = i
    if V[n] == INF:
        return INF, []
    routes = []
    j = n
    while j > 0:
        i = pred[j]
        routes.append(tour[i:j])
        j = i
    routes.reverse()
    return V[n], routes


def total_cost(inst, routes, fixed=0.0):
    return sum(route_dist(inst, r) + fixed for r in routes)
