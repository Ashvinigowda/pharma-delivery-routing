"""Route-based local search: relocate, swap, 2-opt* (inter-route) and 2-opt (intra-route).

First-improvement, restricted to the K nearest neighbours of each customer.
Objective: sum(route distance) + fixed * (#routes).
"""
from .route import route_dist

GAIN = 1e-7


def local_search(inst, routes, fixed=0.0, k=12, rng=None, max_rounds=30):
    routes = [list(r) for r in routes if r]
    dist = [route_dist(inst, r) for r in routes]
    neigh = inst.neighbours(k)
    custs = list(range(1, inst.n + 1))

    def locate():
        pos = {}
        for ri, r in enumerate(routes):
            for pi, c in enumerate(r):
                pos[c] = (ri, pi)
        return pos

    for _ in range(max_rounds):
        improved = False
        if rng:
            rng.shuffle(custs)
        pos = locate()
        for c in custs:
            ra, ia = pos[c]
            if ra is None or ra >= len(routes) or c not in routes[ra]:
                pos = locate()
                ra, ia = pos[c]
            A = routes[ra]
            for v in neigh[c]:
                rb, ib = pos[v]
                if ra == rb:
                    continue
                B = routes[rb]
                # ---- relocate c next to v (before / after) -------------------------
                A2 = A[:ia] + A[ia + 1:]
                da2 = route_dist(inst, A2) if A2 else 0.0
                done = False
                for ins in (ib, ib + 1):
                    B2 = B[:ins] + [c] + B[ins:]
                    db2 = route_dist(inst, B2)
                    if db2 is None:
                        continue
                    delta = da2 + db2 - dist[ra] - dist[rb] - (0 if A2 else fixed)
                    if delta < -GAIN:
                        routes[ra], routes[rb] = A2, B2
                        dist[ra], dist[rb] = da2, db2
                        done = True
                        break
                if done:
                    improved = True
                    routes_dist_clean(routes, dist)
                    pos = locate()
                    break
                # ---- swap c and v --------------------------------------------------
                A3 = A[:ia] + [v] + A[ia + 1:]
                B3 = B[:ib] + [c] + B[ib + 1:]
                da3 = route_dist(inst, A3)
                if da3 is not None:
                    db3 = route_dist(inst, B3)
                    if db3 is not None and da3 + db3 < dist[ra] + dist[rb] - GAIN:
                        routes[ra], routes[rb] = A3, B3
                        dist[ra], dist[rb] = da3, db3
                        improved = True
                        pos = locate()
                        break
                # ---- 2-opt*: exchange tails after c and after v ----------------------
                A4 = A[:ia + 1] + B[ib + 1:]
                B4 = B[:ib + 1] + A[ia + 1:]
                da4 = route_dist(inst, A4)
                if da4 is not None:
                    db4 = route_dist(inst, B4) if B4 else 0.0
                    if db4 is not None:
                        delta = da4 + db4 - dist[ra] - dist[rb] - (0 if B4 else fixed)
                        if delta < -GAIN:
                            routes[ra], routes[rb] = A4, B4
                            dist[ra], dist[rb] = da4, db4 if B4 else 0.0
                            improved = True
                            routes_dist_clean(routes, dist)
                            pos = locate()
                            break
        # ---- intra-route: 2-opt (segment reversal) and or-opt shift -------------------
        for ri in range(len(routes)):
            r = routes[ri]
            L = len(r)
            if L < 3:
                continue
            best = dist[ri]
            for i in range(L - 1):
                for j in range(i + 1, L):
                    cand = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    dc = route_dist(inst, cand)
                    if dc is not None and dc < best - GAIN:
                        r, best = cand, dc
                        improved = True
            for i in range(L):
                for j in range(L):
                    if i == j:
                        continue
                    cand = r[:i] + r[i + 1:]
                    cand = cand[:j] + [r[i]] + cand[j:]
                    dc = route_dist(inst, cand)
                    if dc is not None and dc < best - GAIN:
                        r, best = cand, dc
                        improved = True
            routes[ri], dist[ri] = r, best
        if not improved:
            break
    return routes


def routes_dist_clean(routes, dist):
    """Remove empty routes in place (keeping dist aligned)."""
    i = 0
    while i < len(routes):
        if not routes[i]:
            del routes[i]
            del dist[i]
        else:
            i += 1
