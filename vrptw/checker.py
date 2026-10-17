"""Independent solution verifier (does not reuse route.py logic)."""
import math


def verify(inst, routes, tol=1e-6):
    """Return (ok, total_distance, message). Recomputes everything from coordinates."""
    seen = []
    total = 0.0
    for ri, r in enumerate(routes):
        if not r:
            return False, 0.0, f"route {ri} empty"
        load, t = 0, 0.0
        px, py = inst.x[0], inst.y[0]
        for c in r:
            seen.append(c)
            dx = math.hypot(inst.x[c] - px, inst.y[c] - py)
            total += dx
            t += dx
            t = max(t, inst.ready[c])
            if t > inst.due[c] + tol:
                return False, total, f"route {ri}: customer {c} late ({t:.2f}>{inst.due[c]})"
            t += inst.service[c]
            load += inst.demand[c]
            px, py = inst.x[c], inst.y[c]
        if load > inst.cap:
            return False, total, f"route {ri}: capacity {load}>{inst.cap}"
        back = math.hypot(px - inst.x[0], py - inst.y[0])
        total += back
        if t + back > inst.due[0] + tol:
            return False, total, f"route {ri}: returns late to depot"
    if sorted(seen) != list(range(1, inst.n + 1)):
        return False, total, "customers missing or duplicated"
    return True, total, "ok"
