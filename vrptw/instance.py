"""Instance loading (Solomon format) and synthetic pharma instance generator."""
import math
import random

EPS = 1e-9


class Instance:
    """Node 0 is the depot (distributor warehouse); nodes 1..n are chemists."""

    def __init__(self, name, cap, vehicles, x, y, demand, ready, due, service):
        self.name = name
        self.cap = cap
        self.vehicles = vehicles
        self.x, self.y = x, y
        self.demand, self.ready, self.due, self.service = demand, ready, due, service
        self.n = len(x) - 1
        N = self.n + 1
        # Full-precision Euclidean distances (no rounding), travel time == distance
        self.dist = [[math.hypot(x[i] - x[j], y[i] - y[j]) for j in range(N)] for i in range(N)]
        self._neigh = {}

    def neighbours(self, k):
        """k nearest customers of every customer (cached)."""
        if k not in self._neigh:
            nb = [[]]
            for i in range(1, self.n + 1):
                order = sorted((j for j in range(1, self.n + 1) if j != i),
                               key=lambda j: self.dist[i][j])
                nb.append(order[:k])
            self._neigh[k] = nb
        return self._neigh[k]


def load_solomon(path):
    """Parse a Solomon-format text file."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    name = lines[0]
    vehicles = cap = None
    rows = []
    for i, ln in enumerate(lines):
        parts = ln.split()
        if ln.upper().startswith("NUMBER"):
            vehicles, cap = int(lines[i + 1].split()[0]), int(lines[i + 1].split()[1])
        elif len(parts) == 7 and all(p.lstrip("-").replace(".", "", 1).isdigit() for p in parts):
            rows.append([float(p) for p in parts])
    rows.sort(key=lambda r: r[0])
    return Instance(name, cap, vehicles,
                    [r[1] for r in rows], [r[2] for r in rows], [int(r[3]) for r in rows],
                    [r[4] for r in rows], [r[5] for r in rows], [r[6] for r in rows])


def make_pharma(n=60, seed=2026, cap=200, vehicles=15):
    """Synthetic pharma distributor instance (NOT real data).

    Warehouse at the centre of a 60x60 km area. Chemists are clustered around
    town centres; each is open for a delivery slot (morning / midday / afternoon)
    of 60-120 min inside an 8-hour day (480 min, time unit = minute, speed 1 km/min).
    """
    rnd = random.Random(seed)
    centres = [(15, 15), (45, 15), (15, 45), (45, 45), (30, 30)]
    x, y = [30.0], [30.0]
    demand, ready, due, service = [0], [0.0], [480.0], [0.0]
    slots = [(0, 180), (120, 300), (240, 420), (0, 420)]
    for _ in range(n):
        cx, cy = rnd.choice(centres)
        x.append(min(60.0, max(0.0, round(rnd.gauss(cx, 6)))))
        y.append(min(60.0, max(0.0, round(rnd.gauss(cy, 6)))))
        demand.append(rnd.choice([5, 10, 10, 15, 20, 25, 30]))
        lo, hi = rnd.choice(slots)
        width = rnd.choice([60, 90, 120])
        r = rnd.randint(lo, max(lo, hi - width))
        ready.append(float(r))
        due.append(float(r + width))
        service.append(float(rnd.choice([8, 10, 12])))
    return Instance("PHARMA60", cap, vehicles, x, y, demand, ready, due, service)
