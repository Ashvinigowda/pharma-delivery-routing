"""Small statistics helpers implemented from scratch."""
import math


def mean(a):
    return sum(a) / len(a)


def std(a):
    if len(a) < 2:
        return 0.0
    m = mean(a)
    return math.sqrt(sum((x - m) ** 2 for x in a) / (len(a) - 1))


def mann_whitney_p(a, b):
    """Two-sided Mann-Whitney U p-value (normal approximation, tie-corrected)."""
    n1, n2 = len(a), len(b)
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = [0.0] * len(allv), 0
    tie_term = 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    N = n1 + n2
    mu = n1 * n2 / 2
    var = n1 * n2 / 12 * ((N + 1) - tie_term / (N * (N - 1)))
    if var <= 0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    return max(0.0, min(1.0, math.erfc(max(z, 0) / math.sqrt(2))))
