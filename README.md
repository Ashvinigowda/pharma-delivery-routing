# Pharma Delivery Routing (VRPTW) with an Evolutionary / Memetic Algorithm

Soft Computing AAT, Problem 1 (SDG 11): plan daily routes for a pharmaceutical distributor's fleet, minimising total
distance subject to vehicle capacity and chemists' delivery time windows.

**Technique:** Evolutionary Algorithm (giant-tour chromosome + optimal Split decoder), plus a memetic variant with local search.
**No optimisation libraries:** everything (EA, Split, local search, baselines, checker, Mann-Whitney test) is plain Python standard library.
`matplotlib` is used only by `analyse.py` for figures.

## Layout
```
vrptw/            instance.py  loader + synthetic pharma instance
                  route.py     route evaluation + Split decoder
                  ea.py        plain / memetic EA
                  localsearch.py  relocate, swap, 2-opt*, 2-opt
                  baselines.py Nearest Neighbour, Clarke-Wright savings
                  checker.py   independent feasibility/distance verifier
                  stats.py     mean, std, Mann-Whitney U
data/solomon/     9 Solomon benchmark instances (C101 C102 C201 R101 R102 R201 RC101 RC102 RC201)
tests/            unit tests (incl. brute-force check that Split is optimal)
run_experiments.py   main / params / fleet experiments (resumable, results/raw/)
analyse.py        tables (results/*.md, *.csv) and figures (figures/*.png)
```

## Quick start
```bash
python -m unittest discover -s tests -v          # 10 tests
python run_experiments.py main --runs 30 --time 30 --jobs 4
python run_experiments.py params --runs 10 --time 20 --jobs 4
python run_experiments.py fleet  --runs 10 --time 20 --jobs 4
pip install -r requirements.txt && python analyse.py
```
Solve one instance from Python:
```python
from vrptw.instance import load_solomon
from vrptw.ea import EA
inst = load_solomon("data/solomon/R101.txt")
res = EA(inst, memetic=True, seed=1).run(time_limit=30)
print(res["cost"], len(res["routes"]))
```

## Method
- **Chromosome:** permutation of customers. **Decoding:** Bellman-DP Split gives the optimal feasible routes for that order, so every individual is feasible.
- **Operators:** tournament selection, order crossover, swap / inversion / relocate mutation, adaptive mutation rate, elitism.
- **Memetic:** Lamarckian local search on the best offspring each generation (and on others with probability `ls_prob`).
- **Seeding:** Nearest Neighbour, Clarke-Wright and due-date-sorted tours join the random initial population.
- **Objective:** total distance (optional fixed cost per vehicle). Euclidean distances are not rounded; travel time equals distance.

## Results (5 runs x 20 s per run, 1 CPU core)
| Instance | BKS | NN | CW | Plain EA mean | Memetic best | Memetic mean | Memetic std | Gap% (best) | Gap% (mean) | Veh | p-value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C101 | 828.94 | 1870.69 | 930.12 | 928.77 | 828.94 | 828.94 | 0.00 | +0.00 | +0.00 | 10 | 0.0075 |
| C102 | 828.94 | 1992.46 | 1034.24 | 983.39 | 828.94 | 828.94 | 0.00 | +0.00 | +0.00 | 10 | 0.0109 |
| C201 | 591.56 | 1880.47 | 751.82 | 726.00 | 591.56 | 591.56 | 0.00 | +0.00 | +0.00 | 3 | 0.0071 |
| R101 | 1650.80 | 2623.24 | 2002.40 | 1942.05 | 1663.06 | 1670.25 | 6.57 | +0.74 | +1.18 | 21 | 0.0122 |
| R102 | 1486.12 | 2442.78 | 1736.40 | 1674.35 | 1487.28 | 1502.88 | 9.63 | +0.08 | +1.13 | 19 | 0.0122 |
| R201 | 1252.37 | 1984.96 | 1457.48 | 1451.34 | 1173.45 | 1194.02 | 18.58 | -6.30 | -4.66 | 11 | 0.0122 |
| RC101 | 1696.95 | 2711.30 | 2134.89 | 1995.07 | 1682.72 | 1704.12 | 12.90 | -0.84 | +0.42 | 17 | 0.0122 |
| RC102 | 1554.75 | 2647.10 | 1921.14 | 1798.97 | 1516.05 | 1540.27 | 18.12 | -2.49 | -0.93 | 15 | 0.0122 |
| RC201 | 1406.94 | 2468.25 | 1736.57 | 1703.24 | 1304.36 | 1332.31 | 17.75 | -7.29 | -5.30 | 11 | 0.0122 |
| PHARMA60 | 541.04* | 822.78 | 657.92 | 586.07 | 541.04 | 542.78 | 2.66 | +0.00 | +0.32 | 7 | 0.0119 |

\* PHARMA60 is synthetic; the value shown is the best solution found by our own runs, not a literature value.
BKS minimise vehicles first, then distance, so a distance-only solver can legitimately go below them (negative gap) by using more vehicles.

Memetic vs plain EA (equal wall-clock time): Mann-Whitney p < 0.05 on all 10 instances. With only 5 runs per
variant the smallest attainable two-sided p is about 0.0079 (0.012 with continuity correction), so use `--runs 30` for stronger statistics.
The memetic EA matches the best-known distance exactly on C101, C102 and C201.

**Reading the BKS gaps.** Literature best-known solutions minimise vehicle count first, then distance. This project
minimises distance only, so on R2xx/RC2xx it uses more vehicles and gets shorter distances (negative gaps). That is
not a claim of beating the literature. On R101 the memetic EA is about 1% above the best-known distance. BKS values
were taken from literature as recalled; verify them against the SINTEF table before citing.
PHARMA60 is a synthetic instance generated with a fixed seed (`make_pharma`), not real data.

## Parameter study (R101, 3 runs x 10 s)
Parameter study on R101 (distance over independent runs; differences are small relative to run-to-run noise)

| Configuration | Mean | Std | Best | Runs |
|---|---|---|---|---|
| adaptive=off | 1686.14 | 7.12 | 1678.20 | 3 |
| default | 1686.31 | 14.97 | 1669.02 | 3 |
| ls_prob=0.0 | 1680.67 | 15.30 | 1664.79 | 3 |
| ls_prob=0.1 | 1686.31 | 14.97 | 1669.02 | 3 |
| ls_prob=0.3 | 1686.52 | 14.10 | 1676.52 | 3 |
| ls_prob=0.6 | 1681.17 | 3.37 | 1677.32 | 3 |
| mut=0.05 | 1685.07 | 8.25 | 1677.06 | 3 |
| mut=0.2 | 1686.31 | 14.97 | 1669.02 | 3 |
| mut=0.5 | 1685.59 | 13.03 | 1674.61 | 3 |
| mut=0.8 | 1688.13 | 10.28 | 1676.44 | 3 |
| pop=10 | 1687.81 | 11.89 | 1678.21 | 3 |
| pop=20 | 1684.04 | 6.31 | 1678.79 | 3 |
| pop=30 | 1686.31 | 14.97 | 1669.02 | 3 |
| pop=50 | 1679.79 | 5.03 | 1674.41 | 3 |
| seeding=off | 1676.16 | 16.79 | 1659.58 | 3 |

Differences between settings are within run-to-run noise at this sample size; no setting is shown to be clearly better.
Plain `pop`, `mutation_rate`, `ls_prob` and adaptivity have small effects here. Rerun with more runs before drawing conclusions.

## Fleet-size study (R101, 3 runs x 10 s)
| Fixed cost per vehicle | Mean distance | Mean vehicles | Min vehicles |
|---|---|---|---|
| 0 | 1686.31 | 22.7 | 22 |
| 25 | 1671.38 | 21.0 | 20 |
| 50 | 1682.04 | 22.0 | 22 |
| 100 | 1689.39 | 21.3 | 21 |
| 200 | 1673.47 | 21.3 | 21 |
| 500 | 1702.85 | 22.0 | 22 |

A fixed cost per vehicle did **not** produce a clear reduction in vehicles within the 10 s budget (it stays at 20-22,
close to the pure-distance solution, far from the best-known 19). Treat this as inconclusive, not as a trade-off curve.

## Limitations
- Short per-run budgets and few runs (limited by a single CPU core); the scripts scale to 30+ runs.
- Distance-only objective; vehicle count is not minimised first. The instance fleet limit (25) is not a hard constraint in the solver (all EA solutions stay below it; the NN baseline can exceed it).
- Only 100-customer instances were tested.

## Figures
`figures/convergence.png`, `boxplots.png`, `routes_*.png`, `fleet_tradeoff.png`.

## Data and references
Solomon, M.M. (1987), "Algorithms for the vehicle routing and scheduling problems with time window constraints", Operations Research 35(2). Instance files obtained from the iRB-Lab/py-ga-VRPTW repository.
