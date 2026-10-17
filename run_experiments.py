#!/usr/bin/env python3
"""Run all experiments. Results are cached per run in results/raw/ (resumable).

  python run_experiments.py main   --runs 10 --time 20      # EA variants + baselines
  python run_experiments.py params --runs 3  --time 10      # parameter study
  python run_experiments.py fleet  --runs 3  --time 10      # vehicle-count trade-off
  python run_experiments.py all
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

from vrptw.baselines import nearest_neighbour, savings
from vrptw.checker import verify
from vrptw.ea import EA
from vrptw.instance import load_solomon, make_pharma
from vrptw.route import total_cost

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "results", "raw")
INSTANCES = ["C101", "C102", "C201", "R101", "R102", "R201", "RC101", "RC102", "RC201", "PHARMA60"]

# Best-known Solomon solutions (minimise #vehicles first, then distance) from the literature.
# Verify against the SINTEF table before quoting: https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/
BKS = {"C101": (828.94, 10), "C102": (828.94, 10), "C201": (591.56, 3),
       "R101": (1650.80, 19), "R102": (1486.12, 17), "R201": (1252.37, 4),
       "RC101": (1696.95, 14), "RC102": (1554.75, 12), "RC201": (1406.94, 4)}


def get_instance(name):
    if name == "PHARMA60":
        return make_pharma()
    return load_solomon(os.path.join(ROOT, "data", "solomon", name + ".txt"))


def run_one(job):
    """job = dict(tag, instance, params, seed, time). Cached on disk."""
    path = os.path.join(RAW, job["tag"] + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    inst = get_instance(job["instance"])
    p = dict(job["params"])
    ea = EA(inst, seed=job["seed"], **p)
    res = ea.run(time_limit=job["time"])
    ok, dist, msg = verify(inst, res["routes"])
    out = {"tag": job["tag"], "instance": job["instance"], "params": p, "seed": job["seed"],
           "cost": res["cost"], "distance": dist, "vehicles": len(res["routes"]), "valid": ok,
           "generations": res["generations"], "time": res["time"], "history": res["history"],
           "routes": res["routes"]}
    os.makedirs(RAW, exist_ok=True)
    json.dump(out, open(path, "w"))
    return out


def run_jobs(jobs, nproc):
    os.makedirs(RAW, exist_ok=True)
    if nproc > 1:
        with Pool(nproc) as pool:
            return pool.map(run_one, jobs, chunksize=1)
    out = []
    for i, j in enumerate(jobs):
        t = time.time()
        out.append(run_one(j))
        print(f"[{i + 1}/{len(jobs)}] {j['tag']}  cost={out[-1]['cost']:.2f}  ({time.time() - t:.1f}s)", flush=True)
    return out


def baselines():
    res = {}
    for name in INSTANCES:
        inst = get_instance(name)
        row = {}
        for label, fn in (("NN", nearest_neighbour), ("CW", savings)):
            t = time.time()
            r = fn(inst)
            ok, d, _ = verify(inst, r)
            row[label] = {"distance": d, "vehicles": len(r), "valid": ok, "time": time.time() - t}
        res[name] = row
    json.dump(res, open(os.path.join(ROOT, "results", "baselines.json"), "w"), indent=1)
    return res


def cmd_main(a):
    baselines()
    variants = {"memetic": dict(memetic=True), "plain": dict(memetic=False)}
    jobs = [dict(tag=f"main_{n}_{v}_{s}", instance=n, params=p, seed=s, time=a.time)
            for n in a.instances for v, p in variants.items() for s in range(a.runs)]
    run_jobs(jobs, a.jobs)


def cmd_params(a):
    inst = a.param_instance
    configs = {}
    for v in (10, 20, 30, 50):
        configs[f"pop={v}"] = dict(pop_size=v)
    for v in (0.05, 0.2, 0.5, 0.8):
        configs[f"mut={v}"] = dict(mutation_rate=v)
    for v in (0.0, 0.1, 0.3, 0.6):
        configs[f"ls_prob={v}"] = dict(ls_prob=v)
    configs["adaptive=off"] = dict(adaptive=False)
    configs["seeding=off"] = dict(seed_heuristics=False)
    configs["default"] = dict()
    jobs = [dict(tag=f"param_{inst}_{c}_{s}", instance=inst, params=p, seed=s, time=a.time)
            for c, p in configs.items() for s in range(a.runs)]
    run_jobs(jobs, a.jobs)


def cmd_fleet(a):
    inst = a.param_instance
    jobs = [dict(tag=f"fleet_{inst}_F{f}_{s}", instance=inst, params=dict(fixed=float(f)), seed=s, time=a.time)
            for f in (0, 25, 50, 100, 200, 500) for s in range(a.runs)]
    run_jobs(jobs, a.jobs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "params", "fleet", "all"])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--time", type=float, default=20, help="seconds per run")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes")
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--param-instance", default="R101")
    a = ap.parse_args()
    if a.cmd in ("main", "all"):
        cmd_main(a)
    if a.cmd in ("params", "all"):
        cmd_params(a)
    if a.cmd in ("fleet", "all"):
        cmd_fleet(a)
