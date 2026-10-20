"""Regenerate every revision result once all five seeds are complete.

Runs the rev_lib regression check (the submitted seed-42 numbers must be reproduced exactly) and then
each analysis module, writing one log per module to experiments/logs/. Refuses to run on incomplete
seeds unless --allow-partial is given, so that no paper number can come from a partial run.
"""
import os, sys, subprocess, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_lib as L

MODULES = ["rev_main", "rev_variance", "rev_calibration", "rev_robustness", "rev_transfer", "rev_annotation",
           "rev_ablation", "rev_cost"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--only", nargs="+", default=MODULES)
    a = ap.parse_args()
    seeds = {ds: L.complete_seeds(ds) for ds in L.IN_DOMAIN}
    print("complete seeds:", seeds, flush=True)
    if any(len(v) < len(L.SEEDS) for v in seeds.values()) and not a.allow_partial:
        sys.exit("not all seeds are complete; rerun when training has finished (or pass --allow-partial)")
    logs = os.path.join(L.ROOT, "experiments", "logs")
    py = [sys.executable, "-u"]
    steps = [("rev_lib_regression", py + [os.path.join(L.ROOT, "experiments", "rev_lib.py")])]
    steps += [(m, py + [os.path.join(L.ROOT, "experiments", f"{m}.py")]) for m in a.only]
    for name, cmd in steps:
        t0 = time.time()
        with open(os.path.join(logs, f"{name}.log"), "w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, cwd=L.ROOT, stdout=fh, stderr=subprocess.STDOUT)
        print(f"{name:20s} exit={r.returncode} {time.time() - t0:6.0f}s", flush=True)
        if r.returncode != 0:
            sys.exit(f"{name} failed; see experiments/logs/{name}.log")
        if name == "rev_lib_regression":
            tail = open(os.path.join(logs, f"{name}.log"), encoding="utf-8").read().strip().splitlines()[-1]
            if "PASSED" not in tail:
                sys.exit(f"regression check did not pass: {tail}")
    print("all revision analyses regenerated", flush=True)


if __name__ == "__main__":
    main()
