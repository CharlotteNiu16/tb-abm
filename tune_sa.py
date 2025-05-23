#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid / Bayesian search for South-Africa β & contact_scale
author: you
usage:
    python tune_sa.py --model_file 333.py
"""

import argparse, importlib.util, pathlib, sys, time
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

################################################################################
# 1  加载 run_sim()
################################################################################
def load_run_sim(model_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("tb_abm", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # type: ignore
    return module.run_sim

################################################################################
# 2  网格搜索
################################################################################
def grid_search(run_sim, data_dir, pop_real):
    POP_SIM   = 40_000
    BURNIN    = 8
    MONTHS    = 6
    START     = "2022-01-01"
    SEED      = 42
    CASESFILE = data_dir / "observed_ZAF_2023.csv"

    beta_space   = np.linspace(0.75, 0.90, 4)        # 0.75, 0.80, 0.85, 0.90
    cscale_space = np.linspace(1.00, 1.15, 4)        # 1.00, 1.05, 1.10, 1.15
    results = []

    for b, cs in tqdm(list(product(beta_space, cscale_space)), desc="Grid"):
        out = run_sim("south_africa", data_dir, POP_SIM,
                      MONTHS, SEED, -1, pop_real, START, CASESFILE,
                      burnin_months=BURNIN, show_plot=False,
                      beta_mul=b, contact_scale_mul=cs)
        results.append({"beta_mul": b, "cscale_mul": cs, "MAPE": out["MAPE"]})

    df = pd.DataFrame(results).sort_values("MAPE")
    best = df.iloc[0]
    print("\n=== Grid search TOP-3 ===")
    print(df.head(3).to_string(index=False, formatters={"MAPE": "{:.2f}%".format}))
    return best["beta_mul"], best["cscale_mul"]

################################################################################
# 3  可选：简单贝叶斯优化（需要 scikit-optimize；否则跳过）
################################################################################
def bayes_opt(run_sim, data_dir, pop_real, init_points=8, n_calls=20):
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        print("scikit-optimize 未安装，跳过贝叶斯优化"); return None

    POP_SIM   = 40_000
    BURNIN    = 8
    MONTHS    = 6
    START     = "2022-01-01"
    SEED      = 42
    CASESFILE = data_dir / "observed_ZAF_2023.csv"

    def objective(x):
        b, cs = x
        mape = run_sim("south_africa", data_dir, POP_SIM,
                       MONTHS, SEED, -1, pop_real, START, CASESFILE,
                       burnin_months=BURNIN, show_plot=False,
                       beta_mul=b, contact_scale_mul=cs)["MAPE"]
        return mape

    space = [Real(0.70, 0.90, name="beta_mul"),
             Real(0.95, 1.20, name="cscale_mul")]

    res = gp_minimize(objective, space,
                      n_initial_points=init_points,
                      n_calls=n_calls, random_state=SEED)
    print("\n=== Bayesian best ===")
    print(f"beta_mul={res.x[0]:.3f}, cscale_mul={res.x[1]:.3f}, MAPE={res.fun:.2f}%")
    return res.x

################################################################################
# 4  大规模正式仿真
################################################################################
def final_run(run_sim, data_dir, pop_real, beta_mul, cscale_mul):
    POP_SIM = 400_000               # 回到大规模
    BURNIN  = 14
    MONTHS  = 10
    START   = "2022-01-01"
    SEED    = 42
    CASESFILE = data_dir / "observed_ZAF_2023.csv"

    out = run_sim("south_africa", data_dir, POP_SIM,
                  MONTHS, SEED, -1, pop_real, START, CASESFILE,
                  burnin_months=BURNIN, show_plot=True,
                  beta_mul=beta_mul, contact_scale_mul=cscale_mul)
    print(f"\nFINAL MAPE = {out['MAPE']:.2f}%")

################################################################################
# 5  main
################################################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_file", type=pathlib.Path, default="333.py")
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data"))
    ap.add_argument("--pop_real", type=int, default=60_456_600)
    ap.add_argument("--mode", choices=["grid", "bayes"], default="grid")
    args = ap.parse_args()

    run_sim = load_run_sim(args.model_file)

    tic = time.time()
    if args.mode == "grid":
        beta_mul, cscale_mul = grid_search(run_sim, args.data, args.pop_real)
    else:
        res = bayes_opt(run_sim, args.data, args.pop_real)
        if res is None:
            beta_mul, cscale_mul = 0.8, 1.10        # fall-back
        else:
            beta_mul, cscale_mul = res

    print(f"\n>>> BEST beta_mul={beta_mul:.3f}, cscale_mul={cscale_mul:.3f} <<<")
    print("  —— now running full-scale simulation …")
    final_run(run_sim, args.data, args.pop_real, beta_mul, cscale_mul)
    print(f"Total elapsed: {time.time()-tic:.1f} s")

if __name__ == "__main__":
    main()
