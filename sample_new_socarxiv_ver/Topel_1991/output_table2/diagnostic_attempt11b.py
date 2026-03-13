#!/usr/bin/env python3
"""
Detailed check: what significance stars does the paper expect vs what we get?
Also check: what if we DON'T reconstruct tenure for in-progress jobs?
"""
import pandas as pd
import numpy as np
from scipy import stats

# Paper ground truth
gt = [
    (1, 'Delta Tenure', 0.1242, 0.0161),
    (1, 'd_exp_sq', -0.6051, 0.1430),
    (1, 'd_exp_cu', 0.1460, 0.0482),
    (1, 'd_exp_qu', 0.0131, 0.0054),
    (2, 'Delta Tenure', 0.1265, 0.0162),
    (2, 'd_ten_sq', -0.0518, 0.0178),
    (2, 'd_exp_sq', -0.6144, 0.1430),
    (2, 'd_exp_cu', 0.1620, 0.0485),
    (2, 'd_exp_qu', 0.0151, 0.0055),
    (3, 'Delta Tenure', 0.1258, 0.0162),
    (3, 'd_ten_sq', -0.4592, 0.1080),
    (3, 'd_ten_cu', 0.1846, 0.0526),
    (3, 'd_ten_qu', -0.0245, 0.0079),
    (3, 'd_exp_sq', -0.4067, 0.1546),
    (3, 'd_exp_cu', 0.0989, 0.0517),
    (3, 'd_exp_qu', 0.0089, 0.0058),
]

print("Paper's expected significance stars:")
for mod, var, c, s in gt:
    t_stat = abs(c / s) if s > 0 else 0
    p = 2 * (1 - stats.norm.cdf(t_stat))
    star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  Model {mod} {var:20s}: coef={c:>8.4f} SE={s:.4f} |t|={t_stat:.2f} => {star}")

# Our best attempt (attempt 8, retry_10) coefficients
our = [
    (1, 'Delta Tenure', 0.1212, 0.0145),
    (1, 'd_exp_sq', -0.1856, 0.1085),
    (1, 'd_exp_cu', 0.0143, 0.0360),
    (1, 'd_exp_qu', 0.0004, 0.0040),
    (2, 'Delta Tenure', 0.1225, 0.0146),
    (2, 'd_ten_sq', -0.0339, 0.0146),
    (2, 'd_exp_sq', -0.1893, 0.1085),
    (2, 'd_exp_cu', 0.0185, 0.0361),
    (2, 'd_exp_qu', 0.0009, 0.0040),
    (3, 'Delta Tenure', 0.1220, 0.0146),
    (3, 'd_ten_sq', -0.2240, 0.0850),
    (3, 'd_ten_cu', 0.1173, 0.0451),
    (3, 'd_ten_qu', -0.0168, 0.0071),
    (3, 'd_exp_sq', -0.1507, 0.1116),
    (3, 'd_exp_cu', 0.0003, 0.0371),
    (3, 'd_exp_qu', 0.0012, 0.0041),
]

print("\nOur best attempt's significance:")
for mod, var, c, s in our:
    t_stat = abs(c / s) if s > 0 else 0
    p = 2 * (1 - stats.norm.cdf(t_stat))
    star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  Model {mod} {var:20s}: coef={c:>8.4f} SE={s:.4f} |t|={t_stat:.2f} => {star}")

# Compare stars
print("\nComparison (paper_star vs our_star):")
matches = 0
for (pm, pv, pc, ps), (om, ov, oc, os) in zip(gt, our):
    pt = abs(pc / ps); pp = 2 * (1 - stats.norm.cdf(pt))
    ot = abs(oc / os); op = 2 * (1 - stats.norm.cdf(ot))
    pst = '***' if pp < 0.01 else '**' if pp < 0.05 else '*' if pp < 0.1 else ''
    ost = '***' if op < 0.01 else '**' if op < 0.05 else '*' if op < 0.1 else ''
    match = 'MATCH' if pst == ost else 'MISS'
    if pst == ost:
        matches += 1
    print(f"  M{pm} {pv:20s}: paper={pst:>3s}  ours={ost:>3s}  {match}")
print(f"\nTotal matches: {matches}/16")
