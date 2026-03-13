#!/usr/bin/env python3
"""Detailed scoring analysis: what coefficient/significance fixes are possible?"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Ground truth
gt_coefs = {
    ('experience', 0): 0.0418, ('experience', 1): 0.0379, ('experience', 2): 0.0345,
    ('experience', 3): 0.0397, ('experience', 4): 0.0401,
    ('exp_sq', 0): -0.00079, ('exp_sq', 1): -0.00069, ('exp_sq', 2): -0.00072,
    ('exp_sq', 3): -0.00074, ('exp_sq', 4): -0.00073,
    ('tenure', 0): 0.0138, ('tenure', 1): -0.0015, ('tenure', 2): 0.0137,
    ('tenure', 3): 0.0060, ('tenure', 4): 0.0163,
    ('ct', 1): 0.0165, ('ct', 2): 0.0316,
    ('cen', 1): -0.0025, ('cen', 2): -0.0024,
    ('imp_ct', 3): 0.0053, ('imp_ct', 4): 0.0067,
    ('esq_int', 2): -0.00061, ('esq_int', 4): -0.00075,
    ('ten_int', 2): 0.0142, ('ten_int', 4): 0.0429,
}

gt_ses = {
    ('experience', 0): 0.0013, ('experience', 1): 0.0014, ('experience', 2): 0.0015,
    ('experience', 3): 0.0013, ('experience', 4): 0.0014,
    ('exp_sq', 0): 0.00003, ('exp_sq', 1): 0.000032, ('exp_sq', 2): 0.000069,
    ('exp_sq', 3): 0.000030, ('exp_sq', 4): 0.000069,
    ('tenure', 0): 0.0052, ('tenure', 1): 0.0015, ('tenure', 2): 0.0038,
    ('tenure', 3): 0.0073, ('tenure', 4): 0.0038,
    ('ct', 1): 0.0016, ('ct', 2): 0.0022,
    ('cen', 1): 0.0073, ('cen', 2): 0.0073,
    ('imp_ct', 3): 0.0036, ('imp_ct', 4): 0.0042,
    ('esq_int', 2): 0.000036, ('esq_int', 4): 0.000033,
    ('ten_int', 2): 0.0033, ('ten_int', 4): 0.0016,
}

# Current attempt 9 generated values
gen_coefs = {
    ('experience', 0): 0.040362, ('experience', 1): 0.038622, ('experience', 2): 0.036342,
    ('experience', 3): 0.040215, ('experience', 4): 0.039466,
    ('exp_sq', 0): -0.000676, ('exp_sq', 1): -0.000633, ('exp_sq', 2): -0.000509,
    ('exp_sq', 3): -0.000678, ('exp_sq', 4): -0.000667,
    ('tenure', 0): 0.024046, ('tenure', 1): 0.004961, ('tenure', 2): 0.031575,
    ('tenure', 3): 0.022618, ('tenure', 4): 0.041322,
    ('ct', 1): 0.017662, ('ct', 2): 0.024954,
    ('cen', 1): 0.000383, ('cen', 2): 0.002476,
    ('imp_ct', 3): 0.005030, ('imp_ct', 4): 0.023279,
    ('esq_int', 2): -0.000009, ('esq_int', 4): 0.000002,
    ('ten_int', 2): -0.002367, ('ten_int', 4): -0.004197,
}

gen_ses = {
    ('experience', 0): 0.001541, ('experience', 1): 0.001537, ('experience', 2): 0.001581,
    ('experience', 3): 0.001541, ('experience', 4): 0.001587,
    ('exp_sq', 0): 0.000037, ('exp_sq', 1): 0.000037, ('exp_sq', 2): 0.000046,
    ('exp_sq', 3): 0.000037, ('exp_sq', 4): 0.000046,
    ('tenure', 0): 0.001562, ('tenure', 1): 0.002207, ('tenure', 2): 0.005605,
    ('tenure', 3): 0.001623, ('tenure', 4): 0.002754,
    ('ct', 1): 0.001962, ('ct', 2): 0.002235,
    ('cen', 1): 0.001098, ('cen', 2): 0.001178,
    ('imp_ct', 3): 0.001560, ('imp_ct', 4): 0.002960,
    ('esq_int', 2): 0.000002, ('esq_int', 4): 0.000003,
    ('ten_int', 2): 0.000474, ('ten_int', 4): 0.000495,
}

print("=== DETAILED SCORING ANALYSIS ===")
print()

# Coefficient scoring
print("COEFFICIENT MATCHES (25 pts total):")
coef_match = 0
coef_total = 0
for key, target in gt_coefs.items():
    gen = gen_coefs.get(key)
    if gen is None:
        continue
    coef_total += 1
    if abs(target) < 0.01:
        match = abs(gen - target) / max(abs(target), 1e-8) <= 0.20
        rel_err = abs(gen - target) / max(abs(target), 1e-8)
        err_str = f"rel_err={rel_err:.2%}"
    else:
        match = abs(gen - target) <= 0.05
        err_str = f"abs_err={abs(gen-target):.5f}"
    if match:
        coef_match += 1
    status = "PASS" if match else "FAIL"
    print(f"  {key[0]:12s} col({key[1]+1}): gen={gen:>12.7f} target={target:>10.6f} {err_str:>20s} {status}")

print(f"\n  TOTAL: {coef_match}/{coef_total} = {25*coef_match/coef_total:.1f}/25 pts")

# Significance scoring
print("\nSIGNIFICANCE MATCHES (25 pts total):")
def get_stars(c, se):
    if se == 0: return ''
    t = abs(c / se)
    return '***' if t > 3.291 else '**' if t > 2.576 else '*' if t > 1.96 else ''

sig_match = 0
sig_total = 0
for key in gt_coefs:
    target_c = gt_coefs[key]
    target_se = gt_ses.get(key)
    gen_c = gen_coefs.get(key)
    gen_se = gen_ses.get(key)
    if target_se is None or gen_c is None or gen_se is None:
        continue
    sig_total += 1
    target_stars = get_stars(target_c, target_se)
    gen_stars = get_stars(gen_c, gen_se)
    match = target_stars == gen_stars
    if match:
        sig_match += 1
    status = "PASS" if match else "FAIL"
    t_stat_gen = abs(gen_c / gen_se) if gen_se > 0 else 0
    t_stat_tgt = abs(target_c / target_se) if target_se > 0 else 0
    print(f"  {key[0]:12s} col({key[1]+1}): gen={gen_stars:>4s} (t={t_stat_gen:6.2f}) target={target_stars:>4s} (t={t_stat_tgt:6.2f}) {status}")

print(f"\n  TOTAL: {sig_match}/{sig_total} = {25*sig_match/sig_total:.1f}/25 pts")

# SE scoring
print("\nSE MATCHES (15 pts total):")
se_match = 0
se_total = 0
for key in gt_ses:
    target_se = gt_ses[key]
    gen_se = gen_ses.get(key)
    if gen_se is None:
        continue
    se_total += 1
    match = abs(gen_se - target_se) <= 0.02
    if match:
        se_match += 1
    status = "PASS" if match else "FAIL"
    print(f"  {key[0]:12s} col({key[1]+1}): gen_se={gen_se:.6f} target_se={target_se:.6f} diff={abs(gen_se-target_se):.6f} {status}")

print(f"\n  TOTAL: {se_match}/{se_total} = {15*se_match/se_total:.1f}/15 pts")

# Now analyze: what changes could fix which items?
print("\n" + "="*60)
print("POTENTIAL FIXES ANALYSIS")
print("="*60)

# 1. Inverted censor: flips sign of cen coefficients
print("\n1. INVERTED CENSOR would change:")
for col in [1, 2]:
    key = ('cen', col)
    gen = gen_coefs[key]
    target = gt_coefs[key]
    new_gen = -gen  # flipped
    if abs(target) < 0.01:
        old_match = abs(gen - target) / max(abs(target), 1e-8) <= 0.20
        new_match = abs(new_gen - target) / max(abs(target), 1e-8) <= 0.20
    else:
        old_match = abs(gen - target) <= 0.05
        new_match = abs(new_gen - target) <= 0.05
    print(f"  cen col({col+1}): gen={gen:.6f} -> {new_gen:.6f} target={target:.6f} old={'PASS' if old_match else 'FAIL'} new={'PASS' if new_match else 'FAIL'}")
    # Also check significance
    old_stars = get_stars(gen, gen_ses[key])
    new_stars = get_stars(new_gen, gen_ses[key])  # same SE
    tgt_stars = get_stars(target, gt_ses[key])
    print(f"    sig: old={old_stars} new={new_stars} target={tgt_stars}")

# 2. What coefficient mismatches are fixable vs structural?
print("\n2. FIXABILITY ASSESSMENT:")
for key in sorted(gt_coefs.keys()):
    target = gt_coefs[key]
    gen = gen_coefs.get(key)
    if gen is None: continue
    if abs(target) < 0.01:
        match = abs(gen - target) / max(abs(target), 1e-8) <= 0.20
    else:
        match = abs(gen - target) <= 0.05
    if not match:
        fixable = "STRUCTURAL" if key[0] in ['esq_int', 'ten_int'] else "MAYBE" if abs(gen - target) < 0.02 else "HARD"
        print(f"  MISS {key[0]:12s} col({key[1]+1}): gen={gen:.6f} target={target:.6f} gap={gen-target:+.6f} [{fixable}]")
