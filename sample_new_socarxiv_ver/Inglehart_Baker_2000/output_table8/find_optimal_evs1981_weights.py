"""
Find the optimal weight variable for EVS 1981 data to match paper's Table 8 values.
Test all available weight variables in ZA4804.
"""
import pandas as pd
import numpy as np
import os

evs_long_path = '/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/OldFiles/Replication_Claude_IB/data/ZA4804_v3-1-0.dta/ZA4804_v3-1-0.dta'

# Read with weight columns
print("Loading EVS Longitudinal data...")
# First find all weight columns
reader = pd.read_stata(evs_long_path, convert_categoricals=False, iterator=True)
all_labels = reader.variable_labels()
weight_cols = []
for var, label in all_labels.items():
    if 'weight' in var.lower() or 'weight' in label.lower() or var in ['S017', 'S017A', 'S017B', 'S018', 'S018A']:
        weight_cols.append(var)
        print(f"  Weight column: {var} - {label}")

# Also look for common weight variable names
for var in all_labels:
    if var.lower().startswith(('w_', 'wt_', 'gew', 'poids', 'peso')):
        if var not in weight_cols:
            weight_cols.append(var)
            print(f"  Weight column: {var} - {all_labels[var]}")

# Read the needed columns
cols_to_read = ['S002EVS', 'S003', 'S003A', 'F001', 'S020'] + weight_cols
print(f"\nReading columns: {cols_to_read}")
evs = pd.read_stata(evs_long_path, convert_categoricals=False, columns=cols_to_read)

# Wave 1 only
w1 = evs[evs['S002EVS'] == 1]
print(f"Wave 1: {len(w1)} rows")

# Paper ground truth for 1981
gt = {
    56: ('Belgium', 22),
    124: ('Canada', 38),
    250: ('France', 36),
    276: ('West Germany', 29),
    826: ('Great Britain', 34),
    352: ('Iceland', 39),
    372: ('Ireland', 25),
    380: ('Italy', 36),
    528: ('Netherlands', 21),
    578: ('Norway', 26),
    724: ('Spain', 24),
    752: ('Sweden', 20),
    840: ('United States', 48),
    909: ('Northern Ireland', 29),
}

def pct(f001, weights=None):
    mask = f001 > 0
    f = f001[mask]
    if len(f) == 0: return None
    if weights is not None:
        w = weights[mask].fillna(1)
        return ((f == 1) * w).sum() / w.sum() * 100
    return (f == 1).mean() * 100

# Test each weight
print(f"\n{'Country':<22} {'Paper':>6}", end='')
print(f" {'Unwtd':>7}", end='')
for wc in weight_cols:
    print(f" {wc:>10}", end='')
print()
print("-" * (30 + 7 + 11 * len(weight_cols)))

# Track matches per weight
matches = {'unwtd': 0}
for wc in weight_cols:
    matches[wc] = 0

for s003, (name, paper_val) in sorted(gt.items(), key=lambda x: x[1][0]):
    sub = w1[w1['S003'] == s003]
    print(f"{name:<22} {paper_val:>6}", end='')

    # Unweighted
    p = pct(sub['F001'])
    r = round(p) if p else None
    marker = '*' if r == paper_val else ' '
    print(f" {r:>5}{marker}", end='')
    if r == paper_val: matches['unwtd'] += 1

    # Each weight
    for wc in weight_cols:
        p = pct(sub['F001'], sub[wc])
        r = round(p) if p else None
        marker = '*' if r == paper_val else ' '
        print(f" {r:>8}{marker}", end='')
        if r == paper_val: matches[wc] += 1

    print()

print()
print("Matches per weight:")
for wc, m in sorted(matches.items(), key=lambda x: -x[1]):
    print(f"  {wc}: {m}/{len(gt)} exact matches")

# Also show which weight is best for each country
print("\nBest weight per country:")
for s003, (name, paper_val) in sorted(gt.items(), key=lambda x: x[1][0]):
    sub = w1[w1['S003'] == s003]
    best_diff = 999
    best_w = None
    all_opts = {}

    p = pct(sub['F001'])
    r = round(p) if p else None
    if r is not None:
        all_opts['unwtd'] = (p, r)
        diff = abs(r - paper_val)
        if diff < best_diff:
            best_diff = diff
            best_w = 'unwtd'

    for wc in weight_cols:
        p = pct(sub['F001'], sub[wc])
        r = round(p) if p else None
        if r is not None:
            all_opts[wc] = (p, r)
            diff = abs(r - paper_val)
            if diff < best_diff:
                best_diff = diff
                best_w = wc

    status = 'EXACT' if best_diff == 0 else f'OFF_BY_{best_diff}'
    print(f"  {name:<22}: best={best_w:<10} ({status}), paper={paper_val}, got={all_opts[best_w][1]}, raw={all_opts[best_w][0]:.2f}")
