import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('gss1993_clean.csv')

# Check the paper's Table 1 racism scale stats: mean=2.65, SD=1.56
# Our old 5-item (sum of 0/1, range 0-5): mean ~3.00, SD ~1.29
# Our correct 5-item (sum of 0/1, range 0-5, strict all 5): mean ~2.42, SD ~1.47

# The paper says mean=2.65, SD=1.56
# What if the scale is actually a mean rather than a sum?
# If mean of 5 items (range 0-1), mean would be ~0.53, SD ~0.26 -- too small
# If sum of 5 items (range 0-5), mean ~2.65 -- matches!
# But our old 5-item gives mean=3.00, not 2.65

# What if we use the racism score computed only on the analysis sample?
# The Table 1 descriptives might be for a different sample than Table 2

# Let me check what racism mean/SD we get with different configurations
# and try to match mean=2.65, SD=1.56

for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())
df['r_racdif4'] = (df['racdif4'] == 1).astype(float).where(df['racdif4'].notna())

# Check ALL possible 5-item combinations and find which gives mean=2.65, SD=1.56
import itertools
all_items = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3', 'r_racdif4']
all_item_names = ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3(2)', 'racdif4']

print("Testing all 5-item combinations (strict, all 5 required):")
best_match = None
best_dist = 999
for combo in itertools.combinations(range(6), 5):
    items = [all_items[i] for i in combo]
    names = [all_item_names[i] for i in combo]
    score = pd.concat([df[c] for c in items], axis=1).sum(axis=1, min_count=5)
    mean = score.mean()
    sd = score.std(ddof=1)
    n_valid = score.notna().sum()
    dist = abs(mean - 2.65) + abs(sd - 1.56)
    if dist < best_dist:
        best_dist = dist
        best_match = (names, mean, sd, n_valid)
    if dist < 0.3:
        print(f"  {'+'.join(names)}: mean={mean:.2f}, SD={sd:.2f}, N_valid={n_valid}, dist={dist:.3f}")

print(f"\nBest match: {best_match[0]}, mean={best_match[1]:.2f}, SD={best_match[2]:.2f}, N={best_match[3]}")

# Also test with person-mean imputation (min 4)
print("\nTesting all 5-item combinations (PM imputation, min 4):")
best_match = None
best_dist = 999
for combo in itertools.combinations(range(6), 5):
    items = [all_items[i] for i in combo]
    names = [all_item_names[i] for i in combo]
    vals_list = []
    for idx in df.index:
        vals = [df.loc[idx, c] for c in items]
        n_v = sum(1 for v in vals if not np.isnan(v))
        if n_v >= 4:
            valid_v = [v for v in vals if not np.isnan(v)]
            pm = np.mean(valid_v)
            vals_list.append(sum(v if not np.isnan(v) else pm for v in vals))
        else:
            vals_list.append(np.nan)
    score = pd.Series(vals_list)
    mean = score.mean()
    sd = score.std(ddof=1)
    n_valid = score.notna().sum()
    dist = abs(mean - 2.65) + abs(sd - 1.56)
    if dist < best_dist:
        best_dist = dist
        best_match = (names, mean, sd, n_valid)
    if dist < 0.3:
        print(f"  {'+'.join(names)}: mean={mean:.2f}, SD={sd:.2f}, N_valid={n_valid}, dist={dist:.3f}")

print(f"\nBest match: {best_match[0]}, mean={best_match[1]:.2f}, SD={best_match[2]:.2f}, N={best_match[3]}")

# Also test 6-item scaled to 0-5 with PM imputation
print("\n\n6-item (all 6), scaled to 0-5, PM imputation min 4:")
items_6 = all_items
vals_list = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in items_6]
    n_v = sum(1 for v in vals if not np.isnan(v))
    if n_v >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        total = sum(v if not np.isnan(v) else pm for v in vals) * (5.0/6.0)
        vals_list.append(total)
    else:
        vals_list.append(np.nan)
score = pd.Series(vals_list)
print(f"  mean={score.mean():.2f}, SD={score.std(ddof=1):.2f}, N={score.notna().sum()}")
