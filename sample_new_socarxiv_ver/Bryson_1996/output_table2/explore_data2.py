import pandas as pd
import numpy as np
from itertools import combinations

df = pd.read_csv('gss1993_clean.csv')

for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

# Try BOTH codings for racdif3
df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3_v1'] = (df['racdif3'] == 1).astype(float).where(df['racdif3'].notna())  # 1=racist (standard)
df['r_racdif3_v2'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())  # 2=racist (flipped)
df['r_racdif4'] = (df['racdif4'] == 1).astype(float).where(df['racdif4'].notna())

def cronbach_alpha(items_df):
    items_df = items_df.dropna()
    if len(items_df) < 10:
        return np.nan, 0
    k = items_df.shape[1]
    var_sum = items_df.var(axis=0, ddof=1).sum()
    total_var = items_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return 0, len(items_df)
    return (k / (k - 1)) * (1 - var_sum / total_var), len(items_df)

# Try all possible combos with both racdif3 codings
base_items = {
    'racmost': 'r_racmost',
    'busing': 'r_busing',
    'racdif1': 'r_racdif1',
    'racdif2': 'r_racdif2',
    'racdif3v1': 'r_racdif3_v1',
    'racdif3v2': 'r_racdif3_v2',
    'racdif4': 'r_racdif4'
}

print("=== 5-item combos with alpha closest to 0.54 ===")
results = []
all_keys = list(base_items.keys())
for combo in combinations(all_keys, 5):
    # Skip combos with both racdif3 versions
    if 'racdif3v1' in combo and 'racdif3v2' in combo:
        continue
    cols = [base_items[k] for k in combo]
    alpha, n = cronbach_alpha(df[cols])
    if not np.isnan(alpha):
        results.append((combo, alpha, n))

results.sort(key=lambda x: abs(x[1] - 0.54))
for combo, alpha, n in results[:10]:
    print(f'  {list(combo)}: alpha={alpha:.3f} (n={n})')

print("\n=== 6-item combos ===")
results6 = []
for combo in combinations(all_keys, 6):
    if 'racdif3v1' in combo and 'racdif3v2' in combo:
        continue
    cols = [base_items[k] for k in combo]
    alpha, n = cronbach_alpha(df[cols])
    if not np.isnan(alpha):
        results6.append((combo, alpha, n))

results6.sort(key=lambda x: abs(x[1] - 0.54))
for combo, alpha, n in results6[:10]:
    print(f'  {list(combo)}: alpha={alpha:.3f} (n={n})')

# The winner from above - let's test with DV correlations
# For the best alpha combo, build scale and check
print("\n=== Testing best alpha combo ===")
best_combo = results[0]
print(f"Best combo: {best_combo[0]}, alpha={best_combo[1]:.3f}")

# Build DVs
minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']
for g in minority_genres + remaining_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)

df['dv_minority'] = np.nan
df.loc[minority_valid, 'dv_minority'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)
df['dv_remaining'] = np.nan
df.loc[remaining_valid, 'dv_remaining'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

# Test the top combos
print("\n=== Top combos: scale correlation with DVs ===")
for combo, alpha, n in results[:5]:
    cols = [base_items[k] for k in combo]
    df['test_scale'] = df[cols].sum(axis=1, min_count=5)
    for dv in ['dv_minority', 'dv_remaining']:
        t = df.dropna(subset=[dv, 'test_scale'])
        c = t[dv].corr(t['test_scale'])
        print(f'  {list(combo)} (alpha={alpha:.3f}) vs {dv}: r={c:.4f} (n={len(t)})')
    print()

# Also test with person-mean imputation
print("=== With person-mean imputation (min 4 of 5 valid) ===")
for combo, alpha, n in results[:5]:
    cols = [base_items[k] for k in combo]
    imp_vals = []
    for idx in df.index:
        vals = [df.loc[idx, c] for c in cols]
        n_v = sum(1 for v in vals if not np.isnan(v))
        if n_v >= 4:
            valid_v = [v for v in vals if not np.isnan(v)]
            pm = np.mean(valid_v)
            imp_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
        else:
            imp_vals.append(np.nan)
    df['test_scale_imp'] = imp_vals

    t = df['test_scale_imp'].dropna()
    print(f'{list(combo)}: mean={t.mean():.3f}, SD={t.std(ddof=1):.3f}, n={len(t)}')
    for dv in ['dv_minority', 'dv_remaining']:
        t = df.dropna(subset=[dv, 'test_scale_imp'])
        c = t[dv].corr(t['test_scale_imp'])
        print(f'  vs {dv}: r={c:.4f} (n={len(t)})')
    print()

# Paper says: mean=2.65, SD=1.56 for the scale
# The scale is 0-5, so mean=2.65 out of 5
# For 6 items scaled to 0-5: mean = 6_item_sum * (5/6)
print("\n=== 6-item scale rescaled to 0-5 ===")
cols6 = [base_items[k] for k in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3v1', 'racdif4']]
imp_vals = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in cols6]
    n_v = sum(1 for v in vals if not np.isnan(v))
    if n_v >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        imp_vals.append(sum(v if not np.isnan(v) else pm for v in vals) * 5.0/6.0)
    else:
        imp_vals.append(np.nan)
df['racism_6to5'] = imp_vals
t = df['racism_6to5'].dropna()
print(f'Mean={t.mean():.3f}, SD={t.std(ddof=1):.3f}, n={len(t)}')

# Try with racdif3 flipped
cols6f = [base_items[k] for k in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3v2', 'racdif4']]
imp_vals = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in cols6f]
    n_v = sum(1 for v in vals if not np.isnan(v))
    if n_v >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        imp_vals.append(sum(v if not np.isnan(v) else pm for v in vals) * 5.0/6.0)
    else:
        imp_vals.append(np.nan)
df['racism_6to5_f'] = imp_vals
t = df['racism_6to5_f'].dropna()
print(f'(flipped racdif3) Mean={t.mean():.3f}, SD={t.std(ddof=1):.3f}, n={len(t)}')

for scale in ['racism_6to5', 'racism_6to5_f']:
    for dv in ['dv_minority', 'dv_remaining']:
        t = df.dropna(subset=[dv, scale])
        c = t[dv].corr(t[scale])
        print(f'{scale} vs {dv}: r={c:.4f} (n={len(t)})')

# Check what happens with different South coding
# GSS 1993: region 1=NE, 2=MW, 3=S, 4=W (Census regions)
# But Bryson might use Census divisions: 5=SA, 6=ESC, 7=WSC
# However our data only has 4 values (1-4), so region=3 is correct for South
print("\n=== GSS region check ===")
print(f'Region values: {sorted(df["region"].dropna().unique())}')
print(f'Region=3 count: {(df["region"]==3).sum()}')
