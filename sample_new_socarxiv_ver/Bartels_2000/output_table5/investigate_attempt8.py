"""Investigate potential improvements for attempt 8."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

# 1. Check weight variables
vcf9_cols = [c for c in cdf.columns if c.startswith('VCF0009')]
print('=== WEIGHT VARIABLES ===')
print('VCF0009* columns:', vcf9_cols)
for c in vcf9_cols:
    for year in [1960, 1976, 1992]:
        vals = cdf.loc[cdf['VCF0004']==year, c].dropna()
        if len(vals) > 0:
            print(f'  {c} ({year}): n={len(vals)}, min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}, nunique={vals.nunique()}')

# 2. Check VCF0006 columns
vcf6_cols = [c for c in cdf.columns if c.startswith('VCF0006')]
print(f'\nVCF0006* columns: {vcf6_cols}')

# 3. For each year, check panel identification
for year_curr, year_lag in [(1960, 1958), (1976, 1974), (1992, 1990)]:
    cdf_curr = cdf[cdf['VCF0004']==year_curr].copy()
    cdf_lag = cdf[cdf['VCF0004']==year_lag].copy()
    print(f'\n=== {year_curr} Panel ===')
    print(f'  {year_curr} study total: {len(cdf_curr)}')
    print(f'  {year_lag} study total: {len(cdf_lag)}')

    ids_lag = set(cdf_lag['VCF0006a'].values)
    threshold = year_curr * 10000
    panel = cdf_curr[cdf_curr['VCF0006a'] < threshold]
    panel_matched = panel[panel['VCF0006a'].isin(ids_lag)]
    print(f'  Panel (VCF0006a < {threshold}): {len(panel)}')
    print(f'  Panel matched in {year_lag} data: {len(panel_matched)}')

    # Check VCF0006a distribution
    ids = cdf_curr['VCF0006a']
    print(f'  VCF0006a range: {ids.min()} to {ids.max()}')

    # Check if VCF0006b exists and is useful
    if 'VCF0006b' in cdf.columns:
        v6b = cdf_curr['VCF0006b'].dropna()
        if len(v6b) > 0:
            print(f'  VCF0006b: n={len(v6b)}, unique values: {sorted(v6b.unique())[:10]}')

# 4. Try different approaches for 1960 to see which matches better
print('\n=== 1960 APPROACHES ===')

def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf58 = cdf[cdf['VCF0004']==1958].copy()
panel60 = cdf60[cdf60['VCF0006a'] < 19600000].copy()
merged = panel60.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# Try different weight rounding approaches
wt_raw = merged['VCF0009x'].fillna(1.0)
print(f'\n1960 weights: min={wt_raw.min()}, max={wt_raw.max()}, mean={wt_raw.mean():.3f}')
print(f'Weight distribution:')
print(wt_raw.value_counts().sort_index())

# Approach A: Round to nearest int
wt_int = wt_raw.round().astype(int)
sum_a = wt_int.sum()

# Approach B: Use raw float weights - multiply by scaling factor
# Target N = 911, current unweighted N with valid data varies
# Try different scaling

# First, get the valid respondents with union vote
merged_test = merged.copy()
merged_test['house_vote'] = merged_test['VCF0707']
mask_fill = merged_test['house_vote'].isna() & merged_test['VCF0706'].isin([1.0, 2.0])
merged_test.loc[mask_fill, 'house_vote'] = merged_test.loc[mask_fill, 'VCF0706']
valid_mask = (
    merged_test['house_vote'].isin([1.0, 2.0]) &
    merged_test['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged_test['VCF0301_lag'].isin([1,2,3,4,5,6,7])
)
valid_unwtd = merged_test[valid_mask]
print(f'\nUnweighted valid N (union): {len(valid_unwtd)}')
wt_valid = valid_unwtd['VCF0009x'].fillna(1.0)
print(f'Sum of weights for valid: {wt_valid.sum():.1f}')
print(f'Sum of int weights for valid: {wt_valid.astype(int).sum()}')
print(f'Sum of rounded weights for valid: {wt_valid.round().astype(int).sum()}')

# Approach C: Scale weights to match N=911
scale_factor = 911 / wt_valid.sum()
print(f'Scale factor to get N=911: {scale_factor:.4f}')
wt_scaled = (wt_valid * scale_factor).round().astype(int).clip(lower=1)
print(f'Scaled weight sum: {wt_scaled.sum()}')

# Approach D: Try VCF0707 only (no union) with weights
valid_707 = merged_test.copy()
valid_707 = valid_707[
    valid_707['VCF0707'].isin([1.0, 2.0]) &
    valid_707['VCF0301'].isin([1,2,3,4,5,6,7]) &
    valid_707['VCF0301_lag'].isin([1,2,3,4,5,6,7])
]
wt_707 = valid_707['VCF0009x'].fillna(1.0)
print(f'\nVCF0707-only valid N: {len(valid_707)}')
print(f'Sum of int weights (707-only): {wt_707.astype(int).sum()}')
print(f'Sum of rounded weights (707-only): {wt_707.round().astype(int).sum()}')

# 5. Investigate VCF0009z (if exists)
print('\n=== VCF0009z for 1960 ===')
if 'VCF0009z' in cdf.columns:
    v9z = cdf60['VCF0009z'].dropna()
    print(f'VCF0009z: n={len(v9z)}, min={v9z.min():.3f}, max={v9z.max():.3f}')
    print(f'Distribution:')
    print(v9z.describe())
else:
    print('VCF0009z not found')

# 6. Check if VCF0009x are actually float weights that should be applied differently
print('\n=== Float weights approach ===')
# Try using statsmodels freq_weights with rounded weights
valid_union_exp = merged_test[valid_mask].copy()
valid_union_exp['house_rep'] = (valid_union_exp['house_vote'] == 2.0).astype(int)
valid_union_exp = construct_pid_vars(valid_union_exp, 'VCF0301', 'curr')
valid_union_exp = construct_pid_vars(valid_union_exp, 'VCF0301_lag', 'lag')

# Method 1: Expand by int weights (current approach)
wt = valid_union_exp['VCF0009x'].fillna(1.0).astype(int)
expanded = valid_union_exp.loc[valid_union_exp.index.repeat(wt)].reset_index(drop=True)
X_exp = sm.add_constant(expanded[['strong_curr','weak_curr','lean_curr']])
mod_exp = Probit(expanded['house_rep'].astype(float), X_exp).fit(disp=0)
print(f'Expanded int (N={len(expanded)}): LL={mod_exp.llf:.1f}, Strong={mod_exp.params["strong_curr"]:.3f}, Weak={mod_exp.params["weak_curr"]:.3f}')

# Method 2: Expand by rounded weights
wt_r = valid_union_exp['VCF0009x'].fillna(1.0).round().astype(int)
expanded_r = valid_union_exp.loc[valid_union_exp.index.repeat(wt_r)].reset_index(drop=True)
X_r = sm.add_constant(expanded_r[['strong_curr','weak_curr','lean_curr']])
mod_r = Probit(expanded_r['house_rep'].astype(float), X_r).fit(disp=0)
print(f'Expanded round (N={len(expanded_r)}): LL={mod_r.llf:.1f}, Strong={mod_r.params["strong_curr"]:.3f}, Weak={mod_r.params["weak_curr"]:.3f}')

# Method 3: No weights at all
X_nw = sm.add_constant(valid_union_exp[['strong_curr','weak_curr','lean_curr']])
mod_nw = Probit(valid_union_exp['house_rep'].astype(float), X_nw).fit(disp=0)
print(f'No weights (N={len(valid_union_exp)}): LL={mod_nw.llf:.1f}, Strong={mod_nw.params["strong_curr"]:.3f}, Weak={mod_nw.params["weak_curr"]:.3f}')

# Method 4: VCF0707-only, no weights
valid_707_only = merged.copy()
valid_707_only = valid_707_only[
    valid_707_only['VCF0707'].isin([1.0, 2.0]) &
    valid_707_only['VCF0301'].isin([1,2,3,4,5,6,7]) &
    valid_707_only['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid_707_only['house_rep'] = (valid_707_only['VCF0707'] == 2.0).astype(int)
valid_707_only = construct_pid_vars(valid_707_only, 'VCF0301', 'curr')
X_707 = sm.add_constant(valid_707_only[['strong_curr','weak_curr','lean_curr']])
mod_707 = Probit(valid_707_only['house_rep'].astype(float), X_707).fit(disp=0)
print(f'VCF0707-only, no weights (N={len(valid_707_only)}): LL={mod_707.llf:.1f}, Strong={mod_707.params["strong_curr"]:.3f}, Weak={mod_707.params["weak_curr"]:.3f}')

# 7. For 1976 and 1992 - check alternative vote variables
print('\n=== 1976 ALTERNATIVE APPROACHES ===')
cdf76 = cdf[cdf['VCF0004']==1976].copy()
cdf74 = cdf[cdf['VCF0004']==1974].copy()
panel76 = cdf76[cdf76['VCF0006a'] < 19760000].copy()
merged76 = panel76.merge(cdf74[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# VCF0707 only
v707_76 = merged76[merged76['VCF0707'].isin([1.0, 2.0]) &
                    merged76['VCF0301'].isin([1,2,3,4,5,6,7]) &
                    merged76['VCF0301_lag'].isin([1,2,3,4,5,6,7])].copy()
v707_76['house_rep'] = (v707_76['VCF0707'] == 2.0).astype(int)
v707_76 = construct_pid_vars(v707_76, 'VCF0301', 'curr')
X76_707 = sm.add_constant(v707_76[['strong_curr','weak_curr','lean_curr']])
mod76_707 = Probit(v707_76['house_rep'].astype(float), X76_707).fit(disp=0)
print(f'VCF0707-only (N={len(v707_76)}): LL={mod76_707.llf:.1f}, R2={mod76_707.prsquared:.4f}')
print(f'  Strong={mod76_707.params["strong_curr"]:.3f}({mod76_707.bse["strong_curr"]:.3f})')
print(f'  Weak={mod76_707.params["weak_curr"]:.3f}({mod76_707.bse["weak_curr"]:.3f})')
print(f'  Lean={mod76_707.params["lean_curr"]:.3f}({mod76_707.bse["lean_curr"]:.3f})')
print(f'  Int={mod76_707.params["const"]:.3f}({mod76_707.bse["const"]:.3f})')

# Union
merged76u = merged76.copy()
merged76u['house_vote'] = merged76u['VCF0707']
mask_fill76 = merged76u['house_vote'].isna() & merged76u['VCF0706'].isin([1.0, 2.0])
merged76u.loc[mask_fill76, 'house_vote'] = merged76u.loc[mask_fill76, 'VCF0706']
v_union_76 = merged76u[merged76u['house_vote'].isin([1.0, 2.0]) &
                        merged76u['VCF0301'].isin([1,2,3,4,5,6,7]) &
                        merged76u['VCF0301_lag'].isin([1,2,3,4,5,6,7])].copy()
v_union_76['house_rep'] = (v_union_76['house_vote'] == 2.0).astype(int)
v_union_76 = construct_pid_vars(v_union_76, 'VCF0301', 'curr')
X76_u = sm.add_constant(v_union_76[['strong_curr','weak_curr','lean_curr']])
mod76_u = Probit(v_union_76['house_rep'].astype(float), X76_u).fit(disp=0)
print(f'\nUnion (N={len(v_union_76)}): LL={mod76_u.llf:.1f}, R2={mod76_u.prsquared:.4f}')
print(f'  Strong={mod76_u.params["strong_curr"]:.3f}({mod76_u.bse["strong_curr"]:.3f})')
print(f'  Weak={mod76_u.params["weak_curr"]:.3f}({mod76_u.bse["weak_curr"]:.3f})')
print(f'  Lean={mod76_u.params["lean_curr"]:.3f}({mod76_u.bse["lean_curr"]:.3f})')
print(f'  Int={mod76_u.params["const"]:.3f}({mod76_u.bse["const"]:.3f})')

# Targets for 1976 current: Strong=1.087, Weak=0.624, Lean=0.622, Int=-0.123, LL=-358.2, R2=0.24, N=682
print(f'\nTarget 1976: N=682, LL=-358.2, R2=0.24, Strong=1.087, Weak=0.624, Lean=0.622, Int=-0.123')

# 8. For 1992 - try using VCF0706 (intent) as alternative
print('\n=== 1992 ALTERNATIVE APPROACHES ===')
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf90 = cdf[cdf['VCF0004']==1990].copy()
panel92 = cdf92[cdf92['VCF0006a'] < 19920000].copy()
merged92 = panel92.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# Current approach: VCF0707 only
v707_92 = merged92[merged92['VCF0707'].isin([1.0, 2.0]) &
                    merged92['VCF0301'].isin([1,2,3,4,5,6,7]) &
                    merged92['VCF0301_lag'].isin([1,2,3,4,5,6,7])].copy()
v707_92['house_rep'] = (v707_92['VCF0707'] == 2.0).astype(int)
v707_92 = construct_pid_vars(v707_92, 'VCF0301', 'curr')
X92_707 = sm.add_constant(v707_92[['strong_curr','weak_curr','lean_curr']])
mod92_707 = Probit(v707_92['house_rep'].astype(float), X92_707).fit(disp=0)
print(f'VCF0707-only (N={len(v707_92)}): LL={mod92_707.llf:.1f}, R2={mod92_707.prsquared:.4f}')
print(f'  Strong={mod92_707.params["strong_curr"]:.3f}({mod92_707.bse["strong_curr"]:.3f})')
print(f'  Weak={mod92_707.params["weak_curr"]:.3f}({mod92_707.bse["weak_curr"]:.3f})')
print(f'  Lean={mod92_707.params["lean_curr"]:.3f}({mod92_707.bse["lean_curr"]:.3f})')

# Union
merged92u = merged92.copy()
merged92u['house_vote'] = merged92u['VCF0707']
mask_fill92 = merged92u['house_vote'].isna() & merged92u['VCF0706'].isin([1.0, 2.0])
merged92u.loc[mask_fill92, 'house_vote'] = merged92u.loc[mask_fill92, 'VCF0706']
v_union_92 = merged92u[merged92u['house_vote'].isin([1.0, 2.0]) &
                        merged92u['VCF0301'].isin([1,2,3,4,5,6,7]) &
                        merged92u['VCF0301_lag'].isin([1,2,3,4,5,6,7])].copy()
v_union_92['house_rep'] = (v_union_92['house_vote'] == 2.0).astype(int)
v_union_92 = construct_pid_vars(v_union_92, 'VCF0301', 'curr')
X92_u = sm.add_constant(v_union_92[['strong_curr','weak_curr','lean_curr']])
mod92_u = Probit(v_union_92['house_rep'].astype(float), X92_u).fit(disp=0)
print(f'\nUnion (N={len(v_union_92)}): LL={mod92_u.llf:.1f}, R2={mod92_u.prsquared:.4f}')
print(f'  Strong={mod92_u.params["strong_curr"]:.3f}({mod92_u.bse["strong_curr"]:.3f})')
print(f'  Weak={mod92_u.params["weak_curr"]:.3f}({mod92_u.bse["weak_curr"]:.3f})')
print(f'  Lean={mod92_u.params["lean_curr"]:.3f}({mod92_u.bse["lean_curr"]:.3f})')

print(f'\nTarget 1992: N=760, LL=-408.2, R2=0.20, Strong=0.975, Weak=0.627, Lean=0.472, Int=-0.211')

# 9. Score comparison: For each panel, which approach is closer to targets
# Score each coefficient's distance from target
print('\n=== SCORING BY APPROACH ===')

targets = {
    '1960': {'strong': 1.358, 'weak': 1.028, 'lean': 0.855, 'int': 0.035, 'N': 911, 'LL': -372.7, 'R2': 0.41},
    '1976': {'strong': 1.087, 'weak': 0.624, 'lean': 0.622, 'int': -0.123, 'N': 682, 'LL': -358.2, 'R2': 0.24},
    '1992': {'strong': 0.975, 'weak': 0.627, 'lean': 0.472, 'int': -0.211, 'N': 760, 'LL': -408.2, 'R2': 0.20},
}

# Score 1960 approaches
print('\n1960 Current PID scoring (coefficient distance sum):')
for label, mod, n in [('Expanded int + union', mod_exp, len(expanded)),
                       ('Expanded round + union', mod_r, len(expanded_r)),
                       ('No wt + union', mod_nw, len(valid_union_exp)),
                       ('VCF0707 only, no wt', mod_707, len(valid_707_only))]:
    t = targets['1960']
    cd = abs(mod.params['strong_curr'] - t['strong']) + abs(mod.params['weak_curr'] - t['weak']) + \
         abs(mod.params['lean_curr'] - t['lean']) + abs(mod.params['const'] - t['int'])
    ld = abs(mod.llf - t['LL'])
    r2d = abs(mod.prsquared - t['R2'])
    nd = abs(n - t['N']) / t['N']
    print(f'  {label}: coef_dist={cd:.3f}, LL_diff={ld:.1f}, R2_diff={r2d:.4f}, N_diff={nd:.1%} (N={n})')

# Score 1976 approaches
print('\n1976 Current PID scoring:')
for label, mod, n in [('VCF0707 only', mod76_707, len(v707_76)),
                       ('Union', mod76_u, len(v_union_76))]:
    t = targets['1976']
    cd = abs(mod.params['strong_curr'] - t['strong']) + abs(mod.params['weak_curr'] - t['weak']) + \
         abs(mod.params['lean_curr'] - t['lean']) + abs(mod.params['const'] - t['int'])
    ld = abs(mod.llf - t['LL'])
    r2d = abs(mod.prsquared - t['R2'])
    nd = abs(n - t['N']) / t['N']
    print(f'  {label}: coef_dist={cd:.3f}, LL_diff={ld:.1f}, R2_diff={r2d:.4f}, N_diff={nd:.1%} (N={n})')

# Score 1992 approaches
print('\n1992 Current PID scoring:')
for label, mod, n in [('VCF0707 only', mod92_707, len(v707_92)),
                       ('Union', mod92_u, len(v_union_92))]:
    t = targets['1992']
    cd = abs(mod.params['strong_curr'] - t['strong']) + abs(mod.params['weak_curr'] - t['weak']) + \
         abs(mod.params['lean_curr'] - t['lean']) + abs(mod.params['const'] - t['int'])
    ld = abs(mod.llf - t['LL'])
    r2d = abs(mod.prsquared - t['R2'])
    nd = abs(n - t['N']) / t['N']
    print(f'  {label}: coef_dist={cd:.3f}, LL_diff={ld:.1f}, R2_diff={r2d:.4f}, N_diff={nd:.1%} (N={n})')
