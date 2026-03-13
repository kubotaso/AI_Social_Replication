"""Investigate alternative PID variable constructions and other tricks."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

# Check if there's a VCF0302 or VCF0303 or other PID vars
pid_cols = [c for c in cdf.columns if c.startswith('VCF030')]
print('PID-related columns:', pid_cols)

# Check VCF0302 (PID strength)
for c in pid_cols:
    for year in [1960, 1976, 1992, 1958, 1974, 1990]:
        vals = cdf.loc[cdf['VCF0004']==year, c].dropna()
        if len(vals) > 0:
            print(f'  {c} ({year}): n={len(vals)}, unique={sorted(vals.unique())}')

# Check if VCF0301 coding is consistent - is it really 1-7?
print('\n=== VCF0301 value distributions ===')
for year in [1958, 1960, 1974, 1976, 1990, 1992]:
    vals = cdf.loc[cdf['VCF0004']==year, 'VCF0301']
    print(f'  {year}: {dict(vals.value_counts().sort_index())}')

# Check if there are respondents with VCF0301 == 0 or 8 or 9 that we might be excluding
print('\n=== Panel respondents with unusual PID values ===')
for year_curr, year_lag in [(1960, 1958), (1976, 1974), (1992, 1990)]:
    cdf_curr = cdf[cdf['VCF0004']==year_curr].copy()
    cdf_lag = cdf[cdf['VCF0004']==year_lag].copy()
    threshold = year_curr * 10000
    panel = cdf_curr[cdf_curr['VCF0006a'] < threshold].copy()
    merged = panel.merge(cdf_lag[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

    # How many have VCF0301 outside 1-7?
    out_curr = merged[~merged['VCF0301'].isin([1,2,3,4,5,6,7])]
    out_lag = merged[~merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])]
    print(f'\n  {year_curr} panel:')
    print(f'    Current PID outside 1-7: {len(out_curr)} ({out_curr["VCF0301"].value_counts().to_dict()})')
    print(f'    Lagged PID outside 1-7: {len(out_lag)} ({out_lag["VCF0301_lag"].value_counts().to_dict()})')

# Check alternative: maybe Bartels includes respondents with PID=0 or PID=8
# Try including PID=0 (apolitical/DK) as part of the independent category (PID=4)
print('\n=== Try including PID=0 as Independent ===')
def construct_pid_vars_with0(df, pid_col, suffix):
    """Include PID=0 (apolitical) as independent."""
    pid = df[pid_col].copy()
    pid = pid.replace(0, 4)  # Recode 0 to 4 (independent)
    df[f'strong_{suffix}'] = np.where(pid == 7, 1, np.where(pid == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(pid == 6, 1, np.where(pid == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(pid == 5, 1, np.where(pid == 3, -1, 0))
    return df

# Try for 1960 with expanded weights + union
cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf58 = cdf[cdf['VCF0004']==1958].copy()
panel60 = cdf60[cdf60['VCF0006a'] < 19600000].copy()
merged60 = panel60.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# Expand weights
wt = merged60['VCF0009x'].fillna(1.0).astype(int)
expanded = merged60.loc[merged60.index.repeat(wt)].reset_index(drop=True)

# Union vote
expanded['house_vote'] = expanded['VCF0707']
mask_fill = expanded['house_vote'].isna() & expanded['VCF0706'].isin([1.0, 2.0])
expanded.loc[mask_fill, 'house_vote'] = expanded.loc[mask_fill, 'VCF0706']

# Try including PID values 0 and 8/9
for pid_range_label, pid_range in [('1-7', [1,2,3,4,5,6,7]),
                                     ('0-7', [0,1,2,3,4,5,6,7]),
                                     ('0-7 (0->4)', 'recode0')]:
    if pid_range == 'recode0':
        valid = expanded[
            expanded['house_vote'].isin([1.0, 2.0]) &
            expanded['VCF0301'].isin([0,1,2,3,4,5,6,7]) &
            expanded['VCF0301_lag'].isin([0,1,2,3,4,5,6,7])
        ].copy()
        valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
        valid = construct_pid_vars_with0(valid, 'VCF0301', 'curr')
        valid = construct_pid_vars_with0(valid, 'VCF0301_lag', 'lag')
    else:
        valid = expanded[
            expanded['house_vote'].isin([1.0, 2.0]) &
            expanded['VCF0301'].isin(pid_range) &
            expanded['VCF0301_lag'].isin(pid_range)
        ].copy()
        valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)

        def construct_pid_vars(df, pid_col, suffix):
            df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
            df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
            df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
            return df
        valid = construct_pid_vars(valid, 'VCF0301', 'curr')
        valid = construct_pid_vars(valid, 'VCF0301_lag', 'lag')

    X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
    mod = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
    print(f'\n  1960 {pid_range_label}: N={len(valid)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}')
    print(f'    Strong={mod.params["strong_curr"]:.3f}({mod.bse["strong_curr"]:.3f})')
    print(f'    Weak={mod.params["weak_curr"]:.3f}({mod.bse["weak_curr"]:.3f})')
    print(f'    Lean={mod.params["lean_curr"]:.3f}({mod.bse["lean_curr"]:.3f})')
    print(f'    Int={mod.params["const"]:.3f}({mod.bse["const"]:.3f})')

# 4. Check VCF0706 coding - is 1=Dem and 2=Rep consistent with VCF0707?
print('\n=== VCF0706 vs VCF0707 coding check ===')
for year in [1960, 1976, 1992]:
    subset = cdf[cdf['VCF0004']==year]
    both = subset[subset['VCF0707'].notna() & subset['VCF0706'].notna()]
    if len(both) > 0:
        agree = (both['VCF0707'] == both['VCF0706']).mean()
        print(f'  {year}: {len(both)} with both, agreement={agree:.1%}')
        cross = pd.crosstab(both['VCF0707'], both['VCF0706'])
        print(cross)

# 5. Check whether Bartels might be using VCF0706 (pre-election intent) instead of VCF0707
print('\n=== VCF0706-only for 1976 ===')
cdf76 = cdf[cdf['VCF0004']==1976].copy()
cdf74 = cdf[cdf['VCF0004']==1974].copy()
panel76 = cdf76[cdf76['VCF0006a'] < 19760000].copy()
merged76 = panel76.merge(cdf74[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

v706_76 = merged76[
    merged76['VCF0706'].isin([1.0, 2.0]) &
    merged76['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged76['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
v706_76['house_rep'] = (v706_76['VCF0706'] == 2.0).astype(int)

def cpv(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

v706_76 = cpv(v706_76, 'VCF0301', 'curr')
X_706 = sm.add_constant(v706_76[['strong_curr','weak_curr','lean_curr']])
mod_706 = Probit(v706_76['house_rep'].astype(float), X_706).fit(disp=0)
print(f'  VCF0706-only: N={len(v706_76)}, LL={mod_706.llf:.1f}, R2={mod_706.prsquared:.4f}')
print(f'    Strong={mod_706.params["strong_curr"]:.3f} Weak={mod_706.params["weak_curr"]:.3f} Lean={mod_706.params["lean_curr"]:.3f}')
print(f'  Target: N=682, LL=-358.2, R2=0.24, Strong=1.087, Weak=0.624, Lean=0.622')

# 6. Try VCF0706-only for 1960
print('\n=== VCF0706-only for 1960 (expanded) ===')
v706_60 = expanded[
    expanded['VCF0706'].isin([1.0, 2.0]) &
    expanded['VCF0301'].isin([1,2,3,4,5,6,7]) &
    expanded['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
if len(v706_60) > 0:
    v706_60['house_rep'] = (v706_60['VCF0706'] == 2.0).astype(int)
    v706_60 = cpv(v706_60, 'VCF0301', 'curr')
    X_706_60 = sm.add_constant(v706_60[['strong_curr','weak_curr','lean_curr']])
    mod_706_60 = Probit(v706_60['house_rep'].astype(float), X_706_60).fit(disp=0)
    print(f'  VCF0706-only: N={len(v706_60)}, LL={mod_706_60.llf:.1f}, R2={mod_706_60.prsquared:.4f}')
    print(f'    Strong={mod_706_60.params["strong_curr"]:.3f} Weak={mod_706_60.params["weak_curr"]:.3f} Lean={mod_706_60.params["lean_curr"]:.3f}')
else:
    print('  No VCF0706 data for 1960 panel respondents')

# 7. Try priority: VCF0706 first, fill with VCF0707 (reverse union)
print('\n=== Reverse union (VCF0706 first, fill with VCF0707) for 1976 ===')
merged76r = merged76.copy()
merged76r['house_vote'] = merged76r['VCF0706']
mask_fill = merged76r['house_vote'].isna() & merged76r['VCF0707'].isin([1.0, 2.0])
merged76r.loc[mask_fill, 'house_vote'] = merged76r.loc[mask_fill, 'VCF0707']
v_rev76 = merged76r[
    merged76r['house_vote'].isin([1.0, 2.0]) &
    merged76r['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged76r['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
v_rev76['house_rep'] = (v_rev76['house_vote'] == 2.0).astype(int)
v_rev76 = cpv(v_rev76, 'VCF0301', 'curr')
X_rev76 = sm.add_constant(v_rev76[['strong_curr','weak_curr','lean_curr']])
mod_rev76 = Probit(v_rev76['house_rep'].astype(float), X_rev76).fit(disp=0)
print(f'  Reverse union: N={len(v_rev76)}, LL={mod_rev76.llf:.1f}, R2={mod_rev76.prsquared:.4f}')
print(f'    Strong={mod_rev76.params["strong_curr"]:.3f} Weak={mod_rev76.params["weak_curr"]:.3f} Lean={mod_rev76.params["lean_curr"]:.3f}')
print(f'  Target: N=682, LL=-358.2, R2=0.24')
