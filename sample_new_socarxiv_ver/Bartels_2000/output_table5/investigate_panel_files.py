"""Investigate original panel data files for Table 5 replication."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

# Load all panel files
p60 = pd.read_csv('panel_1960.csv', low_memory=False)
p76 = pd.read_csv('panel_1976.csv', low_memory=False)
p92 = pd.read_csv('panel_1992.csv', low_memory=False)

print('=== Panel 1960 ===')
print(f'Shape: {p60.shape}')
print(f'Columns: {list(p60.columns[:30])}...')
print(f'VCF0004 values: {sorted(p60["VCF0004"].unique())}')

# For 1960 panel: need year=1960 respondents with valid PID from 1958
# The panel file should have both years
p60_1960 = p60[p60['VCF0004'] == 1960]
p60_1958 = p60[p60['VCF0004'] == 1958]
print(f'1960 wave: {len(p60_1960)}')
print(f'1958 wave: {len(p60_1958)}')

# Check VCF0006a overlap
ids_60 = set(p60_1960['VCF0006a'])
ids_58 = set(p60_1958['VCF0006a'])
overlap = ids_60 & ids_58
print(f'Respondents in both 1958 and 1960: {len(overlap)}')

# Check key variables: VCF0301 (PID), VCF0707 (House vote post), VCF0706 (House vote pre)
for year, subset in [(1960, p60_1960), (1958, p60_1958)]:
    pid = subset['VCF0301'].dropna()
    print(f'\n  Year {year}:')
    print(f'    VCF0301 (PID): n={len(pid)}, values={sorted(pid.unique())}')
    if 'VCF0707' in subset.columns:
        v707 = subset['VCF0707'].dropna()
        print(f'    VCF0707 (House post): n={len(v707)}, values={sorted(v707.unique())}')
    if 'VCF0706' in subset.columns:
        v706 = subset['VCF0706'].dropna()
        print(f'    VCF0706 (House pre): n={len(v706)}, values={sorted(v706.unique())}')
    if 'VCF0009x' in subset.columns:
        wt = subset['VCF0009x'].dropna()
        print(f'    VCF0009x (weight): n={len(wt)}, min={wt.min()}, max={wt.max()}, mean={wt.mean():.3f}')

print('\n=== Panel 1976 ===')
print(f'Shape: {p76.shape}')
p76_76 = p76[p76['VCF0004'] == 1976]
p76_74 = p76[p76['VCF0004'] == 1974]
print(f'1976 wave: {len(p76_76)}')
print(f'1974 wave: {len(p76_74)}')
ids_76 = set(p76_76['VCF0006a'])
ids_74 = set(p76_74['VCF0006a'])
overlap76 = ids_76 & ids_74
print(f'Respondents in both 1974 and 1976: {len(overlap76)}')

for year, subset in [(1976, p76_76), (1974, p76_74)]:
    pid = subset['VCF0301'].dropna()
    print(f'\n  Year {year}:')
    print(f'    VCF0301: n={len(pid)}, values={sorted(pid.unique())}')
    if 'VCF0707' in subset.columns:
        v707 = subset['VCF0707'].dropna()
        print(f'    VCF0707: n={len(v707)}, values={sorted(v707.unique())}')
    if 'VCF0706' in subset.columns:
        v706 = subset['VCF0706'].dropna()
        print(f'    VCF0706: n={len(v706)}, values={sorted(v706.unique())}')
    if 'VCF0009x' in subset.columns:
        wt = subset['VCF0009x'].dropna()
        print(f'    VCF0009x: n={len(wt)}, min={wt.min()}, max={wt.max()}, mean={wt.mean():.3f}')

print('\n=== Panel 1992 ===')
print(f'Shape: {p92.shape}')
print(f'Columns: {list(p92.columns)}')
print(f'First 5 rows:')
print(p92.head())
print(f'\npid_current distribution:')
print(p92['pid_current'].value_counts().sort_index())
print(f'\npid_lagged distribution:')
print(p92['pid_lagged'].value_counts().sort_index())
print(f'\nvote_house distribution:')
print(p92['vote_house'].value_counts().sort_index())
print(f'\nvote_pres distribution:')
print(p92['vote_pres'].value_counts().sort_index())
print(f'\nTotal rows: {len(p92)}')

# Now try running probit on each panel file
def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

# === 1960 Panel ===
print('\n\n========== 1960 PANEL PROBIT ==========')
# Merge 1960 and 1958 waves
merged60 = p60_1960.merge(p60_1958[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
print(f'Merged: {len(merged60)}')

# Expand by weights
wt60 = merged60['VCF0009x'].fillna(1.0).astype(int)
print(f'Weight distribution in merged:')
print(wt60.value_counts().sort_index())
expanded60 = merged60.loc[merged60.index.repeat(wt60)].reset_index(drop=True)

# Union vote
expanded60['house_vote'] = expanded60['VCF0707']
mask_fill = expanded60['house_vote'].isna() & expanded60['VCF0706'].isin([1.0, 2.0])
expanded60.loc[mask_fill, 'house_vote'] = expanded60.loc[mask_fill, 'VCF0706']

valid60 = expanded60[
    expanded60['house_vote'].isin([1.0, 2.0]) &
    expanded60['VCF0301'].isin([1,2,3,4,5,6,7]) &
    expanded60['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid60['house_rep'] = (valid60['house_vote'] == 2.0).astype(int)
valid60 = construct_pid_vars(valid60, 'VCF0301', 'curr')
valid60 = construct_pid_vars(valid60, 'VCF0301_lag', 'lag')

print(f'Valid N (expanded + union): {len(valid60)} (target 911)')
X60 = sm.add_constant(valid60[['strong_curr','weak_curr','lean_curr']])
mod60 = Probit(valid60['house_rep'].astype(float), X60).fit(disp=0)
print(f'Current: LL={mod60.llf:.1f}, R2={mod60.prsquared:.4f}')
print(f'  Strong={mod60.params["strong_curr"]:.3f}({mod60.bse["strong_curr"]:.3f})')
print(f'  Weak={mod60.params["weak_curr"]:.3f}({mod60.bse["weak_curr"]:.3f})')
print(f'  Lean={mod60.params["lean_curr"]:.3f}({mod60.bse["lean_curr"]:.3f})')
print(f'  Int={mod60.params["const"]:.3f}({mod60.bse["const"]:.3f})')
print(f'Targets: N=911, LL=-372.7, R2=0.41, Strong=1.358, Weak=1.028, Lean=0.855, Int=0.035')

X60_lag = sm.add_constant(valid60[['strong_lag','weak_lag','lean_lag']])
mod60_lag = Probit(valid60['house_rep'].astype(float), X60_lag).fit(disp=0)
print(f'\nLagged: LL={mod60_lag.llf:.1f}, R2={mod60_lag.prsquared:.4f}')
print(f'  Strong={mod60_lag.params["strong_lag"]:.3f}({mod60_lag.bse["strong_lag"]:.3f})')
print(f'  Weak={mod60_lag.params["weak_lag"]:.3f}({mod60_lag.bse["weak_lag"]:.3f})')
print(f'  Lean={mod60_lag.params["lean_lag"]:.3f}({mod60_lag.bse["lean_lag"]:.3f})')
print(f'  Int={mod60_lag.params["const"]:.3f}({mod60_lag.bse["const"]:.3f})')
print(f'Targets: LL=-403.9, R2=0.36, Strong=1.363, Weak=0.842, Lean=0.564, Int=0.068')

# Try VCF0707 only
valid60_707 = expanded60[
    expanded60['VCF0707'].isin([1.0, 2.0]) &
    expanded60['VCF0301'].isin([1,2,3,4,5,6,7]) &
    expanded60['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid60_707['house_rep'] = (valid60_707['VCF0707'] == 2.0).astype(int)
valid60_707 = construct_pid_vars(valid60_707, 'VCF0301', 'curr')
X60_707 = sm.add_constant(valid60_707[['strong_curr','weak_curr','lean_curr']])
mod60_707 = Probit(valid60_707['house_rep'].astype(float), X60_707).fit(disp=0)
print(f'\nVCF0707-only expanded: N={len(valid60_707)}, LL={mod60_707.llf:.1f}, R2={mod60_707.prsquared:.4f}')
print(f'  Strong={mod60_707.params["strong_curr"]:.3f} Weak={mod60_707.params["weak_curr"]:.3f} Lean={mod60_707.params["lean_curr"]:.3f}')

# === 1976 Panel ===
print('\n\n========== 1976 PANEL PROBIT ==========')
merged76 = p76_76.merge(p76_74[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
print(f'Merged: {len(merged76)}')

# Union vote
merged76['house_vote'] = merged76['VCF0707']
mask_fill76 = merged76['house_vote'].isna() & merged76['VCF0706'].isin([1.0, 2.0])
merged76.loc[mask_fill76, 'house_vote'] = merged76.loc[mask_fill76, 'VCF0706']

valid76 = merged76[
    merged76['house_vote'].isin([1.0, 2.0]) &
    merged76['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged76['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid76['house_rep'] = (valid76['house_vote'] == 2.0).astype(int)
valid76 = construct_pid_vars(valid76, 'VCF0301', 'curr')
valid76 = construct_pid_vars(valid76, 'VCF0301_lag', 'lag')

print(f'Valid N (union): {len(valid76)} (target 682)')
X76 = sm.add_constant(valid76[['strong_curr','weak_curr','lean_curr']])
mod76 = Probit(valid76['house_rep'].astype(float), X76).fit(disp=0)
print(f'Current: LL={mod76.llf:.1f}, R2={mod76.prsquared:.4f}')
print(f'  Strong={mod76.params["strong_curr"]:.3f}({mod76.bse["strong_curr"]:.3f})')
print(f'  Weak={mod76.params["weak_curr"]:.3f}({mod76.bse["weak_curr"]:.3f})')
print(f'  Lean={mod76.params["lean_curr"]:.3f}({mod76.bse["lean_curr"]:.3f})')
print(f'  Int={mod76.params["const"]:.3f}({mod76.bse["const"]:.3f})')
print(f'Targets: N=682, LL=-358.2, R2=0.24, Strong=1.087, Weak=0.624, Lean=0.622, Int=-0.123')

# VCF0707 only for 1976
valid76_707 = merged76[
    merged76['VCF0707'].isin([1.0, 2.0]) &
    merged76['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged76['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid76_707['house_rep'] = (valid76_707['VCF0707'] == 2.0).astype(int)
valid76_707 = construct_pid_vars(valid76_707, 'VCF0301', 'curr')
X76_707 = sm.add_constant(valid76_707[['strong_curr','weak_curr','lean_curr']])
mod76_707 = Probit(valid76_707['house_rep'].astype(float), X76_707).fit(disp=0)
print(f'\nVCF0707-only: N={len(valid76_707)}, LL={mod76_707.llf:.1f}, R2={mod76_707.prsquared:.4f}')
print(f'  Strong={mod76_707.params["strong_curr"]:.3f} Weak={mod76_707.params["weak_curr"]:.3f} Lean={mod76_707.params["lean_curr"]:.3f}')

# === 1992 Panel ===
print('\n\n========== 1992 PANEL PROBIT ==========')
print(f'Total rows: {len(p92)}')
# Already has pid_current, pid_lagged, vote_house
# Check what coding is used
valid92 = p92[
    p92['vote_house'].isin([1, 2]) &
    p92['pid_current'].isin([0, 1, 2, 3, 4, 5, 6])
].copy()
print(f'vote_house in [1,2] & pid_current in [0-6]: {len(valid92)}')

# Check if pid is 0-6 or 1-7
print(f'pid_current range: {p92["pid_current"].min()} to {p92["pid_current"].max()}')
print(f'pid_current values: {sorted(p92["pid_current"].dropna().unique())}')
print(f'pid_lagged values: {sorted(p92["pid_lagged"].dropna().unique())}')

# If 0-6 scale: 0=StrongDem, 1=WeakDem, 2=LeanDem, 3=Ind, 4=LeanRep, 5=WeakRep, 6=StrongRep
# Or if 1-7: same as CDF
# Let's try both
for pid_label, pid_map in [
    ('0-6 scale', {6: 'strong_r', 5: 'weak_r', 4: 'lean_r', 3: 'ind', 2: 'lean_d', 1: 'weak_d', 0: 'strong_d'}),
    ('1-7 scale (CDF)', {7: 'strong_r', 6: 'weak_r', 5: 'lean_r', 4: 'ind', 3: 'lean_d', 2: 'weak_d', 1: 'strong_d'}),
]:
    df = p92.copy()
    if pid_label == '0-6 scale':
        pid_vals = [0, 1, 2, 3, 4, 5, 6]
        strong_r, weak_r, lean_r = 6, 5, 4
        strong_d, weak_d, lean_d = 0, 1, 2
    else:
        pid_vals = [1, 2, 3, 4, 5, 6, 7]
        strong_r, weak_r, lean_r = 7, 6, 5
        strong_d, weak_d, lean_d = 1, 2, 3

    valid = df[
        df['vote_house'].isin([1, 2]) &
        df['pid_current'].isin(pid_vals) &
        df['pid_lagged'].isin(pid_vals)
    ].copy()

    if len(valid) == 0:
        print(f'\n{pid_label}: No valid rows')
        continue

    # Assume vote_house: 1=Dem, 2=Rep
    valid['house_rep'] = (valid['vote_house'] == 2).astype(int)

    valid['strong_curr'] = np.where(valid['pid_current'] == strong_r, 1,
                            np.where(valid['pid_current'] == strong_d, -1, 0))
    valid['weak_curr'] = np.where(valid['pid_current'] == weak_r, 1,
                          np.where(valid['pid_current'] == weak_d, -1, 0))
    valid['lean_curr'] = np.where(valid['pid_current'] == lean_r, 1,
                          np.where(valid['pid_current'] == lean_d, -1, 0))
    valid['strong_lag'] = np.where(valid['pid_lagged'] == strong_r, 1,
                           np.where(valid['pid_lagged'] == strong_d, -1, 0))
    valid['weak_lag'] = np.where(valid['pid_lagged'] == weak_r, 1,
                         np.where(valid['pid_lagged'] == weak_d, -1, 0))
    valid['lean_lag'] = np.where(valid['pid_lagged'] == lean_r, 1,
                         np.where(valid['pid_lagged'] == lean_d, -1, 0))

    print(f'\n{pid_label}: N={len(valid)} (target 760)')
    X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
    mod = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
    print(f'Current: LL={mod.llf:.1f}, R2={mod.prsquared:.4f}')
    print(f'  Strong={mod.params["strong_curr"]:.3f}({mod.bse["strong_curr"]:.3f})')
    print(f'  Weak={mod.params["weak_curr"]:.3f}({mod.bse["weak_curr"]:.3f})')
    print(f'  Lean={mod.params["lean_curr"]:.3f}({mod.bse["lean_curr"]:.3f})')
    print(f'  Int={mod.params["const"]:.3f}({mod.bse["const"]:.3f})')

    Xl = sm.add_constant(valid[['strong_lag','weak_lag','lean_lag']])
    modl = Probit(valid['house_rep'].astype(float), Xl).fit(disp=0)
    print(f'Lagged: LL={modl.llf:.1f}, R2={modl.prsquared:.4f}')
    print(f'  Strong={modl.params["strong_lag"]:.3f}({modl.bse["strong_lag"]:.3f})')
    print(f'  Weak={modl.params["weak_lag"]:.3f}({modl.bse["weak_lag"]:.3f})')
    print(f'  Lean={modl.params["lean_lag"]:.3f}({modl.bse["lean_lag"]:.3f})')
    print(f'  Int={modl.params["const"]:.3f}({modl.bse["const"]:.3f})')

    # Also try vote_house: 1=Rep, 2=Dem (reverse coding)
    valid2 = valid.copy()
    valid2['house_rep'] = (valid2['vote_house'] == 1).astype(int)
    X2 = sm.add_constant(valid2[['strong_curr','weak_curr','lean_curr']])
    mod2 = Probit(valid2['house_rep'].astype(float), X2).fit(disp=0)
    print(f'  [Reverse vote coding] LL={mod2.llf:.1f}, R2={mod2.prsquared:.4f}')
    print(f'    Strong={mod2.params["strong_curr"]:.3f} Weak={mod2.params["weak_curr"]:.3f}')

print(f'\nTargets 1992: N=760, LL=-408.2, R2=0.20, Strong=0.975, Weak=0.627, Lean=0.472, Int=-0.211')
print(f'Targets lagged: LL=-416.2, R2=0.19, Strong=1.061, Weak=0.404, Lean=0.519, Int=-0.168')
