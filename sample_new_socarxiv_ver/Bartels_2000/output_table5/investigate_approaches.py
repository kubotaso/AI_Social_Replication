"""Test different approaches for each panel to find best coefficient matches."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

BASE = '/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'

GROUND_TRUTH = {
    '1960': {
        'N': 911,
        'current': {'strong': (1.358, 0.094), 'weak': (1.028, 0.083), 'lean': (0.855, 0.131), 'intercept': (0.035, 0.053), 'llf': -372.7, 'r2': 0.41},
        'lagged': {'strong': (1.363, 0.092), 'weak': (0.842, 0.078), 'lean': (0.564, 0.125), 'intercept': (0.068, 0.051), 'llf': -403.9, 'r2': 0.36},
        'iv': {'strong': (1.715, 0.173), 'weak': (0.728, 0.239), 'lean': (1.081, 0.696), 'intercept': (0.032, 0.057), 'llf': -403.9, 'r2': 0.36}
    },
    '1976': {
        'N': 682,
        'current': {'strong': (1.087, 0.105), 'weak': (0.624, 0.086), 'lean': (0.622, 0.110), 'intercept': (-0.123, 0.054), 'llf': -358.2, 'r2': 0.24},
        'lagged': {'strong': (0.966, 0.104), 'weak': (0.738, 0.089), 'lean': (0.486, 0.109), 'intercept': (-0.063, 0.053), 'llf': -371.3, 'r2': 0.21},
        'iv': {'strong': (1.123, 0.178), 'weak': (0.745, 0.251), 'lean': (0.725, 0.438), 'intercept': (-0.102, 0.055), 'llf': -371.3, 'r2': 0.21}
    },
    '1992': {
        'N': 760,
        'current': {'strong': (0.975, 0.094), 'weak': (0.627, 0.084), 'lean': (0.472, 0.098), 'intercept': (-0.211, 0.051), 'llf': -408.2, 'r2': 0.20},
        'lagged': {'strong': (1.061, 0.100), 'weak': (0.404, 0.077), 'lean': (0.519, 0.101), 'intercept': (-0.168, 0.051), 'llf': -416.2, 'r2': 0.19},
        'iv': {'strong': (1.516, 0.180), 'weak': (-0.225, 0.268), 'lean': (1.824, 0.513), 'intercept': (-0.125, 0.053), 'llf': -416.2, 'r2': 0.19}
    }
}

def construct_pid_dummies(pid_series):
    strong = pd.Series(0.0, index=pid_series.index)
    weak = pd.Series(0.0, index=pid_series.index)
    lean = pd.Series(0.0, index=pid_series.index)
    strong[pid_series == 7] = 1; strong[pid_series == 1] = -1
    weak[pid_series == 6] = 1; weak[pid_series == 2] = -1
    lean[pid_series == 5] = 1; lean[pid_series == 3] = -1
    return strong, weak, lean

def run_probit(y, X):
    X = sm.add_constant(X)
    return Probit(y, X).fit(disp=0, maxiter=1000)

def run_iv_probit(y, X_endog, Z_instruments):
    Z_with_const = sm.add_constant(Z_instruments)
    X_hat = pd.DataFrame(index=X_endog.index)
    for col in X_endog.columns:
        ols_result = sm.OLS(X_endog[col], Z_with_const).fit()
        X_hat[col] = ols_result.predict()
    X_hat_with_const = sm.add_constant(X_hat)
    return Probit(y, X_hat_with_const).fit(disp=0, maxiter=1000)

def coef_distance(model, gt_model):
    """Total absolute coefficient distance from ground truth."""
    params = model.params
    total_dist = 0
    for var_key in ['strong', 'weak', 'lean', 'intercept']:
        gt_coef = gt_model[var_key][0]
        if var_key == 'intercept':
            gen_coef = params['const']
        else:
            name = [n for n in params.index if var_key in n.lower()][0]
            gen_coef = params[name]
        total_dist += abs(gen_coef - gt_coef)
    return total_dist

# Load CDF and panel files
cdf = pd.read_csv(f'{BASE}/anes_cumulative.csv', low_memory=False)
df60_panel = pd.read_csv(f'{BASE}/panel_1960.csv')
df76_panel = pd.read_csv(f'{BASE}/panel_1976.csv')
df92_panel = pd.read_csv(f'{BASE}/panel_1992.csv')

print("=" * 70)
print("APPROACH COMPARISON FOR EACH PANEL")
print("=" * 70)

# === 1960 Approaches ===
print("\n=== 1960 PANEL ===")
approaches_60 = {}

# Approach A: Panel file, VCF0707 only, no expansion
mask = df60_panel['VCF0707'].isin([1.0, 2.0]) & df60_panel['VCF0301'].isin(range(1,8)) & df60_panel['VCF0301_lagged'].isin(range(1,8))
d = df60_panel[mask].copy()
vote = (d['VCF0707'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1960']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1960']['lagged'])
approaches_60['A_panel_707'] = {'N': len(d), 'mc': mc, 'ml': ml, 'mi': mi, 'cd': cd, 'ld': ld}
print(f"A: Panel 707 only: N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# Approach B: Panel file, VCF0707+VCF0706 union, no expansion
d = df60_panel.copy()
d['house_vote'] = d['VCF0707']
m = d['house_vote'].isna() & d['VCF0706'].isin([1.0, 2.0])
d.loc[m, 'house_vote'] = d.loc[m, 'VCF0706']
mask = d['house_vote'].isin([1.0, 2.0]) & d['VCF0301'].isin(range(1,8)) & d['VCF0301_lagged'].isin(range(1,8))
d = d[mask].copy()
vote = (d['house_vote'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1960']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1960']['lagged'])
approaches_60['B_panel_union'] = {'N': len(d), 'mc': mc, 'ml': ml, 'mi': mi, 'cd': cd, 'ld': ld}
print(f"B: Panel union:    N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# Approach C: Panel file, VCF0707+VCF0706 union, expanded by VCF0009x
d = df60_panel.copy()
d['house_vote'] = d['VCF0707']
m = d['house_vote'].isna() & d['VCF0706'].isin([1.0, 2.0])
d.loc[m, 'house_vote'] = d.loc[m, 'VCF0706']
# Expand by weights
wt = d['VCF0009x'].fillna(1.0).astype(int)
d = d.loc[d.index.repeat(wt)].reset_index(drop=True)
mask = d['house_vote'].isin([1.0, 2.0]) & d['VCF0301'].isin(range(1,8)) & d['VCF0301_lagged'].isin(range(1,8))
d = d[mask].copy()
vote = (d['house_vote'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1960']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1960']['lagged'])
approaches_60['C_panel_union_exp'] = {'N': len(d), 'mc': mc, 'ml': ml, 'mi': mi, 'cd': cd, 'ld': ld}
print(f"C: Panel union+exp: N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# Approach D: CDF, VCF0707 only, no expansion
cdf_curr = cdf[cdf['VCF0004'] == 1960].copy()
cdf_lag = cdf[cdf['VCF0004'] == 1958].copy()
panel = cdf_curr[cdf_curr['VCF0006a'] < 19600000].copy()
merged = panel.merge(cdf_lag[['VCF0006a', 'VCF0301']], on='VCF0006a', suffixes=('', '_lag'))
mask = merged['VCF0707'].isin([1.0, 2.0]) & merged['VCF0301'].isin(range(1,8)) & merged['VCF0301_lag'].isin(range(1,8))
d = merged[mask].copy()
vote = (d['VCF0707'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lag'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1960']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1960']['lagged'])
approaches_60['D_cdf_707'] = {'N': len(d), 'mc': mc, 'ml': ml, 'mi': mi, 'cd': cd, 'ld': ld}
print(f"D: CDF 707 only:   N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# Approach E: CDF, VCF0707+VCF0706 union, expanded
cdf_curr = cdf[cdf['VCF0004'] == 1960].copy()
cdf_lag = cdf[cdf['VCF0004'] == 1958].copy()
panel = cdf_curr[cdf_curr['VCF0006a'] < 19600000].copy()
merged = panel.merge(cdf_lag[['VCF0006a', 'VCF0301']], on='VCF0006a', suffixes=('', '_lag'))
wt = merged['VCF0009x'].fillna(1.0).astype(int)
merged = merged.loc[merged.index.repeat(wt)].reset_index(drop=True)
merged['house_vote'] = merged['VCF0707']
m = merged['house_vote'].isna() & merged['VCF0706'].isin([1.0, 2.0])
merged.loc[m, 'house_vote'] = merged.loc[m, 'VCF0706']
mask = merged['house_vote'].isin([1.0, 2.0]) & merged['VCF0301'].isin(range(1,8)) & merged['VCF0301_lag'].isin(range(1,8))
d = merged[mask].copy()
vote = (d['house_vote'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lag'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1960']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1960']['lagged'])
approaches_60['E_cdf_union_exp'] = {'N': len(d), 'mc': mc, 'ml': ml, 'mi': mi, 'cd': cd, 'ld': ld}
print(f"E: CDF union+exp:  N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# Approach F: Panel file, VCF0706 only (pre-election intent), no expansion
mask = df60_panel['VCF0706'].isin([1.0, 2.0]) & df60_panel['VCF0301'].isin(range(1,8)) & df60_panel['VCF0301_lagged'].isin(range(1,8))
d = df60_panel[mask].copy()
vote = (d['VCF0706'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1960']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1960']['lagged'])
approaches_60['F_panel_706'] = {'N': len(d), 'mc': mc, 'ml': ml, 'mi': mi, 'cd': cd, 'ld': ld}
print(f"F: Panel 706 only:  N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# Approach G: Panel file, VCF0706 only, expanded
d = df60_panel.copy()
wt = d['VCF0009x'].fillna(1.0).astype(int)
d = d.loc[d.index.repeat(wt)].reset_index(drop=True)
mask = d['VCF0706'].isin([1.0, 2.0]) & d['VCF0301'].isin(range(1,8)) & d['VCF0301_lagged'].isin(range(1,8))
d = d[mask].copy()
vote = (d['VCF0706'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1960']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1960']['lagged'])
approaches_60['G_panel_706_exp'] = {'N': len(d), 'mc': mc, 'ml': ml, 'mi': mi, 'cd': cd, 'ld': ld}
print(f"G: Panel 706+exp:   N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# === 1976 Approaches ===
print("\n=== 1976 PANEL ===")

# Approach A: Panel file, VCF0707 only
mask = df76_panel['VCF0707'].isin([1.0, 2.0]) & df76_panel['VCF0301'].isin(range(1,8)) & df76_panel['VCF0301_lagged'].isin(range(1,8))
d = df76_panel[mask].copy()
vote = (d['VCF0707'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1976']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1976']['lagged'])
print(f"A: Panel 707 only: N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# Approach B: Panel file, union
d = df76_panel.copy()
d['house_vote'] = d['VCF0707']
m = d['house_vote'].isna() & d['VCF0706'].isin([1.0, 2.0])
d.loc[m, 'house_vote'] = d.loc[m, 'VCF0706']
mask = d['house_vote'].isin([1.0, 2.0]) & d['VCF0301'].isin(range(1,8)) & d['VCF0301_lagged'].isin(range(1,8))
d = d[mask].copy()
vote = (d['house_vote'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1976']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1976']['lagged'])
print(f"B: Panel union:    N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# CDF union approach
cdf_curr = cdf[cdf['VCF0004'] == 1976].copy()
cdf_lag = cdf[cdf['VCF0004'] == 1974].copy()
panel = cdf_curr[cdf_curr['VCF0006a'] < 19760000].copy()
merged = panel.merge(cdf_lag[['VCF0006a', 'VCF0301']], on='VCF0006a', suffixes=('', '_lag'))
merged['house_vote'] = merged['VCF0707']
m = merged['house_vote'].isna() & merged['VCF0706'].isin([1.0, 2.0])
merged.loc[m, 'house_vote'] = merged.loc[m, 'VCF0706']
mask = merged['house_vote'].isin([1.0, 2.0]) & merged['VCF0301'].isin(range(1,8)) & merged['VCF0301_lag'].isin(range(1,8))
d = merged[mask].copy()
vote = (d['house_vote'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lag'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1976']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1976']['lagged'])
print(f"C: CDF union:      N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# Approach D: Panel file, VCF0706 only
mask = df76_panel['VCF0706'].isin([1.0, 2.0]) & df76_panel['VCF0301'].isin(range(1,8)) & df76_panel['VCF0301_lagged'].isin(range(1,8))
d = df76_panel[mask].copy()
vote = (d['VCF0706'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1976']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1976']['lagged'])
print(f"D: Panel 706 only:  N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# === 1992 Approaches ===
print("\n=== 1992 PANEL ===")

# A: Panel file, vote_house, standard coding (2=Rep)
mask = df92_panel['vote_house'].isin([1, 2]) & df92_panel['pid_current'].isin(range(1,8)) & df92_panel['pid_lagged'].isin(range(1,8))
d = df92_panel[mask].copy()
vote = (d['vote_house'] == 2).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['pid_current'])
s_l, w_l, l_l = construct_pid_dummies(d['pid_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1992']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1992']['lagged'])
print(f"A: Panel vote_house: N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# B: CDF, VCF0707 only
cdf_curr = cdf[cdf['VCF0004'] == 1992].copy()
cdf_lag = cdf[cdf['VCF0004'] == 1990].copy()
panel = cdf_curr[cdf_curr['VCF0006a'] < 19920000].copy()
merged = panel.merge(cdf_lag[['VCF0006a', 'VCF0301']], on='VCF0006a', suffixes=('', '_lag'))
mask = merged['VCF0707'].isin([1.0, 2.0]) & merged['VCF0301'].isin(range(1,8)) & merged['VCF0301_lag'].isin(range(1,8))
d = merged[mask].copy()
vote = (d['VCF0707'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lag'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
mc = run_probit(vote, X_c); ml = run_probit(vote, X_l); mi = run_iv_probit(vote, X_c, X_l)
cd = coef_distance(mc, GROUND_TRUTH['1992']['current'])
ld = coef_distance(ml, GROUND_TRUTH['1992']['lagged'])
print(f"B: CDF 707:        N={len(d)}, curr_dist={cd:.3f}, lag_dist={ld:.3f}")

# Print detailed comparison for best approaches
print("\n" + "=" * 70)
print("DETAILED BEST APPROACH FOR EACH PANEL")
print("=" * 70)

# For 1960, approaches A (panel 707 only) and F (panel 706 only) - check which has closer coefficients
best_60 = min(approaches_60.items(), key=lambda x: x[1]['cd'] + x[1]['ld'])
print(f"\n1960 Best: {best_60[0]} (total dist = {best_60[1]['cd'] + best_60[1]['ld']:.3f})")
m = best_60[1]
print(f"  N={m['N']}")
print(f"  Current: Strong={m['mc'].params['Strong']:.3f} Weak={m['mc'].params['Weak']:.3f} Lean={m['mc'].params['Lean']:.3f} Int={m['mc'].params['const']:.3f} LL={m['mc'].llf:.1f}")
print(f"  Target:  Strong=1.358 Weak=1.028 Lean=0.855 Int=0.035 LL=-372.7")
print(f"  Lagged: Strong={m['ml'].params['Strong']:.3f} Weak={m['ml'].params['Weak']:.3f} Lean={m['ml'].params['Lean']:.3f} Int={m['ml'].params['const']:.3f} LL={m['ml'].llf:.1f}")
print(f"  Target:  Strong=1.363 Weak=0.842 Lean=0.564 Int=0.068 LL=-403.9")
