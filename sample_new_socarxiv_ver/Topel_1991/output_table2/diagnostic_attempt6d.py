#!/usr/bin/env python3
"""Test the +-2 SD trimming hypothesis and run regression."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df = pd.read_csv('data/psid_panel.csv')

# Fix education
df['educ_raw'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'educ_raw'] = df.loc[cat_mask, 'education_clean'].map({**EDUC_MAP, 9: np.nan})

def get_fixed_educ(group):
    good = group[group['year'].isin([1975, 1976])]['educ_raw'].dropna()
    if len(good) > 0:
        return good.iloc[0]
    mapped = group['educ_raw'].dropna()
    if len(mapped) > 0:
        modes = mapped.mode()
        return modes.iloc[0] if len(modes) > 0 else mapped.median()
    return np.nan

person_educ = df.groupby('person_id').apply(get_fixed_educ)
df['education_fixed'] = df['person_id'].map(person_educ)
df = df[df['education_fixed'].notna()].copy()

df['experience'] = df['age'] - df['education_fixed'] - 6
df['tenure'] = df['tenure_topel'] - 1

# Within-job
df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp = df.groupby(['person_id', 'job_id'])
df['prev_year'] = grp['year'].shift(1)
df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
df['prev_tenure'] = grp['tenure'].shift(1)
df['prev_experience'] = grp['experience'].shift(1)

within = df[
    (df['prev_year'].notna()) &
    (df['year'] - df['prev_year'] == 1)
].copy()
within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
within['d_exp'] = within['experience'] - within['prev_experience']

# Start with d_exp == 1
base = within[within['d_exp'] == 1].copy()
print(f"After d_exp==1 filter: N={len(base)}")
print(f"Mean d_log_wage: {base['d_log_wage'].mean():.4f}")
print(f"SD d_log_wage: {base['d_log_wage'].std():.4f}")

# +-2 SD non-iterative
mean_dw = base['d_log_wage'].mean()
sd_dw = base['d_log_wage'].std()
lo = mean_dw - 2*sd_dw
hi = mean_dw + 2*sd_dw
print(f"\n2-SD bounds: [{lo:.4f}, {hi:.4f}]")
w = base[(base['d_log_wage'] >= lo) & (base['d_log_wage'] <= hi)]
print(f"After +-2 SD trim: N={len(w)}, SD={w['d_log_wage'].std():.4f}")

# Iterative 2-SD trimming
print("\n--- Iterative +-2 SD trimming ---")
w_iter = base.copy()
for i in range(10):
    m = w_iter['d_log_wage'].mean()
    s = w_iter['d_log_wage'].std()
    lo, hi = m - 2*s, m + 2*s
    w_iter = w_iter[(w_iter['d_log_wage'] >= lo) & (w_iter['d_log_wage'] <= hi)]
    print(f"  Iter {i+1}: N={len(w_iter)}, SD={w_iter['d_log_wage'].std():.4f}, bounds=[{lo:.4f}, {hi:.4f}]")
    if len(w_iter) == len(w_iter):
        # Check convergence
        pass

# What about using the NOMINAL d_log_wage for trimming, not real?
# The paper says "change in log real wage" is the dependent variable
# But the d_log_wage in our data is nominal

# Try different SD multiples to find exact N=8683
print("\n--- Fine-tune SD multiple ---")
for k in np.arange(1.9, 2.2, 0.01):
    m = base['d_log_wage'].mean()
    s = base['d_log_wage'].std()
    lo, hi = m - k*s, m + k*s
    w = base[(base['d_log_wage'] >= lo) & (base['d_log_wage'] <= hi)]
    if abs(len(w) - 8683) < 50:
        print(f"  k={k:.2f}: N={len(w)}, SD={w['d_log_wage'].std():.4f}")

# NOW: Run the full regression with the best trimming approach
print("\n" + "="*80)
print("FULL REGRESSION WITH +-2 SD TRIMMING")
print("="*80)

# Apply +-2 SD trim
m0 = base['d_log_wage'].mean()
s0 = base['d_log_wage'].std()
w = base[(base['d_log_wage'] >= m0 - 2*s0) & (base['d_log_wage'] <= m0 + 2*s0)].copy()
print(f"N={len(w)}, persons={w['person_id'].nunique()}")

t = w['tenure'].values.astype(float)
pt = w['prev_tenure'].values.astype(float)
e = w['experience'].values.astype(float)
pe = w['prev_experience'].values.astype(float)

w['d_tenure'] = t - pt
w['d_tenure_sq'] = t**2 - pt**2
w['d_tenure_cu'] = t**3 - pt**3
w['d_tenure_qu'] = t**4 - pt**4
w['d_exp_sq'] = e**2 - pe**2
w['d_exp_cu'] = e**3 - pe**3
w['d_exp_qu'] = e**4 - pe**4

year_dummies = pd.get_dummies(w['year'], prefix='yr', dtype=float)
yr_cols = sorted(year_dummies.columns.tolist())[1:]
y = w['d_log_wage'].values

def run_ols(y_vals, var_list):
    X_main = w[var_list].copy()
    X = pd.concat([X_main.reset_index(drop=True),
                   year_dummies[yr_cols].reset_index(drop=True)], axis=1)
    valid = np.isfinite(X.values).all(axis=1) & np.isfinite(y_vals)
    model = sm.OLS(y_vals[valid], X.loc[valid].values, hasconst=True).fit()
    return model, var_list + yr_cols

def gc(m, n, v):
    if v in n: return m.params[n.index(v)], m.bse[n.index(v)]
    return None, None

m1, n1 = run_ols(y, ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])
m2, n2 = run_ols(y, ['d_tenure', 'd_tenure_sq', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])
m3, n3 = run_ols(y, ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                      'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])

# Detailed comparison
gt_all = [
    (1, 'd_tenure', 1, 0.1242, 0.0161),
    (1, 'd_exp_sq', 100, -0.6051, 0.1430),
    (1, 'd_exp_cu', 1000, 0.1460, 0.0482),
    (1, 'd_exp_qu', 10000, 0.0131, 0.0054),
    (2, 'd_tenure', 1, 0.1265, 0.0162),
    (2, 'd_tenure_sq', 100, -0.0518, 0.0178),
    (2, 'd_exp_sq', 100, -0.6144, 0.1430),
    (2, 'd_exp_cu', 1000, 0.1620, 0.0485),
    (2, 'd_exp_qu', 10000, 0.0151, 0.0055),
    (3, 'd_tenure', 1, 0.1258, 0.0162),
    (3, 'd_tenure_sq', 100, -0.4592, 0.1080),
    (3, 'd_tenure_cu', 1000, 0.1846, 0.0526),
    (3, 'd_tenure_qu', 10000, -0.0245, 0.0079),
    (3, 'd_exp_sq', 100, -0.4067, 0.1546),
    (3, 'd_exp_cu', 1000, 0.0989, 0.0517),
    (3, 'd_exp_qu', 10000, 0.0089, 0.0058),
]

models = {1: (m1, n1), 2: (m2, n2), 3: (m3, n3)}
print(f"\n{'Model':>5s} {'Variable':>15s} {'Generated':>12s} {'Paper':>12s} {'Diff':>10s} {'|diff|<=.05':>12s}")
print("-"*70)
coef_match = 0
for mod, var, scale, gt_c, gt_s in gt_all:
    m, n = models[mod]
    c, s = gc(m, n, var)
    if c is not None:
        gen_c = c * scale
        diff = gen_c - gt_c
        match = abs(diff) <= 0.05
        if match: coef_match += 1
        print(f"{mod:>5d} {var:>15s} {gen_c:>12.4f} {gt_c:>12.4f} {diff:>10.4f} {'YES' if match else 'NO':>12s}")

print(f"\nCoefficients within 0.05: {coef_match}/{len(gt_all)}")

# SEs
print(f"\n{'Model':>5s} {'Variable':>15s} {'Gen SE':>12s} {'Paper SE':>12s} {'Diff':>10s} {'|diff|<=.02':>12s}")
print("-"*70)
se_match = 0
for mod, var, scale, gt_c, gt_s in gt_all:
    m, n = models[mod]
    c, s = gc(m, n, var)
    if s is not None:
        gen_s = s * scale
        diff = gen_s - gt_s
        match = abs(diff) <= 0.02
        if match: se_match += 1
        print(f"{mod:>5d} {var:>15s} {gen_s:>12.4f} {gt_s:>12.4f} {diff:>10.4f} {'YES' if match else 'NO':>12s}")

print(f"\nSEs within 0.02: {se_match}/{len(gt_all)}")

print(f"\nR^2: M1={m1.rsquared:.4f} (paper: .022), M2={m2.rsquared:.4f} (paper: .023), M3={m3.rsquared:.4f} (paper: .025)")
print(f"SE of reg: M1={np.sqrt(m1.mse_resid):.4f} (paper: .218), M2={np.sqrt(m2.mse_resid):.4f}, M3={np.sqrt(m3.mse_resid):.4f}")

# GNP deflated mean
GNP_DEFLATOR = {
    1967: 100.00, 1968: 104.28, 1969: 109.13, 1970: 113.94, 1971: 118.92,
    1972: 123.16, 1973: 130.27, 1974: 143.08, 1975: 155.56, 1976: 163.42,
    1977: 173.43, 1978: 186.18, 1979: 201.33, 1980: 220.39, 1981: 241.02,
    1982: 255.09, 1983: 264.00
}
w['gnp_cur'] = w['year'].map(GNP_DEFLATOR)
w['gnp_prev'] = (w['year'] - 1).map(GNP_DEFLATOR)
w['d_log_real'] = w['d_log_wage'] - np.log(w['gnp_cur'] / w['gnp_prev'])
print(f"\nMean d_log_real_wage (GNP): {w['d_log_real'].mean():.4f} (paper: .026)")
