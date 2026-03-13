#!/usr/bin/env python3
"""
Sensitivity analysis: what adjustments bring us closer to paper values?
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df = pd.read_csv('data/psid_panel.csv')

# Education recode
df['education_years'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(
    {**EDUC_MAP, 9: np.nan}
)
df = df[df['education_years'].notna()].copy()
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'] - 1

# Prepare within-job differences
df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp = df.groupby(['person_id', 'job_id'])
df['prev_year'] = grp['year'].shift(1)
df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
df['prev_tenure'] = grp['tenure'].shift(1)
df['prev_experience'] = grp['experience'].shift(1)

within = df[
    (df['prev_year'].notna()) &
    (df['year'] - df['prev_year'] == 1) &
    df['experience'].notna() &
    df['prev_experience'].notna() &
    (df['experience'] >= 1)
].copy()
within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']

def run_model1(data):
    """Run Model 1 and return key stats."""
    d = data.copy()
    t = d['tenure'].values.astype(float)
    pt = d['prev_tenure'].values.astype(float)
    e = d['experience'].values.astype(float)
    pe = d['prev_experience'].values.astype(float)

    d['d_tenure'] = t - pt
    d['d_exp_sq'] = e**2 - pe**2
    d['d_exp_cu'] = e**3 - pe**3
    d['d_exp_qu'] = e**4 - pe**4

    yr_dum = pd.get_dummies(d['year'], prefix='yr', dtype=float)
    yr_cols = sorted(yr_dum.columns.tolist())[1:]

    X = pd.concat([d[['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']].reset_index(drop=True),
                    yr_dum[yr_cols].reset_index(drop=True)], axis=1)
    y = d['d_log_wage'].values

    valid = np.isfinite(X.values).all(axis=1) & np.isfinite(y)
    m = sm.OLS(y[valid], X.loc[valid].values, hasconst=True).fit()

    dt_coef = m.params[0]  # d_tenure coefficient
    return len(d), dt_coef, np.sqrt(m.mse_resid), m.rsquared

# Baseline
N0, c0, se0, r20 = run_model1(within[within['d_log_wage'].between(-2, 2)])
print(f"Baseline: N={N0}, Delta_T={c0:.4f}, SE={se0:.4f}, R2={r20:.4f}")
print(f"Paper:    N=8683, Delta_T=0.1242, SE=0.218, R2=0.022")
print()

# Test 1: Tighter outlier trim
for lim in [1.5, 1.0, 0.75, 0.5]:
    sub = within[within['d_log_wage'].between(-lim, lim)]
    N, c, se, r2 = run_model1(sub)
    print(f"Trim +-{lim}: N={N}, Delta_T={c:.4f}, SE={se:.4f}, R2={r2:.4f}")

print()

# Test 2: Restrict experience range
for exp_max in [50, 40, 35, 30]:
    sub = within[(within['d_log_wage'].between(-2, 2)) & (within['experience'] <= exp_max)]
    N, c, se, r2 = run_model1(sub)
    print(f"Exp <= {exp_max}: N={N}, Delta_T={c:.4f}, SE={se:.4f}, R2={r2:.4f}")

print()

# Test 3: Restrict tenure range
for t_max in [20, 15, 12, 10]:
    sub = within[(within['d_log_wage'].between(-2, 2)) & (within['tenure'] <= t_max)]
    N, c, se, r2 = run_model1(sub)
    print(f"Tenure <= {t_max}: N={N}, Delta_T={c:.4f}, SE={se:.4f}, R2={r2:.4f}")

print()

# Test 4: Minimum hours worked
for min_hours in [250, 500, 1000, 1500, 2000]:
    sub = within[(within['d_log_wage'].between(-2, 2)) & (within['hours'] >= min_hours)]
    N, c, se, r2 = run_model1(sub)
    print(f"Hours >= {min_hours}: N={N}, Delta_T={c:.4f}, SE={se:.4f}, R2={r2:.4f}")

print()

# Test 5: Wage range filter
for lo, hi in [(1, 100), (1.5, 50), (2, 40), (2, 30)]:
    sub = within[(within['d_log_wage'].between(-2, 2)) &
                 (within['hourly_wage'] >= lo) & (within['hourly_wage'] <= hi)]
    N, c, se, r2 = run_model1(sub)
    print(f"Wage [{lo},{hi}]: N={N}, Delta_T={c:.4f}, SE={se:.4f}, R2={r2:.4f}")

print()

# Test 6: What if we restrict to persons observed for at least X years?
for min_years in [2, 3, 5, 8, 10]:
    person_years = within.groupby('person_id')['year'].count()
    long_pids = person_years[person_years >= min_years].index
    sub = within[(within['d_log_wage'].between(-2, 2)) & within['person_id'].isin(long_pids)]
    N, c, se, r2 = run_model1(sub)
    n_persons = sub['person_id'].nunique()
    print(f"Min {min_years} obs/person: N={N}, persons={n_persons}, Delta_T={c:.4f}, SE={se:.4f}, R2={r2:.4f}")

print()

# Test 7: What if we drop first observation of each job (tenure=0)?
sub_no_first = within[(within['d_log_wage'].between(-2, 2)) & (within['tenure'] >= 1)]
N, c, se, r2 = run_model1(sub_no_first)
print(f"Tenure >= 1: N={N}, Delta_T={c:.4f}, SE={se:.4f}, R2={r2:.4f}")

# Test 8: Check what happens with HC1 (robust) standard errors
print()
sub = within[within['d_log_wage'].between(-2, 2)].copy()
t = sub['tenure'].values.astype(float)
pt = sub['prev_tenure'].values.astype(float)
e = sub['experience'].values.astype(float)
pe = sub['prev_experience'].values.astype(float)
sub['d_tenure'] = t - pt
sub['d_exp_sq'] = e**2 - pe**2
sub['d_exp_cu'] = e**3 - pe**3
sub['d_exp_qu'] = e**4 - pe**4
yr_dum = pd.get_dummies(sub['year'], prefix='yr', dtype=float)
yr_cols = sorted(yr_dum.columns.tolist())[1:]
X = pd.concat([sub[['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']].reset_index(drop=True),
                yr_dum[yr_cols].reset_index(drop=True)], axis=1)
y = sub['d_log_wage'].values
valid = np.isfinite(X.values).all(axis=1) & np.isfinite(y)
m = sm.OLS(y[valid], X.loc[valid].values, hasconst=True).fit(cov_type='HC1')
print(f"HC1 robust SE for Delta Tenure: {m.bse[0]:.4f} (vs non-robust: {sm.OLS(y[valid], X.loc[valid].values, hasconst=True).fit().bse[0]:.4f})")
