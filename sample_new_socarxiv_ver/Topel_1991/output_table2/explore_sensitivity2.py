#!/usr/bin/env python3
"""Test combinations to match paper's N and SE."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df = pd.read_csv('data/psid_panel.csv')
df['education_years'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(
    {**EDUC_MAP, 9: np.nan}
)
df = df[df['education_years'].notna()].copy()
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'] - 1

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

def test_model1(data):
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
    return len(d), m.params[0], np.sqrt(m.mse_resid), m.rsquared

# Combined filters
print("=== COMBINED FILTERS ===")
print(f"{'Filter':60s} {'N':>6s} {'DT':>8s} {'SE':>8s} {'R2':>8s}")
print("-"*90)

# Target: N~8683, SE~0.218, DT~0.1242, R2~0.022

# Trim + hours + wage filter
for trim in [0.75, 0.80, 0.85, 0.90]:
    for min_h in [250, 500, 750, 1000]:
        for wlo, whi in [(1.5, 50), (2, 40)]:
            sub = within[(within['d_log_wage'].between(-trim, trim)) &
                         (within['hours'] >= min_h) &
                         (within['hourly_wage'] >= wlo) &
                         (within['hourly_wage'] <= whi)]
            if 8000 < len(sub) < 9500:
                N, c, se, r2 = test_model1(sub)
                if abs(se - 0.218) < 0.02:
                    desc = f"trim={trim}, hours>={min_h}, wage=[{wlo},{whi}]"
                    print(f"{desc:60s} {N:>6d} {c:>8.4f} {se:>8.4f} {r2:>8.4f}")

# Trim + experience filter
for trim in [0.65, 0.70, 0.75, 0.80]:
    for exp_max in [35, 38, 40, 42]:
        sub = within[(within['d_log_wage'].between(-trim, trim)) &
                     (within['experience'] <= exp_max)]
        if 8000 < len(sub) < 9500:
            N, c, se, r2 = test_model1(sub)
            if abs(se - 0.218) < 0.02:
                desc = f"trim={trim}, exp<={exp_max}"
                print(f"{desc:60s} {N:>6d} {c:>8.4f} {se:>8.4f} {r2:>8.4f}")

# Trim + min person observations
for trim in [0.65, 0.70, 0.75, 0.80, 0.85]:
    for min_obs in [2, 3, 4]:
        sub_base = within[within['d_log_wage'].between(-trim, trim)]
        py = sub_base.groupby('person_id')['year'].count()
        long_pids = py[py >= min_obs].index
        sub = sub_base[sub_base['person_id'].isin(long_pids)]
        if 8000 < len(sub) < 9500:
            N, c, se, r2 = test_model1(sub)
            if abs(se - 0.218) < 0.02:
                desc = f"trim={trim}, min_obs={min_obs}"
                print(f"{desc:60s} {N:>6d} {c:>8.4f} {se:>8.4f} {r2:>8.4f}")

# Trim + min person + hours
for trim in [0.75, 0.80]:
    for min_obs in [3, 4]:
        for min_h in [500, 750]:
            sub_base = within[(within['d_log_wage'].between(-trim, trim)) &
                              (within['hours'] >= min_h)]
            py = sub_base.groupby('person_id')['year'].count()
            long_pids = py[py >= min_obs].index
            sub = sub_base[sub_base['person_id'].isin(long_pids)]
            if 7500 < len(sub) < 9500:
                N, c, se, r2 = test_model1(sub)
                if abs(se - 0.218) < 0.02:
                    desc = f"trim={trim}, min_obs={min_obs}, hours>={min_h}"
                    n_p = sub['person_id'].nunique()
                    print(f"{desc:60s} {N:>6d} {c:>8.4f} {se:>8.4f} {r2:>8.4f} persons={n_p}")
