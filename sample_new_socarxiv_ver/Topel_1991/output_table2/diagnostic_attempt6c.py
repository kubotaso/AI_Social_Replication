#!/usr/bin/env python3
"""Deep dive on wage variable construction and outlier handling."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}
GNP_DEFLATOR = {
    1967: 100.00, 1968: 104.28, 1969: 109.13, 1970: 113.94, 1971: 118.92,
    1972: 123.16, 1973: 130.27, 1974: 143.08, 1975: 155.56, 1976: 163.42,
    1977: 173.43, 1978: 186.18, 1979: 201.33, 1980: 220.39, 1981: 241.02,
    1982: 255.09, 1983: 264.00
}

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
base = within[within['d_exp'] == 1].copy()

# What's the SD of d_log_wage?
print(f"SD of d_log_wage (all d_exp==1): {base['d_log_wage'].std():.4f}")
print(f"Mean: {base['d_log_wage'].mean():.4f}")

# Paper says SE of regression = 0.218
# Our SD of d_log_wage is 0.279
# The SD of d_log_wage should be approximately equal to the SE of regression
# if the model has low R^2 (as it does, R^2 ~ 0.022)
# SE = SD * sqrt(1 - R^2) approx = SD when R^2 is small
# So if paper's SE = 0.218, then paper's SD of d_log_wage ~ 0.218/sqrt(1-0.022) ~ 0.220

# Our SD is 0.279 vs paper's ~0.220 -- that's 27% higher
# This suggests either:
# 1. Different outlier handling
# 2. Different wage variable construction
# 3. Additional sample restrictions we're missing

# Let's look at the distribution of d_log_wage
print(f"\nd_log_wage percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {base['d_log_wage'].quantile(p/100):.4f}")

# Winsorize at 1% and 5%
for p in [1, 2, 5]:
    lo, hi = base['d_log_wage'].quantile(p/100), base['d_log_wage'].quantile(1-p/100)
    w = base[(base['d_log_wage'] >= lo) & (base['d_log_wage'] <= hi)]
    print(f"  Winsorize at {p}%: SD={w['d_log_wage'].std():.4f}, N={len(w)}")

# Check: maybe the issue is hourly_wage computation
# The paper says: "I measure wages as the ratio of annual earnings to annual hours"
# Let's check the raw wage data
print(f"\n--- Raw wage data ---")
print(f"hourly_wage: mean={df['hourly_wage'].mean():.2f}, std={df['hourly_wage'].std():.2f}")
print(f"log_hourly_wage: mean={df['log_hourly_wage'].mean():.4f}, std={df['log_hourly_wage'].std():.4f}")
print(f"labor_inc: mean={df['labor_inc'].mean():.0f}")
print(f"hours: mean={df['hours'].mean():.0f}")

# Recompute hourly wage from labor_inc / hours
df['hw_recomputed'] = df['labor_inc'] / df['hours']
df['lhw_recomputed'] = np.log(df['hw_recomputed'])
print(f"\nRecomputed hw: mean={df['hw_recomputed'].mean():.2f}")
print(f"Recomputed lhw: mean={df['lhw_recomputed'].mean():.4f}")
print(f"Max diff from stored: {(df['log_hourly_wage'] - df['lhw_recomputed']).abs().max():.6f}")

# Check if there are extreme hourly wages
print(f"\nhourly_wage percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {df['hourly_wage'].quantile(p/100):.2f}")

# Maybe Topel trims extreme wages?
# Topel says "I deleted obvious data errors"
# Let's try trimming at reasonable hourly wage levels
print(f"\n--- Wage-level trimming effects ---")
for lo_hw, hi_hw in [(1, 100), (1.5, 75), (2, 50), (1, 250)]:
    mask = (df['hourly_wage'] >= lo_hw) & (df['hourly_wage'] <= hi_hw)
    df_t = df[mask].copy()
    # Redo within-job
    df_t = df_t.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
    grp_t = df_t.groupby(['person_id', 'job_id'])
    df_t['prev_year'] = grp_t['year'].shift(1)
    df_t['prev_log_wage'] = grp_t['log_hourly_wage'].shift(1)
    df_t['prev_tenure'] = grp_t['tenure'].shift(1)
    df_t['prev_experience'] = grp_t['experience'].shift(1)
    w_t = df_t[
        (df_t['prev_year'].notna()) &
        (df_t['year'] - df_t['prev_year'] == 1)
    ].copy()
    w_t['d_log_wage'] = w_t['log_hourly_wage'] - w_t['prev_log_wage']
    w_t['d_exp'] = w_t['experience'] - w_t['prev_experience']
    w_t = w_t[w_t['d_exp'] == 1]
    print(f"  hw [{lo_hw}, {hi_hw}]: N={len(w_t)}, SD(d_lw)={w_t['d_log_wage'].std():.4f}")

# Maybe: trim d_log_wage at a specific SD multiple
print(f"\n--- SD-based trimming ---")
for k in [2, 2.5, 3, 4]:
    mean_dw = base['d_log_wage'].mean()
    sd_dw = base['d_log_wage'].std()
    w = base[base['d_log_wage'].between(mean_dw - k*sd_dw, mean_dw + k*sd_dw)]
    print(f"  +-{k} SD: N={len(w)}, SD={w['d_log_wage'].std():.4f}")

# Most promising: try +-2 SD which would be approximately +-0.56
# Actually, the paper may use the LEVEL wage for outlier detection
# and the CHANGE for regression. Let's try both.

# What if we trim based on wage LEVELS at each year?
# Drop obs where hourly wage is below $1/hour or above $100/hour (in nominal terms)
print(f"\n--- Trimming on LEVEL wage + d_log_wage outlier removal ---")
for lo_hw in [0.5, 1.0, 1.5, 2.0]:
    for hi_hw in [50, 100, 200]:
        mask = (df['hourly_wage'] >= lo_hw) & (df['hourly_wage'] <= hi_hw)
        df_t = df[mask].copy()
        df_t = df_t.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
        grp_t = df_t.groupby(['person_id', 'job_id'])
        df_t['prev_year'] = grp_t['year'].shift(1)
        df_t['prev_log_wage'] = grp_t['log_hourly_wage'].shift(1)
        df_t['prev_tenure'] = grp_t['tenure'].shift(1)
        df_t['prev_experience'] = grp_t['experience'].shift(1)
        w_t = df_t[
            (df_t['prev_year'].notna()) &
            (df_t['year'] - df_t['prev_year'] == 1)
        ].copy()
        w_t['d_log_wage'] = w_t['log_hourly_wage'] - w_t['prev_log_wage']
        w_t['d_exp'] = w_t['experience'] - w_t['prev_experience']
        w_t = w_t[w_t['d_exp'] == 1]
        w_t = w_t[w_t['d_log_wage'].between(-2, 2)]
        if len(w_t) > 8000 and len(w_t) < 9500:
            print(f"  hw [{lo_hw}, {hi_hw}]: N={len(w_t)}, SD(d_lw)={w_t['d_log_wage'].std():.4f}, persons={w_t['person_id'].nunique()}")

# What about hours restriction?
print(f"\n--- Hours restriction ---")
print(f"Hours distribution:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {df['hours'].quantile(p/100):.0f}")

# Topel likely requires reasonable hours (e.g., 250+)
for min_hours in [250, 500, 750, 1000]:
    mask = df['hours'] >= min_hours
    df_t = df[mask].copy()
    df_t = df_t.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
    grp_t = df_t.groupby(['person_id', 'job_id'])
    df_t['prev_year'] = grp_t['year'].shift(1)
    df_t['prev_log_wage'] = grp_t['log_hourly_wage'].shift(1)
    df_t['prev_tenure'] = grp_t['tenure'].shift(1)
    df_t['prev_experience'] = grp_t['experience'].shift(1)
    w_t = df_t[
        (df_t['prev_year'].notna()) &
        (df_t['year'] - df_t['prev_year'] == 1)
    ].copy()
    w_t['d_log_wage'] = w_t['log_hourly_wage'] - w_t['prev_log_wage']
    w_t['d_exp'] = w_t['experience'] - w_t['prev_experience']
    w_t = w_t[w_t['d_exp'] == 1]
    w_t = w_t[w_t['d_log_wage'].between(-2, 2)]
    print(f"  min_hours={min_hours}: N={len(w_t)}, SD(d_lw)={w_t['d_log_wage'].std():.4f}, persons={w_t['person_id'].nunique()}")

# What about requiring BOTH current and previous year to have reasonable hours?
print(f"\n--- Both-year hours restriction ---")
df_t = df.copy()
df_t = df_t.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp_t = df_t.groupby(['person_id', 'job_id'])
df_t['prev_year'] = grp_t['year'].shift(1)
df_t['prev_hours'] = grp_t['hours'].shift(1)
df_t['prev_log_wage'] = grp_t['log_hourly_wage'].shift(1)
df_t['prev_tenure'] = grp_t['tenure'].shift(1)
df_t['prev_experience'] = grp_t['experience'].shift(1)
w_all = df_t[
    (df_t['prev_year'].notna()) &
    (df_t['year'] - df_t['prev_year'] == 1)
].copy()
w_all['d_log_wage'] = w_all['log_hourly_wage'] - w_all['prev_log_wage']
w_all['d_exp'] = w_all['experience'] - w_all['prev_experience']

for min_h in [250, 500, 750, 1000]:
    w_t = w_all[(w_all['hours'] >= min_h) & (w_all['prev_hours'] >= min_h)]
    w_t = w_t[w_t['d_exp'] == 1]
    w_t = w_t[w_t['d_log_wage'].between(-2, 2)]
    print(f"  both_hours >= {min_h}: N={len(w_t)}, SD(d_lw)={w_t['d_log_wage'].std():.4f}, persons={w_t['person_id'].nunique()}")
