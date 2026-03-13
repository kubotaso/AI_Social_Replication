#!/usr/bin/env python3
"""Investigate tenure distribution and polynomial issues."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

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

base = within[within['d_exp'] == 1].copy()

# Apply 2-SD trim
m0, s0 = base['d_log_wage'].mean(), base['d_log_wage'].std()
w = base[(base['d_log_wage'] >= m0 - 2*s0) & (base['d_log_wage'] <= m0 + 2*s0)].copy()
print(f"N = {len(w)}")

# Tenure distribution
print(f"\nTenure distribution:")
print(w['tenure'].describe())
print(f"\nTenure value counts (top 20):")
print(w['tenure'].value_counts().sort_index().head(20))

# Note: tenure in the differenced data refers to the CURRENT year's tenure
# d_tenure_sq = tenure^2 - prev_tenure^2 = (tenure-1+1)^2 - tenure-1^2 = 2*tenure - 1
# So d_tenure_sq = 2*tenure - 1 when d_tenure = 1

# Wait -- actually tenure starts at 0 in our coding
# prev_tenure = tenure - 1 (since d_tenure = 1)
# d_tenure_sq = tenure^2 - (tenure-1)^2 = 2*tenure - 1
# d_tenure_cu = tenure^3 - (tenure-1)^3 = 3*tenure^2 - 3*tenure + 1
# d_tenure_qu = tenure^4 - (tenure-1)^4 = 4*tenure^3 - 6*tenure^2 + 4*tenure - 1

# These are polynomial functions of tenure, so their variation depends on tenure range
# The paper's Table A1 says mean tenure = 9.365 for within-job observations

print(f"\nMean tenure: {w['tenure'].mean():.2f} (paper: ~9.365 from Table A1)")
print(f"Mean experience: {w['experience'].mean():.1f}")
print(f"Mean prev_tenure: {w['prev_tenure'].mean():.2f}")

# Check: is our d_tenure always exactly 1?
print(f"\nd_tenure distribution:")
d_ten = w['tenure'] - w['prev_tenure']
print(d_ten.value_counts())

# The high SEs in Model 3 for tenure polynomials might be because
# d_tenure_cu and d_tenure_qu have too much collinearity
# Let's check the correlation matrix
t = w['tenure'].values.astype(float)
pt = w['prev_tenure'].values.astype(float)

w['d_tenure'] = t - pt  # always 1
w['d_tenure_sq'] = t**2 - pt**2  # = 2t - 1
w['d_tenure_cu'] = t**3 - pt**3  # = 3t^2 - 3t + 1
w['d_tenure_qu'] = t**4 - pt**4  # = 4t^3 - 6t^2 + 4t - 1

print(f"\nd_tenure_sq stats: mean={w['d_tenure_sq'].mean():.2f}, std={w['d_tenure_sq'].std():.2f}")
print(f"d_tenure_cu stats: mean={w['d_tenure_cu'].mean():.2f}, std={w['d_tenure_cu'].std():.2f}")
print(f"d_tenure_qu stats: mean={w['d_tenure_qu'].mean():.2f}, std={w['d_tenure_qu'].std():.2f}")

corr = w[['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu']].corr()
print(f"\nCorrelation matrix:")
print(corr.to_string(float_format='%.4f'))

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_ten = w[['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu']].values
print(f"\nVIF:")
for i, name in enumerate(['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu']):
    vif = variance_inflation_factor(X_ten, i)
    print(f"  {name}: {vif:.1f}")

# CRITICAL INSIGHT: maybe the paper SCALES the polynomial terms BEFORE regression
# If you scale d_tenure_sq by 1/100, d_tenure_cu by 1/1000, d_tenure_qu by 1/10000
# the variables have different scales but the VIF doesn't change
# So scaling shouldn't affect the issue

# But wait -- could the issue be that tenure should be scaled by 10 before polynomials?
# What if Topel uses tenure/10 in polynomials?
# "Delta Tenure^2 (x10^2)" could mean (tenure/10)^2 * 10^2 = tenure^2
# No, that's the same thing

# Actually, the paper says "x10^2" means the coefficient is MULTIPLIED by 10^2
# i.e., coefficient * 100 = reported value
# So the raw regressor is (tenure_t^2 - tenure_{t-1}^2) without any scaling

# The problem might be that our tenure variable doesn't match the paper's.
# Paper's Table A1 says mean tenure (S) = 9.365 for within-job obs
# Let me check: what is our mean tenure for the within-job sample?

# Actually, Table A1 describes the cross-sectional level data, not the differenced data
# In the differenced data, tenure ranges from 1 to max_tenure
# The mean tenure in differenced data should be different

# Check tenure in the undifferenced (level) data
# Paper's Table A1: N=10,894 obs, Mean S=9.365
# Our data (level):
print(f"\n--- Level data stats ---")
df_level = df.copy()
print(f"N (level): {len(df_level)}")
print(f"Mean tenure (level): {df_level['tenure'].mean():.2f}")
print(f"Mean experience (level): {df_level['experience'].mean():.1f}")

# Hmm, our level data has 13,993 obs vs paper's 10,894
# Maybe the paper applies restrictions before computing within-job differences

# What if Topel restricts EXPERIENCE >= 1 and TENURE >= 0 in the LEVEL data first?
df_restricted = df_level[(df_level['experience'] >= 1) & (df_level['tenure'] >= 0)]
print(f"\nAfter exp >= 1, tenure >= 0 (level): N={len(df_restricted)}")
print(f"Mean tenure: {df_restricted['tenure'].mean():.2f}")
print(f"Mean experience: {df_restricted['experience'].mean():.1f}")

# What about more restrictive experience?
for max_exp in [35, 40, 45]:
    dr = df_level[(df_level['experience'] >= 1) & (df_level['experience'] <= max_exp)]
    print(f"exp 1-{max_exp}: N_level={len(dr)}, mean_tenure={dr['tenure'].mean():.2f}")

# NOW: try scaling tenure and experience by /10 before taking polynomials
# This is a common numerical approach to reduce multicollinearity
print("\n\n=== Try with tenure/10 and experience/10 polynomial terms ===")
# This shouldn't change coefficients (they'll be scaled by 10^k) but might help numerically

# Actually, let me try a COMPLETELY different approach:
# What if the polynomial terms are constructed differently?
# Instead of d(T^2) = T^2 - (T-1)^2 = 2T - 1
# What if Topel uses T^2 - T_{t-1}^2 where T_{t-1} might not equal T-1?
# No, within a job d_tenure = 1 always

# Let me check: does using the FULL panel (1970-1983) help?
print("\n\n=== Using psid_panel_full.csv ===")
df2 = pd.read_csv('data/psid_panel_full.csv')
print(f"Full: {len(df2)} obs, years {sorted(df2['year'].unique())}")

# Apply same pipeline
df2['educ_raw'] = df2['education_clean'].copy()
cat_mask2 = ~df2['year'].isin([1975, 1976])
df2.loc[cat_mask2, 'educ_raw'] = df2.loc[cat_mask2, 'education_clean'].map({**EDUC_MAP, 9: np.nan})

person_educ2 = df2.groupby('person_id').apply(get_fixed_educ)
df2['education_fixed'] = df2['person_id'].map(person_educ2)
df2 = df2[df2['education_fixed'].notna()].copy()
df2['experience'] = df2['age'] - df2['education_fixed'] - 6
df2['tenure'] = df2['tenure_topel'] - 1

df2 = df2.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp2 = df2.groupby(['person_id', 'job_id'])
df2['prev_year'] = grp2['year'].shift(1)
df2['prev_log_wage'] = grp2['log_hourly_wage'].shift(1)
df2['prev_tenure'] = grp2['tenure'].shift(1)
df2['prev_experience'] = grp2['experience'].shift(1)

within2 = df2[
    (df2['prev_year'].notna()) &
    (df2['year'] - df2['prev_year'] == 1)
].copy()
within2['d_log_wage'] = within2['log_hourly_wage'] - within2['prev_log_wage']
within2['d_exp'] = within2['experience'] - within2['prev_experience']

base2 = within2[within2['d_exp'] == 1].copy()
m02, s02 = base2['d_log_wage'].mean(), base2['d_log_wage'].std()

print(f"Full panel: d_exp==1 obs = {len(base2)}, mean={m02:.4f}, SD={s02:.4f}")

# 2-SD trim
w2 = base2[(base2['d_log_wage'] >= m02 - 2*s02) & (base2['d_log_wage'] <= m02 + 2*s02)].copy()
print(f"After 2-SD trim: N={len(w2)}")

# What N do we get with various k?
for k in np.arange(1.8, 2.2, 0.02):
    wk = base2[(base2['d_log_wage'] >= m02 - k*s02) & (base2['d_log_wage'] <= m02 + k*s02)]
    if abs(len(wk) - 8683) < 100:
        print(f"  k={k:.2f}: N={len(wk)}")
