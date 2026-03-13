#!/usr/bin/env python3
"""
Find the best combination: constructed experience + right filters for N.
The key insight from retry_7 was that constructed experience (always d_exp=1)
gives the best DT coefficient but N was too high (10,601).
We need to find the right filters to bring N down to ~8,683.
"""
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

# Constructed experience: always increments by 1
df = df.sort_values(['person_id', 'year']).reset_index(drop=True)
first_obs = df.groupby('person_id').first()[['age', 'year']].reset_index()
first_obs.columns = ['person_id', 'age_first', 'year_first']
df = df.merge(first_obs, on='person_id')
df['initial_exp'] = df['age_first'] - df['education_fixed'] - 6
df['experience'] = df['initial_exp'] + (df['year'] - df['year_first'])
# Also compute age-based experience for comparison
df['experience_age'] = df['age'] - df['education_fixed'] - 6

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

print(f"All within-job obs: {len(within)}")
print(f"d_exp distribution: {within['d_exp'].value_counts().sort_index().to_dict()}")

# With constructed experience, d_exp should always be 1
# But need to filter outliers

# Various filter strategies to get N ~ 8,683

# Strategy A: 2-SD trim
base = within.copy()
m0, s0 = base['d_log_wage'].mean(), base['d_log_wage'].std()
wA = base[(base['d_log_wage'] >= m0 - 2*s0) & (base['d_log_wage'] <= m0 + 2*s0)]
print(f"\nStrategy A (2-SD trim): N={len(wA)}")

# Strategy B: initial_exp >= 1 + 2-SD trim
wB = within[within['initial_exp'] >= 1].copy()
m0, s0 = wB['d_log_wage'].mean(), wB['d_log_wage'].std()
wB = wB[(wB['d_log_wage'] >= m0 - 2*s0) & (wB['d_log_wage'] <= m0 + 2*s0)]
print(f"Strategy B (initial_exp>=1 + 2-SD): N={len(wB)}")

# Strategy C: experience >= 1 + 2-SD trim
wC = within[within['experience'] >= 1].copy()
m0, s0 = wC['d_log_wage'].mean(), wC['d_log_wage'].std()
wC = wC[(wC['d_log_wage'] >= m0 - 2*s0) & (wC['d_log_wage'] <= m0 + 2*s0)]
print(f"Strategy C (exp>=1 + 2-SD): N={len(wC)}")

# Strategy D: experience >= 1 + experience_age == experience (no age inconsistency) + 2-SD
# This filters out observations where age changed by != 1
wD = within[(within['experience'] >= 1)].copy()
# Also check: d_exp_age (from age-based experience)
wD['d_exp_age'] = wD['experience_age'] - grp['experience_age'].shift(1).reindex(wD.index)
# Actually, we already computed within from df, let me redo properly
df['prev_experience_age'] = grp['experience_age'].shift(1)
within2 = df[
    (df['prev_year'].notna()) &
    (df['year'] - df['prev_year'] == 1)
].copy()
within2['d_log_wage'] = within2['log_hourly_wage'] - within2['prev_log_wage']
within2['d_exp'] = within2['experience'] - within2['prev_experience']
within2['d_exp_age'] = within2['experience_age'] - within2['prev_experience_age']

# Strategy D: keep only obs where d_exp_age == 1 (natural filter) + use constructed experience
wD = within2[within2['d_exp_age'] == 1].copy()
m0, s0 = wD['d_log_wage'].mean(), wD['d_log_wage'].std()
wD2 = wD[(wD['d_log_wage'] >= m0 - 2*s0) & (wD['d_log_wage'] <= m0 + 2*s0)]
print(f"Strategy D (d_exp_age==1 + 2-SD): N={len(wD2)}")

# Strategy E: d_exp_age==1 only, no SD trim (original approach)
wE = within2[within2['d_exp_age'] == 1].copy()
wE = wE[wE['d_log_wage'].between(-2, 2)]
print(f"Strategy E (d_exp_age==1 + +-2 fixed): N={len(wE)}")

# Strategy F: drop persons with exp_age < 1 + d_exp_age==1 + +-2 fixed
person_min_exp = df.groupby('person_id')['experience_age'].min()
valid_persons = person_min_exp[person_min_exp >= 1].index
wF = within2[within2['person_id'].isin(valid_persons)].copy()
wF = wF[wF['d_exp_age'] == 1]
wF = wF[wF['d_log_wage'].between(-2, 2)]
print(f"Strategy F (drop_pers_exp<1 + d_exp_age==1 + +-2): N={len(wF)}")

# Strategy G: like F but also drop persons with exp_age < 1, and use 2-SD trim instead of +-2
wG = within2[within2['person_id'].isin(valid_persons)].copy()
wG = wG[wG['d_exp_age'] == 1]
m0, s0 = wG['d_log_wage'].mean(), wG['d_log_wage'].std()
wG = wG[(wG['d_log_wage'] >= m0 - 2*s0) & (wG['d_log_wage'] <= m0 + 2*s0)]
print(f"Strategy G (drop_pers_exp<1 + d_exp_age==1 + 2-SD): N={len(wG)}")

# For each strategy that gives N close to 8683, run quick Model 1 regression
def quick_regression(w_df, label):
    """Run Model 1 and return key stats."""
    t = w_df['tenure'].values.astype(float)
    pt = w_df['prev_tenure'].values.astype(float)
    e = w_df['experience'].values.astype(float)
    pe = w_df['prev_experience'].values.astype(float)

    w_df = w_df.copy()
    w_df['d_tenure'] = t - pt
    w_df['d_exp_sq'] = e**2 - pe**2
    w_df['d_exp_cu'] = e**3 - pe**3
    w_df['d_exp_qu'] = e**4 - pe**4

    yd = pd.get_dummies(w_df['year'], prefix='yr', dtype=float)
    yc = sorted(yd.columns.tolist())[1:]
    yv = w_df['d_log_wage'].values

    vl = ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    X = pd.concat([w_df[vl].reset_index(drop=True), yd[yc].reset_index(drop=True)], axis=1)
    valid = np.isfinite(X.values).all(axis=1) & np.isfinite(yv)
    md = sm.OLS(yv[valid], X.loc[valid].values, hasconst=True).fit()
    names = vl + yc

    dt_c = md.params[0]
    exp2_c = md.params[1] * 100
    exp3_c = md.params[2] * 1000
    exp4_c = md.params[3] * 10000
    se_reg = np.sqrt(md.mse_resid)
    r2 = md.rsquared

    print(f"\n{label}:")
    print(f"  N={int(md.nobs)}, SE_reg={se_reg:.4f}, R^2={r2:.4f}")
    print(f"  DT={dt_c:.4f} (paper: 0.1242)")
    print(f"  EXP2={exp2_c:.4f} (paper: -0.6051)")
    print(f"  EXP3={exp3_c:.4f} (paper: 0.1460)")
    print(f"  EXP4={exp4_c:.4f} (paper: 0.0131)")

    return md, names

# Run regressions for key strategies
print("\n" + "="*60)
print("REGRESSION COMPARISON")
print("="*60)

quick_regression(wA, "Strategy A (2-SD trim, all obs)")
quick_regression(wD2, "Strategy D (d_exp_age==1 + 2-SD)")
quick_regression(wE, "Strategy E (d_exp_age==1 + +-2)")
quick_regression(wF, "Strategy F (drop_pers + d_exp_age==1 + +-2)")
quick_regression(wG, "Strategy G (drop_pers + d_exp_age==1 + 2-SD)")

# The trade-off is clear:
# - Larger samples (no trim or +-2) give better coefficient magnitudes
# - Smaller samples (2-SD trim) give better N and R^2
# Strategy F has N=8968 with reasonable coefficients

# Try to find a trimming approach that gives BOTH N~8683 AND good coefficients
# What if we use +-2 fixed AND an experience upper bound?
print("\n\n=== Fine-tuned strategies ===")
wH = within2[within2['d_exp_age'] == 1].copy()
wH = wH[wH['d_log_wage'].between(-2, 2)]
for max_exp in [35, 36, 37, 38, 39, 40]:
    wh = wH[wH['experience'] <= max_exp]
    print(f"  d_exp_age==1 + +-2 + exp<={max_exp}: N={len(wh)}")

# Strategy I: d_exp_age==1 + +-2 + experience <= 40
wI = wH[wH['experience'] <= 40].copy()
quick_regression(wI, "Strategy I (d_exp_age==1 + +-2 + exp<=40)")

# Strategy J: d_exp_age==1 + +-2 + experience 1-40
wJ = wH[(wH['experience'] >= 1) & (wH['experience'] <= 40)].copy()
quick_regression(wJ, "Strategy J (d_exp_age==1 + +-2 + exp 1-40)")

# Let me also try: what if we compute BOTH d_exp and check if d_exp_age == 1?
# and also try a different outlier measure
# What about trimming based on RESIDUALS from a simple model?

# Strategy K: trim top/bottom 2.5% of d_log_wage (instead of 2-SD)
wK = within2[within2['d_exp_age'] == 1].copy()
lo, hi = wK['d_log_wage'].quantile(0.025), wK['d_log_wage'].quantile(0.975)
wK = wK[(wK['d_log_wage'] >= lo) & (wK['d_log_wage'] <= hi)]
print(f"\nStrategy K (d_exp_age==1 + 2.5% trim): N={len(wK)}")
quick_regression(wK, "Strategy K (2.5% trim)")

# Strategy L: trim top/bottom 2% of d_log_wage
wL = within2[within2['d_exp_age'] == 1].copy()
lo, hi = wL['d_log_wage'].quantile(0.02), wL['d_log_wage'].quantile(0.98)
wL = wL[(wL['d_log_wage'] >= lo) & (wL['d_log_wage'] <= hi)]
print(f"\nStrategy L (d_exp_age==1 + 2% trim): N={len(wL)}")
quick_regression(wL, "Strategy L (2% trim)")
