#!/usr/bin/env python3
"""Find the right combination of filters to get N=8,683."""
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

# Experience using age
df['experience'] = df['age'] - df['education_fixed'] - 6

# Tenure
df['tenure'] = df['tenure_topel'] - 1

# Within-job first differences
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
within['d_tenure'] = within['tenure'] - within['prev_tenure']

# Base sample: d_exp == 1
base = within[within['d_exp'] == 1].copy()
base = base[base['d_log_wage'].between(-2, 2)].copy()
print(f"Base: N={len(base)}, persons={base['person_id'].nunique()}")

# Approach 1: Exp range restriction
# Paper says "1 to 40 years of experience" in Table A1
# Table 2 is not explicit, but likely similar
for max_exp in [35, 36, 37, 38, 39, 40]:
    w = base[(base['experience'] >= 1) & (base['experience'] <= max_exp)]
    print(f"  exp 1-{max_exp}: N={len(w)}, persons={w['person_id'].nunique()}")

# Approach 2: Tenure restriction
# What if tenure must be >= some value?
for max_ten in [15, 20, 25]:
    w = base[base['tenure'] <= max_ten]
    print(f"  tenure <= {max_ten}: N={len(w)}, persons={w['person_id'].nunique()}")

# Approach 3: Age restriction
# Topel may restrict to certain age range
for min_age, max_age in [(18, 60), (18, 55), (18, 64), (20, 60)]:
    w = base[(base['age'] >= min_age) & (base['age'] <= max_age)]
    print(f"  age {min_age}-{max_age}: N={len(w)}")

# Approach 4: Combination with experience range
print("\n--- Combinations with exp 1-40 and additional filters ---")
w_exp = base[(base['experience'] >= 1) & (base['experience'] <= 40)]
print(f"exp 1-40: N={len(w_exp)}")

# Try additional N reductions
# Minimum observations per person-job spell
spell_size = w_exp.groupby(['person_id', 'job_id']).size()
for min_spell in [2, 3]:
    valid_spells = spell_size[spell_size >= min_spell].index
    w = w_exp.set_index(['person_id', 'job_id']).loc[valid_spells].reset_index()
    print(f"  + min_spell >= {min_spell}: N={len(w)}")

# Minimum observations per person
person_size = w_exp.groupby('person_id').size()
for min_p in [2, 3]:
    valid = person_size[person_size >= min_p].index
    w = w_exp[w_exp['person_id'].isin(valid)]
    print(f"  + min_person_obs >= {min_p}: N={len(w)}")

# Approach 5: What about using the full panel?
print("\n--- Using psid_panel_full.csv ---")
df_full = pd.read_csv('data/psid_panel_full.csv')
print(f"Full panel: {len(df_full)} obs, {df_full['person_id'].nunique()} persons")
print(f"Years: {sorted(df_full['year'].unique())}")
print(f"Columns: {list(df_full.columns)}")

# Apply same treatment to full panel
df_full['educ_raw'] = df_full['education_clean'].copy()
cat_mask = ~df_full['year'].isin([1975, 1976])
df_full.loc[cat_mask, 'educ_raw'] = df_full.loc[cat_mask, 'education_clean'].map({**EDUC_MAP, 9: np.nan})
person_educ_f = df_full.groupby('person_id').apply(get_fixed_educ)
df_full['education_fixed'] = df_full['person_id'].map(person_educ_f)
df_full = df_full[df_full['education_fixed'].notna()].copy()
df_full['experience'] = df_full['age'] - df_full['education_fixed'] - 6
df_full['tenure'] = df_full['tenure_topel'] - 1

df_full = df_full.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp_f = df_full.groupby(['person_id', 'job_id'])
df_full['prev_year'] = grp_f['year'].shift(1)
df_full['prev_log_wage'] = grp_f['log_hourly_wage'].shift(1)
df_full['prev_tenure'] = grp_f['tenure'].shift(1)
df_full['prev_experience'] = grp_f['experience'].shift(1)

within_f = df_full[
    (df_full['prev_year'].notna()) &
    (df_full['year'] - df_full['prev_year'] == 1)
].copy()
within_f['d_log_wage'] = within_f['log_hourly_wage'] - within_f['prev_log_wage']
within_f['d_exp'] = within_f['experience'] - within_f['prev_experience']

base_f = within_f[within_f['d_exp'] == 1].copy()
base_f = base_f[base_f['d_log_wage'].between(-2, 2)].copy()
print(f"Full panel, d_exp==1, outlier +-2: N={len(base_f)}")

for max_exp in [35, 40, 45]:
    w = base_f[(base_f['experience'] >= 1) & (base_f['experience'] <= max_exp)]
    print(f"  + exp 1-{max_exp}: N={len(w)}")

# Now try the best approach: run a quick regression to see coefficient quality
print("\n\n=== Quick regression with best N approach ===")
# Let's try: d_exp==1, outlier +-2, exp 1-40
w_best = base[(base['experience'] >= 1) & (base['experience'] <= 40)].copy()
print(f"N={len(w_best)}, persons={w_best['person_id'].nunique()}")

t = w_best['tenure'].values.astype(float)
pt = w_best['prev_tenure'].values.astype(float)
e = w_best['experience'].values.astype(float)
pe = w_best['prev_experience'].values.astype(float)

w_best['d_tenure'] = t - pt
w_best['d_tenure_sq'] = t**2 - pt**2
w_best['d_tenure_cu'] = t**3 - pt**3
w_best['d_tenure_qu'] = t**4 - pt**4
w_best['d_exp_sq'] = e**2 - pe**2
w_best['d_exp_cu'] = e**3 - pe**3
w_best['d_exp_qu'] = e**4 - pe**4

year_dummies = pd.get_dummies(w_best['year'], prefix='yr', dtype=float)
yr_cols = sorted(year_dummies.columns.tolist())[1:]

y = w_best['d_log_wage'].values

# Model 1
var_list = ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
X = pd.concat([w_best[var_list].reset_index(drop=True),
               year_dummies[yr_cols].reset_index(drop=True)], axis=1)
m1 = sm.OLS(y, X.values, hasconst=True).fit()
names1 = var_list + yr_cols

# Model 3
var_list3 = ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
             'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
X3 = pd.concat([w_best[var_list3].reset_index(drop=True),
                year_dummies[yr_cols].reset_index(drop=True)], axis=1)
m3 = sm.OLS(y, X3.values, hasconst=True).fit()
names3 = var_list3 + yr_cols

gt = [
    ('d_tenure', 1, 0.1242, 0.0161),
    ('d_exp_sq', 100, -0.6051, 0.1430),
    ('d_exp_cu', 1000, 0.1460, 0.0482),
    ('d_exp_qu', 10000, 0.0131, 0.0054),
]

print("\nModel 1 comparison:")
for var, scale, gt_c, gt_s in gt:
    idx = names1.index(var)
    c, s = m1.params[idx]*scale, m1.bse[idx]*scale
    print(f"  {var:>15s}: gen={c:>8.4f} paper={gt_c:>8.4f} diff={c-gt_c:>8.4f}  SE: gen={s:>8.4f} paper={gt_s:>8.4f}")

gt3 = [
    ('d_tenure', 1, 0.1258, 0.0162),
    ('d_tenure_sq', 100, -0.4592, 0.1080),
    ('d_tenure_cu', 1000, 0.1846, 0.0526),
    ('d_tenure_qu', 10000, -0.0245, 0.0079),
    ('d_exp_sq', 100, -0.4067, 0.1546),
    ('d_exp_cu', 1000, 0.0989, 0.0517),
    ('d_exp_qu', 10000, 0.0089, 0.0058),
]

print("\nModel 3 comparison:")
for var, scale, gt_c, gt_s in gt3:
    idx = names3.index(var)
    c, s = m3.params[idx]*scale, m3.bse[idx]*scale
    print(f"  {var:>15s}: gen={c:>8.4f} paper={gt_c:>8.4f} diff={c-gt_c:>8.4f}  SE: gen={s:>8.4f} paper={gt_s:>8.4f}")

print(f"\nR^2: {m1.rsquared:.4f} (paper: .022), {m3.rsquared:.4f} (paper: .025)")
print(f"SE of regression: {np.sqrt(m1.mse_resid):.4f} (paper: .218), {np.sqrt(m3.mse_resid):.4f} (paper: .218)")

# Now try with experience >= 1 only (no upper limit), but scaling polynomials differently
# What if we scale differently? Check the paper's scaling convention
print("\n\n=== Understanding scaling convention ===")
print("d_tenure = 1 always (it's the intercept/constant in the diff model)")
print(f"d_tenure coef raw = {m1.params[0]:.6f}")
print(f"d_tenure coef (paper reports as is) = {m1.params[0]:.4f}, paper = 0.1242")
print()

# Check: are the experience polynomial terms scaled before or after regression?
# Paper says "x10^2" etc. This means the REPORTED coefficient is the raw coefficient * scale
# So raw_coef * 100 = reported_coef for x10^2
# This is standard: you divide the variable by 100, or equivalently multiply the coefficient by 100
idx_exp_sq = names1.index('d_exp_sq')
print(f"d_exp_sq raw coef = {m1.params[idx_exp_sq]:.8f}")
print(f"d_exp_sq * 100 = {m1.params[idx_exp_sq]*100:.4f}, paper = -0.6051")

# The question is: does Topel SCALE the variables before regression, or scale coefficients after?
# If he scales variables (divides by 100), the raw X values are smaller, which shouldn't change coefs
# Let's check: try scaling variables before regression
print("\n=== Try scaling variables BEFORE regression ===")
w_test = w_best.copy()
w_test['d_exp_sq_s'] = w_test['d_exp_sq'] / 100
w_test['d_exp_cu_s'] = w_test['d_exp_cu'] / 1000
w_test['d_exp_qu_s'] = w_test['d_exp_qu'] / 10000

var_s = ['d_tenure', 'd_exp_sq_s', 'd_exp_cu_s', 'd_exp_qu_s']
X_s = pd.concat([w_test[var_s].reset_index(drop=True),
                 year_dummies[yr_cols].reset_index(drop=True)], axis=1)
m1s = sm.OLS(y, X_s.values, hasconst=True).fit()
names1s = var_s + yr_cols

for var_orig, var_scaled, gt_c, gt_s in [
    ('d_tenure', 'd_tenure', 0.1242, 0.0161),
    ('d_exp_sq_s', 'd_exp_sq_s', -0.6051, 0.1430),
    ('d_exp_cu_s', 'd_exp_cu_s', 0.1460, 0.0482),
    ('d_exp_qu_s', 'd_exp_qu_s', 0.0131, 0.0054),
]:
    idx = names1s.index(var_scaled)
    c, s = m1s.params[idx], m1s.bse[idx]
    print(f"  {var_scaled:>15s}: gen={c:>8.4f} paper={gt_c:>8.4f} diff={c-gt_c:>8.4f}  SE: gen={s:>8.4f} paper={gt_s:>8.4f}")
print("(These should be identical to the post-scaling approach)")
print(f"R^2: {m1s.rsquared:.4f}, SE: {np.sqrt(m1s.mse_resid):.4f}")
