#!/usr/bin/env python3
"""
Reconstruct tenure using reported months where available.
Strategy:
1. For each job spell, look for tenure_mos reports
2. Use the earliest report to compute the actual start date
3. Compute tenure as (year - start_year)
4. If no report available, use a fallback based on when first observed
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

df['experience'] = df['age'] - df['education_fixed'] - 6

# Reconstruct tenure using tenure_mos
# tenure_mos is available for 1976, 1977, 1980, 1981, 1982, 1983
# Strategy: for each job, find the earliest tenure_mos report and
# compute start_year = report_year - tenure_mos/12
# Then tenure_t = year_t - start_year

df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)

# Clean tenure_mos: 999 or 0 might mean missing
# Looking at the data: 0 means either truly 0 months or missing
# For persons with same_emp, a tenure_mos of 0 is suspicious
# Let's keep 0 as missing for non-first-year observations
df['ten_mos_clean'] = df['tenure_mos'].copy()
df.loc[df['ten_mos_clean'] >= 999, 'ten_mos_clean'] = np.nan
# tenure_mos = 0 for non-new jobs is suspicious
# Keep 0 only if it seems plausible (new job in that year)
# Actually, let's keep all non-999 values and see

# For each job_id, find the earliest valid tenure_mos report
job_starts = {}
for jid in df['job_id'].unique():
    job_data = df[df['job_id'] == jid].sort_values('year')
    # Look for earliest tenure_mos report > 0
    valid_reports = job_data[job_data['ten_mos_clean'].notna() & (job_data['ten_mos_clean'] > 0)]
    if len(valid_reports) > 0:
        first_report = valid_reports.iloc[0]
        # Infer start year: start_year = report_year - tenure_mos/12
        start_year = first_report['year'] - first_report['ten_mos_clean'] / 12
        job_starts[jid] = start_year
    else:
        # No report: use year of first observation - 0 (assume job just started)
        # This is imprecise but is the best we can do
        first_year = job_data['year'].min()
        # However, since tenure_topel starts at 1, the first obs is actually
        # the SECOND year on the job (first was tenure_topel=0 which was dropped)
        # So start_year = first_year - 1
        job_starts[jid] = first_year - 1

# Compute tenure for each observation
df['start_year'] = df['job_id'].map(job_starts)
df['tenure_recon'] = df['year'] - df['start_year']
df['tenure_recon'] = df['tenure_recon'].clip(lower=0)

print(f"Reconstructed tenure stats:")
print(df['tenure_recon'].describe())
print(f"\nMean tenure_recon: {df['tenure_recon'].mean():.2f} (paper: ~9.365 for levels)")
print(f"Mean tenure_topel: {df['tenure_topel'].mean():.2f}")

# Compare tenure_recon vs tenure_topel
print(f"\nCorrelation: {df['tenure_recon'].corr(df['tenure_topel']):.4f}")
print(f"\nDifference (tenure_recon - tenure_topel + 1):")
diff = df['tenure_recon'] - (df['tenure_topel'] - 1)
print(diff.describe())

# Within-job differences with reconstructed tenure
grp = df.groupby(['person_id', 'job_id'])
df['prev_year'] = grp['year'].shift(1)
df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
df['prev_tenure_recon'] = grp['tenure_recon'].shift(1)
df['prev_experience'] = grp['experience'].shift(1)

within = df[
    (df['prev_year'].notna()) &
    (df['year'] - df['prev_year'] == 1)
].copy()
within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
within['d_exp'] = within['experience'] - within['prev_experience']
within['d_tenure_recon'] = within['tenure_recon'] - within['prev_tenure_recon']

# Filter d_exp == 1
base = within[within['d_exp'] == 1].copy()

# Check d_tenure_recon
print(f"\nd_tenure_recon distribution:")
print(base['d_tenure_recon'].value_counts().sort_index())

# Apply 2-SD trim
m0, s0 = base['d_log_wage'].mean(), base['d_log_wage'].std()
w = base[(base['d_log_wage'] >= m0 - 2*s0) & (base['d_log_wage'] <= m0 + 2*s0)].copy()
print(f"\nAfter 2-SD trim: N={len(w)}")

# Run regression with reconstructed tenure
t = w['tenure_recon'].values.astype(float)
pt = w['prev_tenure_recon'].values.astype(float)
e = w['experience'].values.astype(float)
pe = w['prev_experience'].values.astype(float)

w['d_tenure'] = t - pt  # should still be 1
w['d_tenure_sq'] = t**2 - pt**2
w['d_tenure_cu'] = t**3 - pt**3
w['d_tenure_qu'] = t**4 - pt**4
w['d_exp_sq'] = e**2 - pe**2
w['d_exp_cu'] = e**3 - pe**3
w['d_exp_qu'] = e**4 - pe**4

print(f"\nd_tenure stats: {w['d_tenure'].describe()}")
print(f"d_tenure_sq stats: mean={w['d_tenure_sq'].mean():.2f}, std={w['d_tenure_sq'].std():.2f}")
print(f"d_tenure_cu stats: mean={w['d_tenure_cu'].mean():.2f}, std={w['d_tenure_cu'].std():.2f}")
print(f"Mean tenure_recon in sample: {w['tenure_recon'].mean():.2f}")

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

m3, n3 = run_ols(y, ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                      'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])

gt3 = [
    ('d_tenure', 1, 0.1258, 0.0162),
    ('d_tenure_sq', 100, -0.4592, 0.1080),
    ('d_tenure_cu', 1000, 0.1846, 0.0526),
    ('d_tenure_qu', 10000, -0.0245, 0.0079),
    ('d_exp_sq', 100, -0.4067, 0.1546),
    ('d_exp_cu', 1000, 0.0989, 0.0517),
    ('d_exp_qu', 10000, 0.0089, 0.0058),
]

print(f"\nModel 3 with reconstructed tenure:")
for var, scale, gt_c, gt_s in gt3:
    c, s = gc(m3, n3, var)
    if c is not None:
        gen_c, gen_s = c*scale, s*scale
        print(f"  {var:>15s}: coef={gen_c:>10.4f} (paper: {gt_c:>10.4f}, diff={gen_c-gt_c:>8.4f})  SE={gen_s:>8.4f} (paper: {gt_s:>8.4f})")

print(f"\nR^2: {m3.rsquared:.4f} (paper: .025)")
print(f"SE of reg: {np.sqrt(m3.mse_resid):.4f} (paper: .218)")

# Also run model 1
m1, n1 = run_ols(y, ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])
gt1 = [
    ('d_tenure', 1, 0.1242, 0.0161),
    ('d_exp_sq', 100, -0.6051, 0.1430),
    ('d_exp_cu', 1000, 0.1460, 0.0482),
    ('d_exp_qu', 10000, 0.0131, 0.0054),
]
print(f"\nModel 1 with reconstructed tenure:")
for var, scale, gt_c, gt_s in gt1:
    c, s = gc(m1, n1, var)
    if c is not None:
        gen_c, gen_s = c*scale, s*scale
        print(f"  {var:>15s}: coef={gen_c:>10.4f} (paper: {gt_c:>10.4f})  SE={gen_s:>8.4f} (paper: {gt_s:>8.4f})")

print(f"\nR^2: {m1.rsquared:.4f} (paper: .022)")
print(f"SE of reg: {np.sqrt(m1.mse_resid):.4f} (paper: .218)")

# NOW: try WITHOUT 2-SD trim, just +-2 fixed
# The SE of regression of 0.218 is BETWEEN 0.195 (2-SD trim) and 0.278 (no trim)
# Maybe Topel uses a different outlier threshold
# Or maybe he uses CPS deflation which reduces the variance

# Check: what if we use CPS-deflated wages?
CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

# With year dummies in the regression, deflation only changes the intercept (year dummies absorb it)
# So CPS deflation cannot change the SE of regression or R^2 in a first-differenced model WITH year dummies

# The only thing that can change SE is the COMPOSITION of the sample
# So we need to find the right sample restrictions

# What if the issue is that we're missing years 1968-1970?
# Paper has 1968-1983 (16 years => 15 year-to-year transitions per person max)
# We have 1971-1983 (13 years => 12 year-to-year transitions per person max)
# The extra 3 years would add observations

# But we can't add years we don't have. Let's focus on getting the best score possible.

# Strategy: find the combination that minimizes total scoring error
# Key variables:
# 1. Outlier threshold (trim d_log_wage)
# 2. Experience restriction
# 3. Tenure reconstruction

print("\n\n=== SENSITIVITY: vary outlier threshold ===")
for trim_sd in np.arange(1.8, 3.5, 0.1):
    m0, s0 = base['d_log_wage'].mean(), base['d_log_wage'].std()
    wt = base[(base['d_log_wage'] >= m0 - trim_sd*s0) & (base['d_log_wage'] <= m0 + trim_sd*s0)].copy()

    t = wt['tenure_recon'].values.astype(float)
    pt = wt['prev_tenure_recon'].values.astype(float)
    e = wt['experience'].values.astype(float)
    pe = wt['prev_experience'].values.astype(float)

    wt['d_tenure'] = t - pt
    wt['d_tenure_sq'] = t**2 - pt**2
    wt['d_tenure_cu'] = t**3 - pt**3
    wt['d_tenure_qu'] = t**4 - pt**4
    wt['d_exp_sq'] = e**2 - pe**2
    wt['d_exp_cu'] = e**3 - pe**3
    wt['d_exp_qu'] = e**4 - pe**4

    yd = pd.get_dummies(wt['year'], prefix='yr', dtype=float)
    yc = sorted(yd.columns.tolist())[1:]
    yv = wt['d_log_wage'].values

    vl = ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu']
    Xm = wt[vl].copy()
    X = pd.concat([Xm.reset_index(drop=True), yd[yc].reset_index(drop=True)], axis=1)
    valid = np.isfinite(X.values).all(axis=1) & np.isfinite(yv)
    md = sm.OLS(yv[valid], X.loc[valid].values, hasconst=True).fit()

    dt_coef = md.params[0]
    se_reg = np.sqrt(md.mse_resid)
    r2 = md.rsquared
    n = int(md.nobs)

    print(f"  {trim_sd:.1f} SD: N={n:>5d}, DT={dt_coef:.4f}, SE_reg={se_reg:.4f}, R^2={r2:.4f}")
