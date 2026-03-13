"""Debug script to explore different strategies for attempt 14"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

df = pd.read_csv('data/psid_panel.csv')
df['education_years'] = df['education_clean'].map(EDUC_MAP).fillna(12)
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df = df[df['tenure_topel'] >= 1].copy()
df = df.dropna(subset=['log_hourly_wage']).copy()

# Don't filter occ=0,3,9 yet - let's see what we get
df_full = df.copy()
df = df[~df['occ_1digit'].isin([0, 3, 9])].copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

df = df.sort_values(['person_id', 'job_id', 'year'])
ju = df.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
df = df.merge(ju, on=['person_id', 'job_id'], how='left')

# Current grouping
ps = df[df['occ_1digit'].isin([1, 2, 4, 8])]
bc_nu = df[df['occ_1digit'].isin([5, 6, 7]) & (df['job_union'] == 0)]
bc_u = df[df['occ_1digit'].isin([5, 6, 7]) & (df['job_union'] == 1)]
print(f'Current: PS={len(ps)}, BC_NU={len(bc_nu)}, BC_U={len(bc_u)}')
print(f'Paper:   PS=4946, BC_NU=2642, BC_U=2741')

# Obs-level union
bc_nu2 = df[df['occ_1digit'].isin([5, 6, 7]) & (df['union_member'] == 0)]
bc_u2 = df[df['occ_1digit'].isin([5, 6, 7]) & (df['union_member'] == 1)]
print(f'Obs-level union: BC_NU={len(bc_nu2)}, BC_U={len(bc_u2)}')

# With occ=9 (laborers)
bc9_nu = df[df['occ_1digit'].isin([5, 6, 7, 9]) & (df['job_union'] == 0)]
bc9_u = df[df['occ_1digit'].isin([5, 6, 7, 9]) & (df['job_union'] == 1)]
print(f'With occ=9: BC_NU={len(bc9_nu)}, BC_U={len(bc9_u)}')

# Check full range including occ=0,3,9
df_full2 = df_full.copy()
df_full2 = df_full2[(df_full2['self_employed'] == 0) | (df_full2['self_employed'].isna())].copy()
df_full2 = df_full2.sort_values(['person_id', 'job_id', 'year'])
ju2 = df_full2.groupby(['person_id', 'job_id'])['union_member'].agg(
    lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member': 'job_union'})
df_full2 = df_full2.merge(ju2, on=['person_id', 'job_id'], how='left')

for occ_val in range(10):
    n = len(df_full2[df_full2['occ_1digit'] == occ_val])
    if n > 0:
        print(f'  occ={occ_val}: n={n}')

# Try obs-level union with occ=9 included (this is "Laborers" which should be BC)
bc_obs_9_nu = df_full2[df_full2['occ_1digit'].isin([5, 6, 7, 9]) & (df_full2['union_member'] == 0)]
bc_obs_9_u = df_full2[df_full2['occ_1digit'].isin([5, 6, 7, 9]) & (df_full2['union_member'] == 1)]
print(f'\nWith occ=9, obs-level union: BC_NU={len(bc_obs_9_nu)}, BC_U={len(bc_obs_9_u)}')

# Try job-level union with occ=9 included
bc_job_9_nu = df_full2[df_full2['occ_1digit'].isin([5, 6, 7, 9]) & (df_full2['job_union'] == 0)]
bc_job_9_u = df_full2[df_full2['occ_1digit'].isin([5, 6, 7, 9]) & (df_full2['job_union'] == 1)]
print(f'With occ=9, job-level union: BC_NU={len(bc_job_9_nu)}, BC_U={len(bc_job_9_u)}')

# Now, the biggest issue: BC beta_1 not differentiated.
# Paper: BC_NU beta_1 = 0.107, BC_U beta_1 = 0.059
# Ours:  BC_NU beta_1 = 0.070, BC_U beta_1 = 0.075

# Let's try running the 2-step with obs-level union and occ=9 included
# to see if sample composition changes beta_1

def quick_2step(sub, b1b2, g2=-0.004592, g3=0.0001846, g4=-0.00000245,
                d2=-0.006051, d3=0.0002067, d4=-0.00000238, name=""):
    """Quick 2-step to get beta_1"""
    sub = sub.copy()
    sub['cps_index'] = sub['year'].map(CPS_WAGE_INDEX)
    sub['gnp_defl'] = (sub['year'] - 1).map(GNP_DEFLATOR)
    sub['log_real_wage'] = (sub['log_hourly_wage']
                           - np.log(sub['gnp_defl'] / 100.0)
                           - np.log(sub['cps_index']))
    sub['log_wage_cps'] = sub['log_hourly_wage'] - np.log(sub['cps_index'])
    sub['tenure'] = sub['tenure_topel'].copy()
    sub['initial_experience'] = (sub['experience'] - sub['tenure']).clip(lower=0)

    for c in ['married', 'disabled', 'region_ne', 'region_nc', 'region_south']:
        if c in sub.columns:
            sub[c] = sub[c].fillna(0)

    for y in range(1969, 1984):
        col = f'year_{y}'
        if col not in sub.columns:
            sub[col] = (sub['year'] == y).astype(int)

    sub = sub.dropna(subset=['log_real_wage']).copy()

    sub['w_star'] = (sub['log_real_wage']
                     - b1b2 * sub['tenure']
                     - g2 * sub['tenure']**2
                     - g3 * sub['tenure']**3
                     - g4 * sub['tenure']**4
                     - d2 * sub['experience']**2
                     - d3 * sub['experience']**3
                     - d4 * sub['experience']**4)

    ctrls = ['education_years', 'married', 'disabled',
             'region_ne', 'region_nc', 'region_south']
    yr_use = [f'year_{y}' for y in range(1969, 1984)
              if f'year_{y}' in sub.columns and sub[f'year_{y}'].std() > 0]

    step2 = sub.dropna(subset=['w_star', 'experience', 'initial_experience']).copy()
    ctrls_use = [c for c in ctrls if c in step2.columns and step2[c].std() > 0]

    y_vals = step2['w_star'].values.astype(float)
    exp_v = step2['experience'].values.astype(float)
    x0_v = step2['initial_experience'].values.astype(float)
    ones = np.ones(len(step2))

    ctrl_cols = ctrls_use + yr_use
    while True:
        C = step2[ctrl_cols].values.astype(float)
        Z = np.column_stack([ones, C, x0_v])
        if np.linalg.matrix_rank(Z) >= Z.shape[1]:
            break
        if len(yr_use) == 0:
            break
        yr_use = yr_use[:-1]
        ctrl_cols = ctrls_use + yr_use

    C = step2[ctrl_cols].values.astype(float)
    Z = np.column_stack([ones, C, x0_v])
    X = np.column_stack([ones, C, exp_v])

    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y_vals, rcond=None)[0]

    beta_1 = b[-1]
    beta_2 = b1b2 - beta_1

    print(f"  {name}: N={len(step2)}, beta_1={beta_1:.4f}, beta_2={beta_2:.4f}")
    return beta_1, beta_2, len(step2)

# Test different configurations
print("\n=== CONFIG TESTS ===")

# Config 1: Current (occ 5,6,7, job-level union)
print("\nConfig 1: occ=[5,6,7], job-level union")
sub_nu = df[df['occ_1digit'].isin([5, 6, 7]) & (df['job_union'] == 0)].copy()
sub_u = df[df['occ_1digit'].isin([5, 6, 7]) & (df['job_union'] == 1)].copy()
quick_2step(sub_nu, 0.1520, name="BC_NU")
quick_2step(sub_u, 0.0992, name="BC_U")

# Config 2: obs-level union
print("\nConfig 2: occ=[5,6,7], obs-level union")
sub_nu = df[df['occ_1digit'].isin([5, 6, 7]) & (df['union_member'] == 0)].copy()
sub_u = df[df['occ_1digit'].isin([5, 6, 7]) & (df['union_member'] == 1)].copy()
quick_2step(sub_nu, 0.1520, name="BC_NU")
quick_2step(sub_u, 0.0992, name="BC_U")

# Config 3: with occ=9, job-level union
print("\nConfig 3: occ=[5,6,7,9], job-level union")
sub_nu = df_full2[df_full2['occ_1digit'].isin([5, 6, 7, 9]) & (df_full2['job_union'] == 0)].copy()
sub_u = df_full2[df_full2['occ_1digit'].isin([5, 6, 7, 9]) & (df_full2['job_union'] == 1)].copy()
quick_2step(sub_nu, 0.1520, name="BC_NU")
quick_2step(sub_u, 0.0992, name="BC_U")

# Config 4: with occ=9, obs-level union
print("\nConfig 4: occ=[5,6,7,9], obs-level union")
sub_nu = df_full2[df_full2['occ_1digit'].isin([5, 6, 7, 9]) & (df_full2['union_member'] == 0)].copy()
sub_u = df_full2[df_full2['occ_1digit'].isin([5, 6, 7, 9]) & (df_full2['union_member'] == 1)].copy()
quick_2step(sub_nu, 0.1520, name="BC_NU")
quick_2step(sub_u, 0.0992, name="BC_U")

# Config 5: Per-subsample step 1 b1+b2 instead of paper values
print("\nConfig 5: Per-subsample step 1 b1+b2, occ=[5,6,7], job-level union")
sub_nu = df[df['occ_1digit'].isin([5, 6, 7]) & (df['job_union'] == 0)].copy()
sub_u = df[df['occ_1digit'].isin([5, 6, 7]) & (df['job_union'] == 1)].copy()

# Run step 1 on each subsample
for label, sub_data in [("BC_NU", sub_nu), ("BC_U", sub_u)]:
    sub_data = sub_data.sort_values(['person_id', 'job_id', 'year']).copy()
    sub_data['cps_index'] = sub_data['year'].map(CPS_WAGE_INDEX)
    sub_data['log_wage_cps'] = sub_data['log_hourly_wage'] - np.log(sub_data['cps_index'])
    sub_data['tenure'] = sub_data['tenure_topel'].copy()
    sub_data['within'] = ((sub_data['person_id'] == sub_data['person_id'].shift(1)) &
                         (sub_data['job_id'] == sub_data['job_id'].shift(1)) &
                         (sub_data['year'] - sub_data['year'].shift(1) == 1))
    sub_data['d_lw'] = sub_data['log_wage_cps'] - sub_data['log_wage_cps'].shift(1)
    for k in [2, 3, 4]:
        sub_data[f'd_t{k}'] = sub_data['tenure']**k - (sub_data['tenure'] - 1)**k
        sub_data[f'd_x{k}'] = sub_data['experience']**k - (sub_data['experience'] - 1)**k

    for y in range(1969, 1984):
        col = f'year_{y}'
        if col not in sub_data.columns:
            sub_data[col] = (sub_data['year'] == y).astype(int)

    yr_cols = sorted([c for c in sub_data.columns if c.startswith('year_') and c != 'year'])
    d_yr = []
    for yc in yr_cols:
        dc = f'd_{yc}'
        sub_data[dc] = sub_data[yc].astype(float) - sub_data[yc].shift(1).astype(float)
        sub_data[dc] = sub_data[dc].fillna(0)
        d_yr.append(dc)

    fd = sub_data[sub_data['within']].dropna(subset=['d_lw']).copy()
    d_yr_use = [c for c in d_yr if fd[c].std() > 1e-10]
    if len(d_yr_use) > 1:
        d_yr_use = d_yr_use[1:]
    x_cols = ['d_t2', 'd_t3', 'd_t4', 'd_x2', 'd_x3', 'd_x4'] + d_yr_use
    y_s1 = fd['d_lw']
    X_s1 = sm.add_constant(fd[x_cols])
    valid = y_s1.notna() & X_s1.notna().all(axis=1)
    model = sm.OLS(y_s1[valid], X_s1[valid]).fit()
    b1b2_est = model.params['const']
    b1b2_se_est = model.bse['const']
    print(f"  {label} step1: b1+b2={b1b2_est:.4f} (se={b1b2_se_est:.4f}), N_wj={valid.sum()}")
    quick_2step(sub_data, b1b2_est, name=f"{label}_own_b1b2")

print("\nPaper targets: BC_NU beta_1=0.1066, BC_U beta_1=0.0592")
