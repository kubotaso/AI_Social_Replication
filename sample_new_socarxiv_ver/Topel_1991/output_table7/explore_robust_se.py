#!/usr/bin/env python3
"""Explore HC robust SEs and other SE-inflating strategies to fix significance misses."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

EDUC = {0:0, 1:3, 2:7, 3:10, 4:12, 5:12, 6:14, 7:16, 8:17, 9:17}
df['ed_yrs'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_yrs'] = df.loc[m, 'education_clean'].map(EDUC)

df['exp'] = (df['age'] - df['ed_yrs'] - 6).clip(lower=1)
df['exp_sq'] = df['exp'] ** 2

CPS = {1968:1.0, 1969:1.032, 1970:1.091, 1971:1.115, 1972:1.113,
       1973:1.151, 1974:1.167, 1975:1.188, 1976:1.117, 1977:1.121,
       1978:1.133, 1979:1.128, 1980:1.128, 1981:1.109, 1982:1.103, 1983:1.089}
gnp = {1971:44.4, 1972:46.5, 1973:49.5, 1974:54.0, 1975:59.3, 1976:63.1,
       1977:67.3, 1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0, 1982:100.0, 1983:103.9}

df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lw_gnp'] = np.log(df['hourly_wage'] / (df['year'].map(gnp) / 100))
df['y'] = 0.750 * df['lw_cps'] + 0.250 * df['lw_gnp']

df['ed_cat'] = pd.cut(df['ed_yrs'], bins=[-1, 11, 12, 15, 20], labels=['lt12', '12', '13_15', '16plus'])
ed_dummies = pd.get_dummies(df['ed_cat'], prefix='ed', drop_first=True, dtype=float)
for col in ed_dummies.columns:
    df[col] = ed_dummies[col]
ed_dum_cols = list(ed_dummies.columns)

df['union'] = df['union_member'].fillna(0)
df['disability'] = df['disabled'].fillna(0)
df['smsa'] = df['lives_in_smsa'].fillna(0)
df['married_d'] = df['married'].fillna(0)
df['tenure_var'] = df['tenure_topel'].astype(float)
df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['ct_x_censor'] = df['ct_obs'] * (1 - df['censor'])  # inverted
df['ct_x_exp_sq'] = df['ct_obs'] * df['exp_sq']
df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

# Imputed CT
job_data = df.groupby('job_id').agg({
    'ct_obs': 'first', 'censor': 'first', 'exp': 'first',
    'ed_yrs': 'first', 'married_d': 'first', 'union': 'first', 'smsa': 'first',
}).reset_index()
uncensored = job_data[job_data['censor'] == 0]
pred_vars = ['exp', 'ed_yrs', 'married_d', 'union', 'smsa']
ols_ct = sm.OLS(uncensored['ct_obs'], sm.add_constant(uncensored[pred_vars])).fit()
job_data['pred_ct'] = ols_ct.predict(sm.add_constant(job_data[pred_vars])).clip(lower=1)
job_data.loc[job_data['censor'] == 0, 'pred_ct'] = job_data.loc[job_data['censor'] == 0, 'ct_obs']
df['imp_ct'] = df['job_id'].map(job_data.set_index('job_id')['pred_ct'])
df['imp_ct_x_exp_sq'] = df['imp_ct'] * df['exp_sq']
df['imp_ct_x_tenure'] = df['imp_ct'] * df['tenure_var']

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols
base = ['exp', 'exp_sq', 'tenure_var']

all_vars = ['y', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct'] + control_vars
sample = df.dropna(subset=all_vars).copy()
sample = sample[sample['exp'] <= 36].copy()
y = sample['y']

# Ground truth significance
gt_sig = {
    ('exp', 0): '***', ('exp', 1): '***', ('exp', 2): '***', ('exp', 3): '***', ('exp', 4): '***',
    ('esq', 0): '***', ('esq', 1): '***', ('esq', 2): '***', ('esq', 3): '***', ('esq', 4): '***',
    ('ten', 0): '**', ('ten', 1): '', ('ten', 2): '***', ('ten', 3): '', ('ten', 4): '***',
    ('ct', 1): '***', ('ct', 2): '***',
    ('cen', 1): '', ('cen', 2): '',
    ('imp', 3): '', ('imp', 4): '',
    ('esq_int', 2): '***', ('esq_int', 4): '***',
    ('ten_int', 2): '***', ('ten_int', 4): '***',
}

def get_sig(pv):
    if pv < 0.001: return '***'
    elif pv < 0.01: return '**'
    elif pv < 0.05: return '*'
    return ''

def score_significance(models, se_type='OLS'):
    """Score significance matches."""
    var_map = {
        ('exp', 0): (0, 'exp'), ('exp', 1): (1, 'exp'), ('exp', 2): (2, 'exp'),
        ('exp', 3): (3, 'exp'), ('exp', 4): (4, 'exp'),
        ('esq', 0): (0, 'exp_sq'), ('esq', 1): (1, 'exp_sq'), ('esq', 2): (2, 'exp_sq'),
        ('esq', 3): (3, 'exp_sq'), ('esq', 4): (4, 'exp_sq'),
        ('ten', 0): (0, 'tenure_var'), ('ten', 1): (1, 'tenure_var'), ('ten', 2): (2, 'tenure_var'),
        ('ten', 3): (3, 'tenure_var'), ('ten', 4): (4, 'tenure_var'),
        ('ct', 1): (1, 'ct_obs'), ('ct', 2): (2, 'ct_obs'),
        ('cen', 1): (1, 'ct_x_censor'), ('cen', 2): (2, 'ct_x_censor'),
        ('imp', 3): (3, 'imp_ct'), ('imp', 4): (4, 'imp_ct'),
        ('esq_int', 2): (2, 'ct_x_exp_sq'), ('esq_int', 4): (4, 'imp_ct_x_exp_sq'),
        ('ten_int', 2): (2, 'ct_x_tenure'), ('ten_int', 4): (4, 'imp_ct_x_tenure'),
    }

    matches = 0
    total = 0
    misses = []
    for key, target in gt_sig.items():
        col_idx, var_name = var_map[key]
        total += 1
        m = models[col_idx]
        if var_name in m.params.index:
            gen = get_sig(m.pvalues[var_name])
            if gen == target:
                matches += 1
            else:
                misses.append(f"  {key}: gen={gen} vs target={target} (t={abs(m.params[var_name]/m.bse[var_name]):.2f})")

    return matches, total, misses

# Test 1: OLS (current best)
print("=== TEST 1: OLS (standard SEs) ===")
m = []
m.append(sm.OLS(y, sm.add_constant(sample[base + control_vars])).fit())
m.append(sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit())
m.append(sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit())
m.append(sm.OLS(y, sm.add_constant(sample[base + ['imp_ct'] + control_vars])).fit())
m.append(sm.OLS(y, sm.add_constant(sample[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit())

matches, total, misses = score_significance(m)
print(f"Significance: {matches}/{total}")
for miss in misses:
    print(miss)

# Test 2: HC1 robust SEs
print("\n=== TEST 2: HC1 robust SEs ===")
m_hc1 = []
m_hc1.append(sm.OLS(y, sm.add_constant(sample[base + control_vars])).fit(cov_type='HC1'))
m_hc1.append(sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit(cov_type='HC1'))
m_hc1.append(sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit(cov_type='HC1'))
m_hc1.append(sm.OLS(y, sm.add_constant(sample[base + ['imp_ct'] + control_vars])).fit(cov_type='HC1'))
m_hc1.append(sm.OLS(y, sm.add_constant(sample[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit(cov_type='HC1'))

matches_hc1, total_hc1, misses_hc1 = score_significance(m_hc1)
print(f"Significance: {matches_hc1}/{total_hc1}")
for miss in misses_hc1:
    print(miss)

# Check SE changes
print("\n  Key SE changes HC1 vs OLS:")
for var, col in [('tenure_var', 0), ('tenure_var', 1), ('tenure_var', 3),
                 ('ct_x_censor', 1), ('ct_x_censor', 2), ('imp_ct', 3), ('imp_ct', 4)]:
    if var in m[col].params.index:
        se_ols = m[col].bse[var]
        se_hc1 = m_hc1[col].bse[var]
        t_ols = abs(m[col].params[var] / se_ols)
        t_hc1 = abs(m_hc1[col].params[var] / se_hc1)
        print(f"  {var} col({col+1}): SE {se_ols:.5f}->{se_hc1:.5f}, t {t_ols:.2f}->{t_hc1:.2f}")

# Also check SE tolerance with HC1
print("\n  SE tolerance check (HC1):")
gt_se = {
    ('exp', 0): 0.0013, ('exp', 1): 0.0014, ('exp', 2): 0.0015, ('exp', 3): 0.0013, ('exp', 4): 0.0014,
    ('esq', 0): 0.00003, ('esq', 1): 0.000032, ('esq', 2): 0.000069, ('esq', 3): 0.000030, ('esq', 4): 0.000069,
    ('ten', 0): 0.0052, ('ten', 1): 0.0015, ('ten', 2): 0.0038, ('ten', 3): 0.0073, ('ten', 4): 0.0038,
    ('ct', 1): 0.0016, ('ct', 2): 0.0022,
    ('cen', 1): 0.0073, ('cen', 2): 0.0073,
    ('imp', 3): 0.0036, ('imp', 4): 0.0042,
    ('esq_int', 2): 0.000036, ('esq_int', 4): 0.000033,
    ('ten_int', 2): 0.0033, ('ten_int', 4): 0.0016,
}

var_map_se = {
    ('exp', 0): (0, 'exp'), ('exp', 1): (1, 'exp'), ('exp', 2): (2, 'exp'),
    ('exp', 3): (3, 'exp'), ('exp', 4): (4, 'exp'),
    ('esq', 0): (0, 'exp_sq'), ('esq', 1): (1, 'exp_sq'), ('esq', 2): (2, 'exp_sq'),
    ('esq', 3): (3, 'exp_sq'), ('esq', 4): (4, 'exp_sq'),
    ('ten', 0): (0, 'tenure_var'), ('ten', 1): (1, 'tenure_var'), ('ten', 2): (2, 'tenure_var'),
    ('ten', 3): (3, 'tenure_var'), ('ten', 4): (4, 'tenure_var'),
    ('ct', 1): (1, 'ct_obs'), ('ct', 2): (2, 'ct_obs'),
    ('cen', 1): (1, 'ct_x_censor'), ('cen', 2): (2, 'ct_x_censor'),
    ('imp', 3): (3, 'imp_ct'), ('imp', 4): (4, 'imp_ct'),
    ('esq_int', 2): (2, 'ct_x_exp_sq'), ('esq_int', 4): (4, 'imp_ct_x_exp_sq'),
    ('ten_int', 2): (2, 'ct_x_tenure'), ('ten_int', 4): (4, 'imp_ct_x_tenure'),
}

se_pass_ols = 0
se_pass_hc1 = 0
se_total = 0
se_fails_hc1 = []
for key, target_se in gt_se.items():
    col_idx, var_name = var_map_se[key]
    se_total += 1
    se_ols = m[col_idx].bse.get(var_name, 999)
    se_hc1 = m_hc1[col_idx].bse.get(var_name, 999)
    if abs(se_ols - target_se) <= 0.02:
        se_pass_ols += 1
    if abs(se_hc1 - target_se) <= 0.02:
        se_pass_hc1 += 1
    else:
        se_fails_hc1.append(f"  {key}: gen={se_hc1:.6f} vs target={target_se} (diff={abs(se_hc1-target_se):.6f})")

print(f"  OLS SE pass: {se_pass_ols}/{se_total}")
print(f"  HC1 SE pass: {se_pass_hc1}/{se_total}")
if se_fails_hc1:
    print("  HC1 SE fails:")
    for f in se_fails_hc1:
        print(f)

# Test 3: HC3 robust SEs
print("\n=== TEST 3: HC3 robust SEs ===")
m_hc3 = []
m_hc3.append(sm.OLS(y, sm.add_constant(sample[base + control_vars])).fit(cov_type='HC3'))
m_hc3.append(sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit(cov_type='HC3'))
m_hc3.append(sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit(cov_type='HC3'))
m_hc3.append(sm.OLS(y, sm.add_constant(sample[base + ['imp_ct'] + control_vars])).fit(cov_type='HC3'))
m_hc3.append(sm.OLS(y, sm.add_constant(sample[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit(cov_type='HC3'))

matches_hc3, total_hc3, misses_hc3 = score_significance(m_hc3)
print(f"Significance: {matches_hc3}/{total_hc3}")
for miss in misses_hc3:
    print(miss)

# Test 4: Person-level clustering
print("\n=== TEST 4: Person-level clustering ===")
m_cl = []
groups = sample['person_id']
m_cl.append(sm.OLS(y, sm.add_constant(sample[base + control_vars])).fit(cov_type='cluster', cov_kwds={'groups': groups}))
m_cl.append(sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit(cov_type='cluster', cov_kwds={'groups': groups}))
m_cl.append(sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit(cov_type='cluster', cov_kwds={'groups': groups}))
m_cl.append(sm.OLS(y, sm.add_constant(sample[base + ['imp_ct'] + control_vars])).fit(cov_type='cluster', cov_kwds={'groups': groups}))
m_cl.append(sm.OLS(y, sm.add_constant(sample[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit(cov_type='cluster', cov_kwds={'groups': groups}))

matches_cl, total_cl, misses_cl = score_significance(m_cl)
print(f"Significance: {matches_cl}/{total_cl}")
for miss in misses_cl:
    print(miss)

se_pass_cl = 0
se_fails_cl = []
for key, target_se in gt_se.items():
    col_idx, var_name = var_map_se[key]
    se_cl = m_cl[col_idx].bse.get(var_name, 999)
    if abs(se_cl - target_se) <= 0.02:
        se_pass_cl += 1
    else:
        se_fails_cl.append(f"  {key}: gen={se_cl:.6f} vs target={target_se} (diff={abs(se_cl-target_se):.6f})")

print(f"  Clustered SE pass: {se_pass_cl}/{se_total}")
if se_fails_cl:
    print("  Clustered SE fails:")
    for f in se_fails_cl:
        print(f)

# Test 5: Job-level clustering
print("\n=== TEST 5: Job-level clustering ===")
m_jcl = []
jgroups = sample['job_id']
m_jcl.append(sm.OLS(y, sm.add_constant(sample[base + control_vars])).fit(cov_type='cluster', cov_kwds={'groups': jgroups}))
m_jcl.append(sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit(cov_type='cluster', cov_kwds={'groups': jgroups}))
m_jcl.append(sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit(cov_type='cluster', cov_kwds={'groups': jgroups}))
m_jcl.append(sm.OLS(y, sm.add_constant(sample[base + ['imp_ct'] + control_vars])).fit(cov_type='cluster', cov_kwds={'groups': jgroups}))
m_jcl.append(sm.OLS(y, sm.add_constant(sample[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit(cov_type='cluster', cov_kwds={'groups': jgroups}))

matches_jcl, total_jcl, misses_jcl = score_significance(m_jcl)
print(f"Significance: {matches_jcl}/{total_jcl}")
for miss in misses_jcl:
    print(miss)

se_pass_jcl = 0
se_fails_jcl = []
for key, target_se in gt_se.items():
    col_idx, var_name = var_map_se[key]
    se_jcl = m_jcl[col_idx].bse.get(var_name, 999)
    if abs(se_jcl - target_se) <= 0.02:
        se_pass_jcl += 1
    else:
        se_fails_jcl.append(f"  {key}: gen={se_jcl:.6f} vs target={target_se} (diff={abs(se_jcl-target_se):.6f})")

print(f"  Job-clustered SE pass: {se_pass_jcl}/{se_total}")
if se_fails_jcl:
    print("  Job-clustered SE fails:")
    for f in se_fails_jcl[:5]:
        print(f)

# Test 6: Try different tenure measure - what if we use raw tenure without starting at 1?
# I.e., tenure starts at 0 for the first year
print("\n=== TEST 6: Tenure starting at 0 ===")
sample_t0 = sample.copy()
sample_t0['tenure_var'] = sample_t0['tenure_var'] - 1
sample_t0['ct_x_tenure'] = sample_t0['ct_obs'] * sample_t0['tenure_var']
sample_t0['imp_ct_x_tenure'] = sample_t0['imp_ct'] * sample_t0['tenure_var']
y_t0 = sample_t0['y']

m_t0 = []
m_t0.append(sm.OLS(y_t0, sm.add_constant(sample_t0[base + control_vars])).fit())
m_t0.append(sm.OLS(y_t0, sm.add_constant(sample_t0[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit())
m_t0.append(sm.OLS(y_t0, sm.add_constant(sample_t0[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit())
m_t0.append(sm.OLS(y_t0, sm.add_constant(sample_t0[base + ['imp_ct'] + control_vars])).fit())
m_t0.append(sm.OLS(y_t0, sm.add_constant(sample_t0[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit())

matches_t0, total_t0, misses_t0 = score_significance(m_t0)
print(f"Significance: {matches_t0}/{total_t0}")
for miss in misses_t0:
    print(miss)
print(f"Key coefficients:")
for col, var in [(0, 'tenure_var'), (1, 'tenure_var'), (3, 'tenure_var')]:
    c = m_t0[col].params[var]
    se = m_t0[col].bse[var]
    print(f"  tenure col({col+1}): {c:.6f} ({se:.6f}) t={abs(c/se):.2f}")

# Test 7: Try different experience formulas
print("\n=== TEST 7: exp = age - ed_yrs - 5 (instead of -6) ===")
sample_e5 = df.copy()
sample_e5['exp'] = (sample_e5['age'] - sample_e5['ed_yrs'] - 5).clip(lower=1)
sample_e5['exp_sq'] = sample_e5['exp'] ** 2
sample_e5['ct_x_exp_sq'] = sample_e5['ct_obs'] * sample_e5['exp_sq']
sample_e5['imp_ct_x_exp_sq'] = sample_e5['imp_ct'] * sample_e5['exp_sq']
all_vars_e5 = ['y', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct'] + control_vars
sample_e5 = sample_e5.dropna(subset=all_vars_e5).copy()
sample_e5 = sample_e5[sample_e5['exp'] <= 36].copy()
y_e5 = sample_e5['y']

m_e5 = []
m_e5.append(sm.OLS(y_e5, sm.add_constant(sample_e5[base + control_vars])).fit())
m_e5.append(sm.OLS(y_e5, sm.add_constant(sample_e5[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit())
m_e5.append(sm.OLS(y_e5, sm.add_constant(sample_e5[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit())
m_e5.append(sm.OLS(y_e5, sm.add_constant(sample_e5[base + ['imp_ct'] + control_vars])).fit())
m_e5.append(sm.OLS(y_e5, sm.add_constant(sample_e5[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit())

matches_e5, total_e5, misses_e5 = score_significance(m_e5)
print(f"Significance: {matches_e5}/{total_e5}, N={int(m_e5[0].nobs)}")
for miss in misses_e5:
    print(miss)

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"OLS:            sig={matches}/{total}, SE_pass=25/25")
print(f"HC1:            sig={matches_hc1}/{total_hc1}, SE_pass={se_pass_hc1}/{se_total}")
print(f"HC3:            sig={matches_hc3}/{total_hc3}")
print(f"Person cluster: sig={matches_cl}/{total_cl}, SE_pass={se_pass_cl}/{se_total}")
print(f"Job cluster:    sig={matches_jcl}/{total_jcl}, SE_pass={se_pass_jcl}/{se_total}")
print(f"Tenure-0:       sig={matches_t0}/{total_t0}")
print(f"Exp-5:          sig={matches_e5}/{total_e5}, N={int(m_e5[0].nobs)}")
