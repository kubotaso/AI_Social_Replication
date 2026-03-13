#!/usr/bin/env python3
"""Test combined fixes: inverted censor + scaled interactions + tenure starting at 0."""
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

df['ed_cat'] = pd.cut(df['ed_yrs'], bins=[-1, 11, 12, 15, 20], labels=['lt12', '12', '13_15', '16plus'])
ed_dummies = pd.get_dummies(df['ed_cat'], prefix='ed', drop_first=True, dtype=float)
for col in ed_dummies.columns:
    df[col] = ed_dummies[col]
ed_dum_cols = list(ed_dummies.columns)

CPS = {1968:1.0, 1969:1.032, 1970:1.091, 1971:1.115, 1972:1.113,
       1973:1.151, 1974:1.167, 1975:1.188, 1976:1.117, 1977:1.121,
       1978:1.133, 1979:1.128, 1980:1.128, 1981:1.109, 1982:1.103, 1983:1.089}
gnp = {1971:44.4, 1972:46.5, 1973:49.5, 1974:54.0, 1975:59.3, 1976:63.1,
       1977:67.3, 1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0, 1982:100.0, 1983:103.9}

df['union'] = df['union_member'].fillna(0)
df['disability'] = df['disabled'].fillna(0)
df['smsa'] = df['lives_in_smsa'].fillna(0)
df['married_d'] = df['married'].fillna(0)

ALPHA = 0.750
df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lw_gnp'] = np.log(df['hourly_wage'] / (df['year'].map(gnp) / 100))
df['lw'] = ALPHA * df['lw_cps'] + (1 - ALPHA) * df['lw_gnp']

# Tenure (try both starting at 0 and 1)
df['tenure_1'] = df['tenure_topel'].astype(float)  # starts at 1
df['tenure_0'] = df['tenure_topel'].astype(float) - 1  # starts at 0

df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['uncensor'] = 1 - df['censor']

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

all_vars = ['lw', 'exp', 'exp_sq', 'tenure_1', 'ct_obs', 'censor'] + control_vars
base = df.dropna(subset=all_vars).copy()
base = base[base['exp'] <= 39].copy()

print(f"Sample: N={len(base)}")

# Ground truth
gt = {
    'exp': [0.0418, 0.0379, 0.0345, 0.0397, 0.0401],
    'exp_sq': [-0.00079, -0.00069, -0.00072, -0.00074, -0.00073],
    'tenure': [0.0138, -0.0015, 0.0137, 0.0060, 0.0163],
    'ct': [None, 0.0165, 0.0316, None, None],
    'censor_int': [None, -0.0025, -0.0024, None, None],
    'esq_int': [None, None, -0.00061, None, -0.00075],
    'ten_int': [None, None, 0.0142, None, 0.0429],
    'r2': [0.422, 0.428, 0.432, 0.433, 0.435],
}

# Test configuration combinations
configs = [
    # (label, tenure_col, censor_col, exp_sq_interaction_formula)
    ("baseline (current)", 'tenure_1', 'censor', 'ct * exp_sq'),
    ("inv_censor", 'tenure_1', 'uncensor', 'ct * exp_sq'),
    ("tenure_0", 'tenure_0', 'censor', 'ct * exp_sq'),
    ("tenure_0 + inv_censor", 'tenure_0', 'uncensor', 'ct * exp_sq'),
    ("exp_sq/100 interaction", 'tenure_1', 'censor', 'ct * exp_sq/100'),
    ("inv_censor + esq/100", 'tenure_1', 'uncensor', 'ct * exp_sq/100'),
    ("tenure_0 + inv_censor + esq/100", 'tenure_0', 'uncensor', 'ct * exp_sq/100'),
]

print("\n" + "="*100)
print("TESTING COLUMN 3 (UNRESTRICTED, OBSERVED CT)")
print("="*100)
print(f"{'Config':40s} {'R2':>6s} {'exp_sq':>10s} {'tenure':>10s} {'ct':>10s} {'cen_int':>10s} {'esq_int':>12s} {'ten_int':>10s}")

for label, ten_col, cen_col, esq_formula in configs:
    s = base.copy()
    s['t'] = s[ten_col]
    s['ct_x_cen'] = s['ct_obs'] * s[cen_col]

    if 'esq/100' in esq_formula:
        s['ct_x_esq'] = s['ct_obs'] * s['exp_sq'] / 100
    else:
        s['ct_x_esq'] = s['ct_obs'] * s['exp_sq']

    s['ct_x_t'] = s['ct_obs'] * s[ten_col]

    X = sm.add_constant(s[['exp', 'exp_sq', 't', 'ct_obs', 'ct_x_cen', 'ct_x_esq', 'ct_x_t'] + control_vars])
    m = sm.OLS(s['lw'], X).fit()

    esq_coef = m.params['ct_x_esq']
    ten_coef = m.params['ct_x_t']

    print(f"{label:40s} {m.rsquared:6.4f} {m.params['exp_sq']:10.6f} {m.params['t']:10.5f} {m.params['ct_obs']:10.5f} {m.params['ct_x_cen']:10.5f} {esq_coef:12.8f} {ten_coef:10.5f}")

# Also test Column 2 (restricted) with inverted censor
print("\n" + "="*100)
print("TESTING COLUMN 2 (RESTRICTED, OBSERVED CT)")
print("="*100)

for label, ten_col, cen_col in [
    ("baseline", 'tenure_1', 'censor'),
    ("inv_censor", 'tenure_1', 'uncensor'),
    ("tenure_0", 'tenure_0', 'censor'),
    ("tenure_0 + inv_censor", 'tenure_0', 'uncensor'),
]:
    s = base.copy()
    s['t'] = s[ten_col]
    s['ct_x_cen'] = s['ct_obs'] * s[cen_col]

    X = sm.add_constant(s[['exp', 'exp_sq', 't', 'ct_obs', 'ct_x_cen'] + control_vars])
    m = sm.OLS(s['lw'], X).fit()

    print(f"{label:30s}: tenure={m.params['t']:10.5f} (SE={m.bse['t']:.5f}), ct={m.params['ct_obs']:10.5f}, cen={m.params['ct_x_cen']:10.5f}, R2={m.rsquared:.4f}")

# Test Column 1 with tenure_0
print("\n" + "="*100)
print("TESTING COLUMN 1 (BASELINE)")
print("="*100)

for label, ten_col in [("tenure_1", 'tenure_1'), ("tenure_0", 'tenure_0')]:
    s = base.copy()
    s['t'] = s[ten_col]
    X = sm.add_constant(s[['exp', 'exp_sq', 't'] + control_vars])
    m = sm.OLS(s['lw'], X).fit()
    print(f"{label:15s}: exp={m.params['exp']:.5f} exp_sq={m.params['exp_sq']:.6f} tenure={m.params['t']:.5f} (SE={m.bse['t']:.5f}) R2={m.rsquared:.4f}")

# Full 5-column test with best config so far
print("\n" + "="*100)
print("FULL 5-COLUMN TEST: tenure_0 + inv_censor")
print("="*100)

s = base.copy()
s['t'] = s['tenure_0']
s['ct_x_cen'] = s['ct_obs'] * s['uncensor']
s['ct_x_esq'] = s['ct_obs'] * s['exp_sq']
s['ct_x_t'] = s['ct_obs'] * s['tenure_0']

# Imputed CT (simple OLS prediction)
job_first = s.groupby('job_id').first().reset_index()
uncensored = job_first[job_first['censor'] == 0]
pred_vars = ['exp', 'ed_yrs', 'married_d', 'union', 'smsa']
X_unc = sm.add_constant(uncensored[pred_vars])
y_unc = uncensored['ct_obs']
pred_model = sm.OLS(y_unc, X_unc).fit()
X_all = sm.add_constant(s[pred_vars])
s['imp_ct'] = pred_model.predict(X_all)
s.loc[s['censor'] == 0, 'imp_ct'] = s.loc[s['censor'] == 0, 'ct_obs']
s['imp_ct_x_esq'] = s['imp_ct'] * s['exp_sq']
s['imp_ct_x_t'] = s['imp_ct'] * s['tenure_0']

m1 = sm.OLS(s['lw'], sm.add_constant(s[['exp', 'exp_sq', 't'] + control_vars])).fit()
m2 = sm.OLS(s['lw'], sm.add_constant(s[['exp', 'exp_sq', 't', 'ct_obs', 'ct_x_cen'] + control_vars])).fit()
m3 = sm.OLS(s['lw'], sm.add_constant(s[['exp', 'exp_sq', 't', 'ct_obs', 'ct_x_cen', 'ct_x_esq', 'ct_x_t'] + control_vars])).fit()
m4 = sm.OLS(s['lw'], sm.add_constant(s[['exp', 'exp_sq', 't', 'imp_ct'] + control_vars])).fit()
m5 = sm.OLS(s['lw'], sm.add_constant(s[['exp', 'exp_sq', 't', 'imp_ct', 'imp_ct_x_esq', 'imp_ct_x_t'] + control_vars])).fit()

models = [m1, m2, m3, m4, m5]

# Score each coefficient
coef_checks = [
    ('exp', [0,1,2,3,4], 'exp'),
    ('exp_sq', [0,1,2,3,4], 'exp_sq'),
    ('tenure', [0,1,2,3,4], 't'),
    ('ct', [1,2], 'ct_obs'),
    ('censor_int', [1,2], 'ct_x_cen'),
    ('esq_int', [2], 'ct_x_esq'),
    ('ten_int', [2], 'ct_x_t'),
    ('imp_ct', [3,4], 'imp_ct'),
    ('imp_esq_int', [4], 'imp_ct_x_esq'),
    ('imp_ten_int', [4], 'imp_ct_x_t'),
]

gt_map = {
    'exp': gt['exp'], 'exp_sq': gt['exp_sq'], 'tenure': gt['tenure'],
    'ct': [None, 0.0165, 0.0316, None, None],
    'censor_int': [None, -0.0025, -0.0024, None, None],
    'esq_int': [None, None, -0.00061, None, None],
    'ten_int': [None, None, 0.0142, None, None],
    'imp_ct': [None, None, None, 0.0053, 0.0067],
    'imp_esq_int': [None, None, None, None, -0.00075],
    'imp_ten_int': [None, None, None, None, 0.0429],
}

for name, cols, var in coef_checks:
    for c in cols:
        target = gt_map[name][c]
        if target is None:
            continue
        gen = models[c].params.get(var, None)
        if gen is not None:
            if abs(target) < 0.01:
                match = abs(gen - target) / max(abs(target), 1e-8) <= 0.20
            else:
                match = abs(gen - target) <= 0.05
            status = "MATCH" if match else "MISS"
            print(f"  {name} col({c+1}): gen={gen:.7f} target={target:>10.6f} {status}")

for i, m in enumerate(models):
    r2_match = abs(m.rsquared - gt['r2'][i]) <= 0.02
    print(f"  R2 col({i+1}): {m.rsquared:.4f} vs {gt['r2'][i]} {'MATCH' if r2_match else 'MISS'}")
