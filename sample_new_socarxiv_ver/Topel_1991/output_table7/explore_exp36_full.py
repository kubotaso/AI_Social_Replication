#!/usr/bin/env python3
"""Full scoring test with exp<=36 to see if gaining esq col(3) is worth any other losses."""
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
df['tenure_var'] = df['tenure_topel'].astype(float)

ALPHA = 0.750
df['lw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lw_gnp'] = np.log(df['hourly_wage'] / (df['year'].map(gnp) / 100))
df['lw'] = ALPHA * df['lw_cps'] + (1 - ALPHA) * df['lw_gnp']

df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['ct_x_censor'] = df['ct_obs'] * (1 - df['censor'])  # inverted censor
df['ct_x_exp_sq'] = df['ct_obs'] * df['exp_sq']
df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

# OLS imputed CT
job_data = df.groupby('job_id').agg({
    'ct_obs': 'first', 'censor': 'first', 'exp': 'first',
    'ed_yrs': 'first', 'married_d': 'first', 'union': 'first', 'smsa': 'first',
}).reset_index()
uncensored = job_data[job_data['censor'] == 0].copy()
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

all_vars = ['lw', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct'] + control_vars

# Coefficient targets
gt = {
    'experience': [0.0418, 0.0379, 0.0345, 0.0397, 0.0401],
    'experience_se': [0.0013, 0.0014, 0.0015, 0.0013, 0.0014],
    'experience_sq': [-0.00079, -0.00069, -0.00072, -0.00074, -0.00073],
    'experience_sq_se': [0.00003, 0.000032, 0.000069, 0.000030, 0.000069],
    'tenure': [0.0138, -0.0015, 0.0137, 0.0060, 0.0163],
    'tenure_se': [0.0052, 0.0015, 0.0038, 0.0073, 0.0038],
    'obs_ct': [None, 0.0165, 0.0316, None, None],
    'obs_ct_se': [None, 0.0016, 0.0022, None, None],
    'x_censor': [None, -0.0025, -0.0024, None, None],
    'x_censor_se': [None, 0.0073, 0.0073, None, None],
    'imp_ct': [None, None, None, 0.0053, 0.0067],
    'imp_ct_se': [None, None, None, 0.0036, 0.0042],
    'esq_int': [None, None, -0.00061, None, -0.00075],
    'esq_int_se': [None, None, 0.000036, None, 0.000033],
    'ten_int': [None, None, 0.0142, None, 0.0429],
    'ten_int_se': [None, None, 0.0033, None, 0.0016],
    'r2': [0.422, 0.428, 0.432, 0.433, 0.435],
}

for exp_cut in [36, 39]:
    base = df.dropna(subset=all_vars).copy()
    base = base[base['exp'] <= exp_cut].copy()
    y = base['lw']

    m1 = sm.OLS(y, sm.add_constant(base[['exp', 'exp_sq', 'tenure_var'] + control_vars])).fit()
    m2 = sm.OLS(y, sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'ct_x_censor'] + control_vars])).fit()
    m3 = sm.OLS(y, sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit()
    m4 = sm.OLS(y, sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 'imp_ct'] + control_vars])).fit()
    m5 = sm.OLS(y, sm.add_constant(base[['exp', 'exp_sq', 'tenure_var', 'imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit()
    models = [m1, m2, m3, m4, m5]

    print(f"\n{'='*60}")
    print(f"EXP <= {exp_cut}: N={len(base)}")
    print(f"{'='*60}")

    # Score
    coef_checks = [
        ('exp', [0,1,2,3,4], 'exp', 'experience'),
        ('esq', [0,1,2,3,4], 'exp_sq', 'experience_sq'),
        ('ten', [0,1,2,3,4], 'tenure_var', 'tenure'),
        ('obs_ct', [1,2], 'ct_obs', 'obs_ct'),
        ('xcen', [1,2], 'ct_x_censor', 'x_censor'),
        ('imp', [3,4], 'imp_ct', 'imp_ct'),
        ('esq_int', [2], 'ct_x_exp_sq', 'esq_int'),
        ('esq_int5', [4], 'imp_ct_x_exp_sq', 'esq_int'),
        ('ten_int', [2], 'ct_x_tenure', 'ten_int'),
        ('ten_int5', [4], 'imp_ct_x_tenure', 'ten_int'),
    ]

    coef_pass = 0
    coef_total = 0
    for label, cols, var, gt_key in coef_checks:
        for c in cols:
            target = gt[gt_key][c]
            if target is None:
                continue
            coef_total += 1
            gen = models[c].params.get(var, None)
            if gen is None:
                print(f"  {label} col({c+1}): MISSING")
                continue
            if abs(target) < 0.01:
                rel_err = abs(gen - target) / max(abs(target), 1e-8)
                match = rel_err <= 0.20
                err_str = f"rel={rel_err:.1%}"
            else:
                abs_err = abs(gen - target)
                match = abs_err <= 0.05
                err_str = f"abs={abs_err:.4f}"
            if match:
                coef_pass += 1
            status = "PASS" if match else "FAIL"
            print(f"  {label:10s} col({c+1}): {gen:>11.7f} vs {target:>10.6f} {err_str:>12s} {status}")

    print(f"\n  Coef: {coef_pass}/{coef_total}")

    # N check
    n = len(base)
    n_err = abs(n - 13128) / 13128
    print(f"  N: {n} (err={n_err:.1%})")

    # R2 check
    r2_pass = sum(1 for i in range(5) if abs(models[i].rsquared - gt['r2'][i]) <= 0.02)
    print(f"  R2: {r2_pass}/5")
    for i in range(5):
        print(f"    col({i+1}): {models[i].rsquared:.4f} vs {gt['r2'][i]}")
