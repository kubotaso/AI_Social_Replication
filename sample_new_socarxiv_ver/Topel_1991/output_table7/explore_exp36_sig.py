#!/usr/bin/env python3
"""Check significance matches for exp<=36 vs exp<=39."""
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
df['ct_x_censor'] = df['ct_obs'] * (1 - df['censor'])
df['ct_x_exp_sq'] = df['ct_obs'] * df['exp_sq']
df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

# OLS imputed CT
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

all_vars = ['lw', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct'] + control_vars

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

def stars(pv):
    return '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''

var_map = {
    'exp': 'exp', 'esq': 'exp_sq', 'ten': 'tenure_var',
    'ct': 'ct_obs', 'cen': 'ct_x_censor', 'imp': 'imp_ct',
    'esq_int': {2: 'ct_x_exp_sq', 4: 'imp_ct_x_exp_sq'},
    'ten_int': {2: 'ct_x_tenure', 4: 'imp_ct_x_tenure'},
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

    sig_pass = 0
    sig_total = 0
    for key, target_stars in gt_sig.items():
        vname, col = key
        if isinstance(var_map[vname], dict):
            model_var = var_map[vname].get(col)
        else:
            model_var = var_map[vname]
        if model_var is None:
            continue

        m = models[col]
        if model_var in m.pvalues.index:
            gen_stars = stars(m.pvalues[model_var])
            sig_total += 1
            match = gen_stars == target_stars
            if match:
                sig_pass += 1
            else:
                t = abs(m.params[model_var] / m.bse[model_var])
                print(f"  SIG MISS {vname:10s} col({col+1}): gen={gen_stars:>4s} target={target_stars:>4s} t={t:.3f}")

    print(f"\n  Significance: {sig_pass}/{sig_total}")
