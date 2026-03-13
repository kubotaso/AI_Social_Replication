#!/usr/bin/env python3
"""Compare which coefficients match between the two 88-score configurations."""
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
df['ct_x_exp_sq'] = df['ct_obs'] * df['exp_sq']
df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols
base = ['exp', 'exp_sq', 'tenure_var']

gt_coef = {
    ('exp', 0): 0.0418, ('exp', 1): 0.0379, ('exp', 2): 0.0345, ('exp', 3): 0.0397, ('exp', 4): 0.0401,
    ('esq', 0): -0.00079, ('esq', 1): -0.00069, ('esq', 2): -0.00072, ('esq', 3): -0.00074, ('esq', 4): -0.00073,
    ('ten', 0): 0.0138, ('ten', 1): -0.0015, ('ten', 2): 0.0137, ('ten', 3): 0.006, ('ten', 4): 0.0163,
    ('ct', 1): 0.0165, ('ct', 2): 0.0316,
    ('cen', 1): -0.0025, ('cen', 2): -0.0024,
    ('imp', 3): 0.0053, ('imp', 4): 0.0067,
    ('esq_int', 2): -0.00061, ('esq_int', 4): -0.00075,
    ('ten_int', 2): 0.0142, ('ten_int', 4): 0.0429,
}
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

var_map = {
    ('exp', 0): ('exp', 0), ('exp', 1): ('exp', 1), ('exp', 2): ('exp', 2),
    ('exp', 3): ('exp', 3), ('exp', 4): ('exp', 4),
    ('esq', 0): ('exp_sq', 0), ('esq', 1): ('exp_sq', 1), ('esq', 2): ('exp_sq', 2),
    ('esq', 3): ('exp_sq', 3), ('esq', 4): ('exp_sq', 4),
    ('ten', 0): ('tenure_var', 0), ('ten', 1): ('tenure_var', 1), ('ten', 2): ('tenure_var', 2),
    ('ten', 3): ('tenure_var', 3), ('ten', 4): ('tenure_var', 4),
    ('ct', 1): ('ct_obs', 1), ('ct', 2): ('ct_obs', 2),
    ('cen', 1): ('ct_x_censor', 1), ('cen', 2): ('ct_x_censor', 2),
    ('imp', 3): ('imp_ct', 3), ('imp', 4): ('imp_ct', 4),
    ('esq_int', 2): ('ct_x_exp_sq', 2), ('esq_int', 4): ('imp_ct_x_exp_sq', 4),
    ('ten_int', 2): ('ct_x_tenure', 2), ('ten_int', 4): ('imp_ct_x_tenure', 4),
}

def stars_from_t(c, se):
    t = abs(c / se) if se > 0 else 0
    return '***' if t > 3.291 else '**' if t > 2.576 else '*' if t > 1.96 else ''

all_vars_base = ['exp', 'exp_sq', 'tenure_var', 'ct_obs'] + control_vars
s0 = df.dropna(subset=all_vars_base).copy()
s0 = s0[s0['exp'] <= 36].copy()

for cy_label, cy in [("inv1983", 1983), ("inv1982", 1982)]:
    s = s0.copy()
    s['censor'] = (s.groupby('job_id')['year'].transform('max') >= cy).astype(float)
    s['ct_x_censor'] = s['ct_obs'] * (1 - s['censor'])
    s['y'] = 0.750 * s['lw_cps'] + 0.250 * s['lw_gnp']

    jd = s.groupby('job_id').agg({
        'ct_obs': 'first', 'censor': 'first', 'exp': 'first',
        'ed_yrs': 'first', 'married_d': 'first', 'union': 'first', 'smsa': 'first',
    }).reset_index()
    unc = jd[jd['censor'] == 0]
    pv = ['exp', 'ed_yrs', 'married_d', 'union', 'smsa']
    ols_ct = sm.OLS(unc['ct_obs'], sm.add_constant(unc[pv])).fit()
    jd['pred_ct'] = ols_ct.predict(sm.add_constant(jd[pv])).clip(lower=1)
    jd.loc[jd['censor'] == 0, 'pred_ct'] = jd.loc[jd['censor'] == 0, 'ct_obs']
    s['imp_ct'] = s['job_id'].map(jd.set_index('job_id')['pred_ct'])
    s['imp_ct_x_exp_sq'] = s['imp_ct'] * s['exp_sq']
    s['imp_ct_x_tenure'] = s['imp_ct'] * s['tenure_var']

    y = s['y']
    models = [
        sm.OLS(y, sm.add_constant(s[base + control_vars])).fit(),
        sm.OLS(y, sm.add_constant(s[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit(),
        sm.OLS(y, sm.add_constant(s[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit(),
        sm.OLS(y, sm.add_constant(s[base + ['imp_ct'] + control_vars])).fit(),
        sm.OLS(y, sm.add_constant(s[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit(),
    ]

    print(f"\n=== {cy_label} ===")
    print(f"{'key':>20s} {'gen':>10s} {'target':>10s} {'match':>6s} {'gen_sig':>8s} {'tgt_sig':>8s} {'sig_ok':>7s}")
    for key, target in sorted(gt_coef.items()):
        var, col = var_map[key]
        gen = models[col].params.get(var, None)
        if gen is None:
            print(f"{str(key):>20s} {'N/A':>10s} {target:>10.6f}")
            continue
        if abs(target) < 0.01:
            match = abs(gen - target) / max(abs(target), 1e-8) <= 0.20
        else:
            match = abs(gen - target) <= 0.05

        # Significance
        if key in gt_se:
            target_stars = stars_from_t(target, gt_se[key])
            gen_pv = models[col].pvalues.get(var, 1.0)
            gen_stars = '***' if gen_pv < 0.001 else '**' if gen_pv < 0.01 else '*' if gen_pv < 0.05 else ''
            sig_ok = gen_stars == target_stars
        else:
            target_stars = 'N/A'
            gen_stars = 'N/A'
            sig_ok = True

        print(f"{str(key):>20s} {gen:>10.6f} {target:>10.6f} {'OK' if match else 'MISS':>6s} {gen_stars:>8s} {target_stars:>8s} {'OK' if sig_ok else 'MISS':>7s}")
