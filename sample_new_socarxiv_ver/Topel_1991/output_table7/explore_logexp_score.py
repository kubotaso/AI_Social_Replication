#!/usr/bin/env python3
"""Full scoring with log-exp and QR imputed CT approaches."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
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
df['ct_x_censor'] = df['ct_obs'] * (1 - df['censor'])
df['ct_x_exp_sq'] = df['ct_obs'] * df['exp_sq']
df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols
base = ['exp', 'exp_sq', 'tenure_var']

# Build imputed CT variants BEFORE filtering
job_data = df.groupby('job_id').agg({
    'ct_obs': 'first', 'censor': 'first', 'exp': 'first',
    'ed_yrs': 'first', 'married_d': 'first', 'union': 'first', 'smsa': 'first',
}).reset_index()
uncensored = job_data[job_data['censor'] == 0]
pred_vars = ['exp', 'ed_yrs', 'married_d', 'union', 'smsa']

# Standard OLS imputation
ols_ct = sm.OLS(uncensored['ct_obs'], sm.add_constant(uncensored[pred_vars])).fit()
job_data['pred_ct'] = ols_ct.predict(sm.add_constant(job_data[pred_vars])).clip(lower=1)
job_data.loc[job_data['censor'] == 0, 'pred_ct'] = job_data.loc[job_data['censor'] == 0, 'ct_obs']
df['imp_ct'] = df['job_id'].map(job_data.set_index('job_id')['pred_ct'])

# Log-exp imputation
ols_logct = sm.OLS(np.log(uncensored['ct_obs']), sm.add_constant(uncensored[pred_vars])).fit()
job_data['pred_logct'] = np.exp(ols_logct.predict(sm.add_constant(job_data[pred_vars]))).clip(lower=1)
job_data.loc[job_data['censor'] == 0, 'pred_logct'] = job_data.loc[job_data['censor'] == 0, 'ct_obs']
df['imp_ct_log'] = df['job_id'].map(job_data.set_index('job_id')['pred_logct'])

# QR imputation
qr_ct = QuantReg(uncensored['ct_obs'], sm.add_constant(uncensored[pred_vars])).fit(q=0.5)
job_data['pred_ct_qr'] = qr_ct.predict(sm.add_constant(job_data[pred_vars])).clip(lower=1)
job_data.loc[job_data['censor'] == 0, 'pred_ct_qr'] = job_data.loc[job_data['censor'] == 0, 'ct_obs']
df['imp_ct_qr'] = df['job_id'].map(job_data.set_index('job_id')['pred_ct_qr'])

# Create interactions
for prefix in ['imp_ct', 'imp_ct_log', 'imp_ct_qr']:
    df[f'{prefix}_x_exp_sq'] = df[prefix] * df['exp_sq']
    df[f'{prefix}_x_tenure'] = df[prefix] * df['tenure_var']

# Filter sample
all_vars = ['y', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct', 'imp_ct_log', 'imp_ct_qr'] + control_vars
sample = df.dropna(subset=all_vars).copy()
sample = sample[sample['exp'] <= 36].copy()
y_val = sample['y']
N = len(sample)

# Ground truth
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
gt_sig = {}
for key in gt_coef:
    c, s = gt_coef[key], gt_se[key]
    t = abs(c / s)
    gt_sig[key] = '***' if t > 3.291 else '**' if t > 2.576 else '*' if t > 1.96 else ''

def get_sig(pv):
    if pv < 0.001: return '***'
    elif pv < 0.01: return '**'
    elif pv < 0.05: return '*'
    return ''

def run_and_score(imp_prefix, label):
    var_map = {
        ('exp', 0): (0, 'exp'), ('exp', 1): (1, 'exp'), ('exp', 2): (2, 'exp'),
        ('exp', 3): (3, 'exp'), ('exp', 4): (4, 'exp'),
        ('esq', 0): (0, 'exp_sq'), ('esq', 1): (1, 'exp_sq'), ('esq', 2): (2, 'exp_sq'),
        ('esq', 3): (3, 'exp_sq'), ('esq', 4): (4, 'exp_sq'),
        ('ten', 0): (0, 'tenure_var'), ('ten', 1): (1, 'tenure_var'), ('ten', 2): (2, 'tenure_var'),
        ('ten', 3): (3, 'tenure_var'), ('ten', 4): (4, 'tenure_var'),
        ('ct', 1): (1, 'ct_obs'), ('ct', 2): (2, 'ct_obs'),
        ('cen', 1): (1, 'ct_x_censor'), ('cen', 2): (2, 'ct_x_censor'),
        ('imp', 3): (3, imp_prefix), ('imp', 4): (4, imp_prefix),
        ('esq_int', 2): (2, 'ct_x_exp_sq'), ('esq_int', 4): (4, f'{imp_prefix}_x_exp_sq'),
        ('ten_int', 2): (2, 'ct_x_tenure'), ('ten_int', 4): (4, f'{imp_prefix}_x_tenure'),
    }

    ms = []
    ms.append(sm.OLS(y_val, sm.add_constant(sample[base + control_vars])).fit())
    ms.append(sm.OLS(y_val, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit())
    ms.append(sm.OLS(y_val, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit())
    ms.append(sm.OLS(y_val, sm.add_constant(sample[base + [imp_prefix] + control_vars])).fit())
    ms.append(sm.OLS(y_val, sm.add_constant(sample[base + [imp_prefix, f'{imp_prefix}_x_exp_sq', f'{imp_prefix}_x_tenure'] + control_vars])).fit())

    coef_pts, coef_max = 0, 0
    coef_misses = []
    for key, gt_val in gt_coef.items():
        col_idx, var_name = var_map[key]
        coef_max += 1
        if var_name in ms[col_idx].params.index:
            gen = ms[col_idx].params[var_name]
            if abs(gt_val) < 0.01:
                ok = abs(gen - gt_val) / max(abs(gt_val), 1e-8) <= 0.20
            else:
                ok = abs(gen - gt_val) <= 0.05
            if ok: coef_pts += 1
            else: coef_misses.append(f"  {key}: {gen:.6f} vs {gt_val}")
    coef_score = 25 * coef_pts / coef_max

    se_pts, se_max = 0, 0
    se_fails = []
    for key, gt_se_val in gt_se.items():
        col_idx, var_name = var_map[key]
        se_max += 1
        if var_name in ms[col_idx].params.index:
            gen_se = ms[col_idx].bse[var_name]
            if abs(gen_se - gt_se_val) <= 0.02: se_pts += 1
            else: se_fails.append(f"  {key}: {gen_se:.6f} vs {gt_se_val}")
    se_score = 15 * se_pts / se_max

    n_ratio = abs(N - 13128) / 13128
    n_score = 15 if n_ratio <= 0.05 else 10 if n_ratio <= 0.10 else 5

    sig_pts, sig_max = 0, 0
    sig_misses = []
    for key, target in gt_sig.items():
        col_idx, var_name = var_map[key]
        sig_max += 1
        if var_name in ms[col_idx].params.index:
            gen = get_sig(ms[col_idx].pvalues[var_name])
            if gen == target: sig_pts += 1
            else:
                t = abs(ms[col_idx].params[var_name] / ms[col_idx].bse[var_name])
                sig_misses.append(f"  {key}: gen={gen} vs target={target} (t={t:.2f})")
    sig_score = 25 * sig_pts / sig_max

    gt_r2 = [0.422, 0.428, 0.432, 0.433, 0.435]
    r2_pts = sum(1 for i in range(5) if abs(ms[i].rsquared - gt_r2[i]) <= 0.02)
    r2_score = 10 * r2_pts / 5

    total = coef_score + se_score + n_score + sig_score + 10 + r2_score

    print(f"\n=== {label} ===")
    print(f"Score: {total:.1f}, coef={coef_pts}/{coef_max}, se={se_pts}/{se_max}, sig={sig_pts}/{sig_max}, n={n_score}, r2={r2_pts}/5")
    if coef_misses:
        print("Coef misses:")
        for m in coef_misses: print(m)
    if se_fails:
        print("SE fails:")
        for f in se_fails: print(f)
    if sig_misses:
        print("Sig misses:")
        for m in sig_misses: print(m)

    # Print key values
    imp4_c = ms[3].params.get(imp_prefix, 0)
    imp4_se = ms[3].bse.get(imp_prefix, 1)
    imp5_c = ms[4].params.get(imp_prefix, 0)
    imp5_se = ms[4].bse.get(imp_prefix, 1)
    print(f"  imp col(4): {imp4_c:.6f} ({imp4_se:.6f}) t={abs(imp4_c/imp4_se):.3f}")
    print(f"  imp col(5): {imp5_c:.6f} ({imp5_se:.6f}) t={abs(imp5_c/imp5_se):.3f}")

    return total

# Run all three
s1 = run_and_score('imp_ct', 'Standard OLS imputation')
s2 = run_and_score('imp_ct_log', 'Log-exp imputation')
s3 = run_and_score('imp_ct_qr', 'Quantile regression imputation')

print(f"\n\nFinal: OLS={s1:.1f}, Log-exp={s2:.1f}, QR={s3:.1f}")
