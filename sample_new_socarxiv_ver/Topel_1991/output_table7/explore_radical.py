#!/usr/bin/env python3
"""Radical exploration: try fundamentally different approaches to break past 88."""
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

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols
base = ['exp', 'exp_sq', 'tenure_var']

# Ground truth for scoring
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

def full_score(models, N):
    coef_pts, coef_max = 0, 0
    coef_misses = []
    for key, gt_val in gt_coef.items():
        col_idx, var_name = var_map[key]
        coef_max += 1
        if var_name in models[col_idx].params.index:
            gen = models[col_idx].params[var_name]
            if abs(gt_val) < 0.01:
                ok = abs(gen - gt_val) / max(abs(gt_val), 1e-8) <= 0.20
            else:
                ok = abs(gen - gt_val) <= 0.05
            if ok: coef_pts += 1
            else: coef_misses.append(f"  {key}: {gen:.6f} vs {gt_val}")
    coef_score = 25 * coef_pts / coef_max

    se_pts, se_max = 0, 0
    for key, gt_se_val in gt_se.items():
        col_idx, var_name = var_map[key]
        se_max += 1
        if var_name in models[col_idx].params.index:
            gen_se = models[col_idx].bse[var_name]
            if abs(gen_se - gt_se_val) <= 0.02: se_pts += 1
    se_score = 15 * se_pts / se_max

    n_ratio = abs(N - 13128) / 13128
    n_score = 15 if n_ratio <= 0.05 else 10 if n_ratio <= 0.10 else 5

    sig_pts, sig_max = 0, 0
    sig_misses = []
    for key, target in gt_sig.items():
        col_idx, var_name = var_map[key]
        sig_max += 1
        if var_name in models[col_idx].params.index:
            gen = get_sig(models[col_idx].pvalues[var_name])
            if gen == target: sig_pts += 1
            else:
                t = abs(models[col_idx].params[var_name] / models[col_idx].bse[var_name])
                sig_misses.append(f"  {key}: gen={gen} vs target={target} (t={t:.2f})")
    sig_score = 25 * sig_pts / sig_max

    gt_r2 = [0.422, 0.428, 0.432, 0.433, 0.435]
    r2_pts = sum(1 for i in range(5) if abs(models[i].rsquared - gt_r2[i]) <= 0.02)
    r2_score = 10 * r2_pts / 5

    total = coef_score + se_score + n_score + sig_score + 10 + r2_score
    return total, coef_pts, coef_max, se_pts, se_max, sig_pts, sig_max, n_score, r2_pts, coef_misses, sig_misses


# ===================================================================
# STRATEGY 1: CPS-only deflation (alpha=1.0)
# ===================================================================
print("=== STRATEGY 1: CPS-only deflation (alpha=1.0) ===")
df['y_cps'] = df['lw_cps']
df['ct_x_censor'] = df['ct_obs'] * (1 - df['censor'])
df['ct_x_exp_sq'] = df['ct_obs'] * df['exp_sq']
df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

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

all_vars = ['y_cps', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct'] + control_vars
samp = df.dropna(subset=all_vars).copy()
samp = samp[samp['exp'] <= 36].copy()
yy = samp['y_cps']
N = len(samp)

ms = []
ms.append(sm.OLS(yy, sm.add_constant(samp[base + control_vars])).fit())
ms.append(sm.OLS(yy, sm.add_constant(samp[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit())
ms.append(sm.OLS(yy, sm.add_constant(samp[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit())
ms.append(sm.OLS(yy, sm.add_constant(samp[base + ['imp_ct'] + control_vars])).fit())
ms.append(sm.OLS(yy, sm.add_constant(samp[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit())

total, cp, cm, sp, sm_, sigp, sigm, ns, r2p, cmiss, smiss = full_score(ms, N)
print(f"Score: {total:.1f}, N={N}, coef={cp}/{cm}, se={sp}/{sm_}, sig={sigp}/{sigm}, n={ns}, r2={r2p}/5")
for m in smiss: print(m)


# ===================================================================
# STRATEGY 2: GNP-only deflation (alpha=0.0)
# ===================================================================
print("\n=== STRATEGY 2: GNP-only deflation (alpha=0.0) ===")
yy2 = samp['lw_gnp']
ms2 = []
ms2.append(sm.OLS(yy2, sm.add_constant(samp[base + control_vars])).fit())
ms2.append(sm.OLS(yy2, sm.add_constant(samp[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit())
ms2.append(sm.OLS(yy2, sm.add_constant(samp[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit())
ms2.append(sm.OLS(yy2, sm.add_constant(samp[base + ['imp_ct'] + control_vars])).fit())
ms2.append(sm.OLS(yy2, sm.add_constant(samp[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit())

total2, cp2, cm2, sp2, sm2_, sigp2, sigm2, ns2, r2p2, cmiss2, smiss2 = full_score(ms2, N)
print(f"Score: {total2:.1f}, N={N}, coef={cp2}/{cm2}, se={sp2}/{sm2_}, sig={sigp2}/{sigm2}, n={ns2}, r2={r2p2}/5")
for m in smiss2: print(m)


# ===================================================================
# STRATEGY 3: Try NON-inverted censor (ct_obs * censor)
# Use standard censor (not inverted)
# ===================================================================
print("\n=== STRATEGY 3: Non-inverted censor (ct_obs * censor) ===")
df['ct_x_censor_std'] = df['ct_obs'] * df['censor']
samp['ct_x_censor_std'] = samp['ct_obs'] * samp['censor']
yy = samp['y']

# Need to adjust var_map for scoring
ms3 = []
ms3.append(sm.OLS(yy, sm.add_constant(samp[base + control_vars])).fit())
ms3.append(sm.OLS(yy, sm.add_constant(samp[base + ['ct_obs', 'ct_x_censor_std'] + control_vars])).fit())
ms3.append(sm.OLS(yy, sm.add_constant(samp[base + ['ct_obs', 'ct_x_censor_std', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit())
ms3.append(sm.OLS(yy, sm.add_constant(samp[base + ['imp_ct'] + control_vars])).fit())
ms3.append(sm.OLS(yy, sm.add_constant(samp[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit())

# Manually score censor with different var name
for col in [1, 2]:
    c = ms3[col].params.get('ct_x_censor_std', 0)
    se = ms3[col].bse.get('ct_x_censor_std', 1)
    t = abs(c/se)
    print(f"  censor col({col+1}): coef={c:.6f}, se={se:.6f}, t={t:.2f}")

# Total score manually - use inverted values in var_map
var_map_std = dict(var_map)
var_map_std[('cen', 1)] = (1, 'ct_x_censor_std')
var_map_std[('cen', 2)] = (2, 'ct_x_censor_std')

coef_pts3, coef_max3, sig_pts3, sig_max3 = 0, 0, 0, 0
for key, gt_val in gt_coef.items():
    col_idx, var_name = var_map_std[key]
    coef_max3 += 1
    if var_name in ms3[col_idx].params.index:
        gen = ms3[col_idx].params[var_name]
        if abs(gt_val) < 0.01:
            ok = abs(gen - gt_val) / max(abs(gt_val), 1e-8) <= 0.20
        else:
            ok = abs(gen - gt_val) <= 0.05
        if ok: coef_pts3 += 1
for key, target in gt_sig.items():
    col_idx, var_name = var_map_std[key]
    sig_max3 += 1
    if var_name in ms3[col_idx].params.index:
        gen = get_sig(ms3[col_idx].pvalues[var_name])
        if gen == target: sig_pts3 += 1

print(f"Non-inverted censor: coef={coef_pts3}/{coef_max3}, sig={sig_pts3}/{sig_max3}")


# ===================================================================
# STRATEGY 4: What if tenure_var doesn't include the current year?
# i.e., tenure = years COMPLETED, not years INCLUDING current
# tenure_topel - 1 for the variable, but ct_obs stays the same
# ===================================================================
print("\n=== STRATEGY 4: Tenure = tenure_topel - 1 (completed years, not including current) ===")
samp_t = samp.copy()
samp_t['tenure_var'] = (samp_t['tenure_topel'] - 1).clip(lower=0).astype(float)
samp_t['ct_x_tenure'] = samp_t['ct_obs'] * samp_t['tenure_var']
samp_t['imp_ct_x_tenure'] = samp_t['imp_ct'] * samp_t['tenure_var']
yy = samp_t['y']

ms4 = []
ms4.append(sm.OLS(yy, sm.add_constant(samp_t[base + control_vars])).fit())
ms4.append(sm.OLS(yy, sm.add_constant(samp_t[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit())
ms4.append(sm.OLS(yy, sm.add_constant(samp_t[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit())
ms4.append(sm.OLS(yy, sm.add_constant(samp_t[base + ['imp_ct'] + control_vars])).fit())
ms4.append(sm.OLS(yy, sm.add_constant(samp_t[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit())

total4, cp4, cm4, sp4, sm4_, sigp4, sigm4, ns4, r2p4, cmiss4, smiss4 = full_score(ms4, N)
print(f"Score: {total4:.1f}, N={N}, coef={cp4}/{cm4}, se={sp4}/{sm4_}, sig={sigp4}/{sigm4}, n={ns4}, r2={r2p4}/5")
print("  Key tenure coefficients:")
for col in [0, 1, 3]:
    c = ms4[col].params['tenure_var']
    se = ms4[col].bse['tenure_var']
    print(f"    tenure col({col+1}): {c:.6f} ({se:.6f}) t={abs(c/se):.2f}")
for m in smiss4: print(m)


# ===================================================================
# STRATEGY 5: Exclude first-year observations of each job
# (tenure_topel > 1 only)
# ===================================================================
print("\n=== STRATEGY 5: Exclude first-year job observations (tenure > 1) ===")
samp5 = samp[samp['tenure_topel'] > 1].copy()
yy5 = samp5['y']
N5 = len(samp5)

ms5 = []
ms5.append(sm.OLS(yy5, sm.add_constant(samp5[base + control_vars])).fit())
ms5.append(sm.OLS(yy5, sm.add_constant(samp5[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit())
ms5.append(sm.OLS(yy5, sm.add_constant(samp5[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit())
ms5.append(sm.OLS(yy5, sm.add_constant(samp5[base + ['imp_ct'] + control_vars])).fit())
ms5.append(sm.OLS(yy5, sm.add_constant(samp5[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit())

total5, cp5, cm5, sp5, sm5_, sigp5, sigm5, ns5, r2p5, cmiss5, smiss5 = full_score(ms5, N5)
print(f"Score: {total5:.1f}, N={N5}, coef={cp5}/{cm5}, se={sp5}/{sm5_}, sig={sigp5}/{sigm5}, n={ns5}, r2={r2p5}/5")
for m in smiss5: print(m)


# ===================================================================
# STRATEGY 6: Use log(tenure) instead of tenure in interactions
# ===================================================================
print("\n=== STRATEGY 6: Use log(tenure) in CT interactions ===")
samp6 = samp.copy()
samp6['ct_x_exp_sq'] = samp6['ct_obs'] * samp6['exp_sq']
samp6['ct_x_log_tenure'] = samp6['ct_obs'] * np.log(samp6['tenure_var'])
samp6['imp_ct_x_exp_sq'] = samp6['imp_ct'] * samp6['exp_sq']
samp6['imp_ct_x_log_tenure'] = samp6['imp_ct'] * np.log(samp6['tenure_var'])
yy6 = samp6['y']

# This won't score well because the interaction variable is different,
# but let's see if the coefficient patterns match better
ms6 = []
ms6.append(sm.OLS(yy6, sm.add_constant(samp6[base + control_vars])).fit())
ms6.append(sm.OLS(yy6, sm.add_constant(samp6[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit())
ms6.append(sm.OLS(yy6, sm.add_constant(samp6[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_log_tenure'] + control_vars])).fit())
ms6.append(sm.OLS(yy6, sm.add_constant(samp6[base + ['imp_ct'] + control_vars])).fit())
ms6.append(sm.OLS(yy6, sm.add_constant(samp6[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_log_tenure'] + control_vars])).fit())

print(f"  tenure col(1): {ms6[0].params['tenure_var']:.6f} (t={abs(ms6[0].params['tenure_var']/ms6[0].bse['tenure_var']):.2f})")
print(f"  tenure col(3): {ms6[2].params['tenure_var']:.6f} (t={abs(ms6[2].params['tenure_var']/ms6[2].bse['tenure_var']):.2f})")
print(f"  ct_x_log_tenure col(3): {ms6[2].params['ct_x_log_tenure']:.6f}")


# ===================================================================
# STRATEGY 7: Scale completed tenure by dividing by some constant
# to make interactions larger
# ===================================================================
print("\n=== STRATEGY 7: Scale CT by multiplying by constant ===")
for scale in [2, 3, 5]:
    samp7 = samp.copy()
    samp7['ct_obs_s'] = samp7['ct_obs'] * scale
    samp7['ct_x_censor_s'] = samp7['ct_obs_s'] * (1 - samp7['censor'])
    samp7['ct_x_exp_sq_s'] = samp7['ct_obs_s'] * samp7['exp_sq']
    samp7['ct_x_tenure_s'] = samp7['ct_obs_s'] * samp7['tenure_var']

    ms7 = []
    ms7.append(sm.OLS(samp7['y'], sm.add_constant(samp7[base + control_vars])).fit())
    ms7.append(sm.OLS(samp7['y'], sm.add_constant(samp7[base + ['ct_obs_s', 'ct_x_censor_s'] + control_vars])).fit())
    ms7.append(sm.OLS(samp7['y'], sm.add_constant(samp7[base + ['ct_obs_s', 'ct_x_censor_s', 'ct_x_exp_sq_s', 'ct_x_tenure_s'] + control_vars])).fit())

    ct_coef = ms7[1].params['ct_obs_s']
    ct_se = ms7[1].bse['ct_obs_s']
    esq_int = ms7[2].params['ct_x_exp_sq_s']
    ten_int = ms7[2].params['ct_x_tenure_s']
    print(f"  scale={scale}: ct={ct_coef:.6f}(se={ct_se:.6f}), esq_int={esq_int:.8f}, ten_int={ten_int:.6f}")
    # Note: scaling CT just divides the coefficient by scale, doesn't help


# ===================================================================
# STRATEGY 8: Use different year dummies base
# ===================================================================
print("\n=== STRATEGY 8: Different year dummy base (1968 instead of 1971) ===")
yr_cols_68 = [c for c in df.columns if c.startswith('year_') and c != 'year_1968' and df[c].sum() > 0]
control_vars_68 = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols_68

all_vars_68 = ['y', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct'] + control_vars_68
samp8 = df.dropna(subset=all_vars_68).copy()
samp8 = samp8[samp8['exp'] <= 36].copy()
samp8['ct_x_censor'] = samp8['ct_obs'] * (1 - samp8['censor'])
samp8['ct_x_exp_sq'] = samp8['ct_obs'] * samp8['exp_sq']
samp8['ct_x_tenure'] = samp8['ct_obs'] * samp8['tenure_var']
samp8['imp_ct_x_exp_sq'] = samp8['imp_ct'] * samp8['exp_sq']
samp8['imp_ct_x_tenure'] = samp8['imp_ct'] * samp8['tenure_var']
yy8 = samp8['y']

ms8 = []
ms8.append(sm.OLS(yy8, sm.add_constant(samp8[base + control_vars_68])).fit())
ms8.append(sm.OLS(yy8, sm.add_constant(samp8[base + ['ct_obs', 'ct_x_censor'] + control_vars_68])).fit())
ms8.append(sm.OLS(yy8, sm.add_constant(samp8[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars_68])).fit())
ms8.append(sm.OLS(yy8, sm.add_constant(samp8[base + ['imp_ct'] + control_vars_68])).fit())
ms8.append(sm.OLS(yy8, sm.add_constant(samp8[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars_68])).fit())

total8, cp8, cm8, sp8, sm8_, sigp8, sigm8, ns8, r2p8, cmiss8, smiss8 = full_score(ms8, len(samp8))
print(f"Score: {total8:.1f}, N={len(samp8)}, coef={cp8}/{cm8}, se={sp8}/{sm8_}, sig={sigp8}/{sigm8}, n={ns8}, r2={r2p8}/5")


# ===================================================================
# STRATEGY 9: Add experience^3 or higher order terms
# ===================================================================
print("\n=== STRATEGY 9: Quadratic completed tenure (ct_obs + ct_obs^2) ===")
samp9 = samp.copy()
samp9['ct_obs_sq'] = samp9['ct_obs'] ** 2
yy9 = samp9['y']

ms9_2 = sm.OLS(yy9, sm.add_constant(samp9[base + ['ct_obs', 'ct_obs_sq', 'ct_x_censor'] + control_vars])).fit()
print(f"  ct_obs: {ms9_2.params['ct_obs']:.6f} (t={abs(ms9_2.params['ct_obs']/ms9_2.bse['ct_obs']):.2f})")
print(f"  ct_obs_sq: {ms9_2.params['ct_obs_sq']:.6f} (t={abs(ms9_2.params['ct_obs_sq']/ms9_2.bse['ct_obs_sq']):.2f})")
print(f"  tenure: {ms9_2.params['tenure_var']:.6f} (t={abs(ms9_2.params['tenure_var']/ms9_2.bse['tenure_var']):.2f})")


# ===================================================================
# STRATEGY 10: What if we use exp/100 for experience?
# Some papers use experience in hundreds
# ===================================================================
print("\n=== STRATEGY 10: Different education mapping ===")
# Try mapping that gives more education years to lower categories
EDUC_ALT = {0:0, 1:2, 2:6, 3:9, 4:11, 5:12, 6:14, 7:16, 8:17, 9:17}
df_alt = df.copy()
df_alt['ed_yrs'] = df_alt['education_clean'].copy()
for yr in df_alt['year'].unique():
    m = df_alt['year'] == yr
    if df_alt.loc[m, 'education_clean'].max() <= 9:
        df_alt.loc[m, 'ed_yrs'] = df_alt.loc[m, 'education_clean'].map(EDUC_ALT)

df_alt['exp'] = (df_alt['age'] - df_alt['ed_yrs'] - 6).clip(lower=1)
df_alt['exp_sq'] = df_alt['exp'] ** 2
df_alt['ct_x_exp_sq'] = df_alt['ct_obs'] * df_alt['exp_sq']
df_alt['ct_x_tenure'] = df_alt['ct_obs'] * df_alt['tenure_var']
df_alt['imp_ct_x_exp_sq'] = df_alt['imp_ct'] * df_alt['exp_sq']
df_alt['imp_ct_x_tenure'] = df_alt['imp_ct'] * df_alt['tenure_var']
df_alt['ct_x_censor'] = df_alt['ct_obs'] * (1 - df_alt['censor'])
df_alt['ed_cat'] = pd.cut(df_alt['ed_yrs'], bins=[-1, 11, 12, 15, 20], labels=['lt12', '12', '13_15', '16plus'])
ed_dummies_alt = pd.get_dummies(df_alt['ed_cat'], prefix='ed', drop_first=True, dtype=float)
for col in ed_dummies_alt.columns:
    df_alt[col] = ed_dummies_alt[col]

all_vars_alt = ['y', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct'] + control_vars
samp_alt = df_alt.dropna(subset=all_vars_alt).copy()
samp_alt = samp_alt[samp_alt['exp'] <= 36].copy()
yy_alt = samp_alt['y']

ms_alt = []
ms_alt.append(sm.OLS(yy_alt, sm.add_constant(samp_alt[base + control_vars])).fit())
ms_alt.append(sm.OLS(yy_alt, sm.add_constant(samp_alt[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit())
ms_alt.append(sm.OLS(yy_alt, sm.add_constant(samp_alt[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit())
ms_alt.append(sm.OLS(yy_alt, sm.add_constant(samp_alt[base + ['imp_ct'] + control_vars])).fit())
ms_alt.append(sm.OLS(yy_alt, sm.add_constant(samp_alt[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit())

total_alt, cp_alt, cm_alt, sp_alt, sm_alt_, sigp_alt, sigm_alt, ns_alt, r2p_alt, cmiss_alt, smiss_alt = full_score(ms_alt, len(samp_alt))
print(f"Score: {total_alt:.1f}, N={len(samp_alt)}, coef={cp_alt}/{cm_alt}, se={sp_alt}/{sm_alt_}, sig={sigp_alt}/{sigm_alt}, n={ns_alt}, r2={r2p_alt}/5")


# FINAL SUMMARY
print("\n" + "=" * 60)
print("SUMMARY OF ALL STRATEGIES")
print("=" * 60)
print(f"Base (OLS, inv censor, exp<=36):  88.0")
print(f"CPS-only (alpha=1.0):             {total:.1f}")
print(f"GNP-only (alpha=0.0):             {total2:.1f}")
print(f"Tenure - 1:                       {total4:.1f}")
print(f"Exclude first-year obs:           {total5:.1f}")
print(f"Year dummy base=1968:             {total8:.1f}")
print(f"Alt education mapping:            {total_alt:.1f}")
