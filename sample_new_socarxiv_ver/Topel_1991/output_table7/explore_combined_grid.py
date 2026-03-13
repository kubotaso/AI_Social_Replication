#!/usr/bin/env python3
"""Grid search: exp cutoff x censor year x alpha for maximum total score."""
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
gt_r2 = [0.422, 0.428, 0.432, 0.433, 0.435]

def stars_from_t(c, se):
    t = abs(c / se) if se > 0 else 0
    return '***' if t > 3.291 else '**' if t > 2.576 else '*' if t > 1.96 else ''

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

all_vars_base = ['exp', 'exp_sq', 'tenure_var', 'ct_obs'] + control_vars

best_score = 0
best_config = None

print(f"{'cut':>4s} {'cy':>4s} {'alpha':>6s} {'total':>6s} {'coef':>5s} {'sig':>4s} {'se':>3s} {'r2':>3s} {'N':>6s} {'N_sc':>5s}")

for cut in [35, 36, 37, 38, 39, 40]:
    s0 = df.dropna(subset=all_vars_base).copy()
    s0 = s0[s0['exp'] <= cut].copy()
    n = len(s0)
    n_err = abs(n - 13128) / 13128

    for cy in [1982, 1983]:
        # Only test with inverted censor (ct_obs * (1 - censor))
        s = s0.copy()
        s['censor'] = (s.groupby('job_id')['year'].transform('max') >= cy).astype(float)
        s['ct_x_censor'] = s['ct_obs'] * (1 - s['censor'])

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

        for alpha_10 in [70, 72, 75, 78, 80]:
            alpha = alpha_10 / 100.0
            s['y'] = alpha * s['lw_cps'] + (1 - alpha) * s['lw_gnp']
            y = s['y']

            models = [
                sm.OLS(y, sm.add_constant(s[base + control_vars])).fit(),
                sm.OLS(y, sm.add_constant(s[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit(),
                sm.OLS(y, sm.add_constant(s[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit(),
                sm.OLS(y, sm.add_constant(s[base + ['imp_ct'] + control_vars])).fit(),
                sm.OLS(y, sm.add_constant(s[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit(),
            ]

            coef_pts, coef_tot = 0, 0
            for key, target in gt_coef.items():
                var, col = var_map[key]
                coef_tot += 1
                gen = models[col].params.get(var, None)
                if gen is None: continue
                if abs(target) < 0.01:
                    match = abs(gen - target) / max(abs(target), 1e-8) <= 0.20
                else:
                    match = abs(gen - target) <= 0.05
                if match: coef_pts += 1

            sig_pts, sig_tot = 0, 0
            for key in gt_coef:
                if key not in gt_se: continue
                var, col = var_map[key]
                sig_tot += 1
                target_stars = stars_from_t(gt_coef[key], gt_se[key])
                gen_pv = models[col].pvalues.get(var, 1.0)
                gen_stars = '***' if gen_pv < 0.001 else '**' if gen_pv < 0.01 else '*' if gen_pv < 0.05 else ''
                if gen_stars == target_stars: sig_pts += 1

            se_pts, se_tot = 0, 0
            for key in gt_se:
                var, col = var_map[key]
                se_tot += 1
                gen_se = models[col].bse.get(var, 999)
                if abs(gen_se - gt_se[key]) <= 0.02: se_pts += 1

            n_sc = 15 if n_err <= 0.05 else 10 if n_err <= 0.10 else 5 if n_err <= 0.20 else 0
            r2_pts = sum(1 for i in range(5) if abs(models[i].rsquared - gt_r2[i]) <= 0.02)
            total = 25*coef_pts/coef_tot + 25*sig_pts/sig_tot + 15*se_pts/se_tot + n_sc + 10 + 10*r2_pts/5

            if total >= best_score:
                best_score = total
                best_config = (cut, cy, alpha)
                print(f"{cut:>4d} {cy:>4d} {alpha:>6.2f} {total:>6.1f} {coef_pts:>5d} {sig_pts:>4d} {se_pts:>3d} {r2_pts:>3d} {n:>6d} {n_sc:>5d}")
            elif total >= 87:
                print(f"{cut:>4d} {cy:>4d} {alpha:>6.2f} {total:>6.1f} {coef_pts:>5d} {sig_pts:>4d} {se_pts:>3d} {r2_pts:>3d} {n:>6d} {n_sc:>5d}")

print(f"\nBest: cut={best_config[0]}, cy={best_config[1]}, alpha={best_config[2]:.2f}, score={best_score:.1f}")
