#!/usr/bin/env python3
"""Side-by-side detailed scoring of inv1983 vs inv1982."""
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

def build_models(s, cy):
    s = s.copy()
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
    y = s['y']
    return [
        sm.OLS(y, sm.add_constant(s[base + control_vars])).fit(),
        sm.OLS(y, sm.add_constant(s[base + ['ct_obs', 'ct_x_censor'] + control_vars])).fit(),
        sm.OLS(y, sm.add_constant(s[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])).fit(),
        sm.OLS(y, sm.add_constant(s[base + ['imp_ct'] + control_vars])).fit(),
        sm.OLS(y, sm.add_constant(s[base + ['imp_ct', 'imp_ct_x_exp_sq', 'imp_ct_x_tenure'] + control_vars])).fit(),
    ]

all_vars_base = ['y', 'exp', 'exp_sq', 'tenure_var', 'ct_obs'] + control_vars
s0 = df.dropna(subset=all_vars_base).copy()
s0 = s0[s0['exp'] <= 36].copy()

m_83 = build_models(s0, 1983)
m_82 = build_models(s0, 1982)

print(f"{'key':>20s}  {'cfg83':>10s} {'c83_ok':>6s}  {'cfg82':>10s} {'c82_ok':>6s}  {'target':>10s}  {'s83':>4s} {'s83_ok':>6s}  {'s82':>4s} {'s82_ok':>6s}  {'tgt_s':>5s}")
c83_pts, c82_pts, s83_pts, s82_pts = 0, 0, 0, 0
c_tot, s_tot = 0, 0

for key in sorted(gt_coef.keys()):
    target = gt_coef[key]
    var, col = var_map[key]
    c_tot += 1

    g83 = m_83[col].params.get(var, None)
    g82 = m_82[col].params.get(var, None)

    def check_coef(gen, tgt):
        if gen is None: return False
        if abs(tgt) < 0.01:
            return abs(gen - tgt) / max(abs(tgt), 1e-8) <= 0.20
        else:
            return abs(gen - tgt) <= 0.05

    ok83 = check_coef(g83, target)
    ok82 = check_coef(g82, target)
    c83_pts += 1 if ok83 else 0
    c82_pts += 1 if ok82 else 0

    # Significance
    if key in gt_se:
        s_tot += 1
        ts = stars_from_t(target, gt_se[key])

        pv83 = m_83[col].pvalues.get(var, 1.0)
        gs83 = '***' if pv83 < 0.001 else '**' if pv83 < 0.01 else '*' if pv83 < 0.05 else ''
        sok83 = gs83 == ts
        s83_pts += 1 if sok83 else 0

        pv82 = m_82[col].pvalues.get(var, 1.0)
        gs82 = '***' if pv82 < 0.001 else '**' if pv82 < 0.01 else '*' if pv82 < 0.05 else ''
        sok82 = gs82 == ts
        s82_pts += 1 if sok82 else 0

        diff_c = 'DIFF' if ok83 != ok82 else ''
        diff_s = 'DIFF' if sok83 != sok82 else ''

        print(f"{str(key):>20s}  {g83:>10.6f} {'OK' if ok83 else 'X':>6s}  {g82:>10.6f} {'OK' if ok82 else 'X':>6s}  {target:>10.6f}  {gs83:>4s} {'OK' if sok83 else 'X':>6s}  {gs82:>4s} {'OK' if sok82 else 'X':>6s}  {ts:>5s} {diff_c:>5s} {diff_s:>5s}")
    else:
        diff_c = 'DIFF' if ok83 != ok82 else ''
        print(f"{str(key):>20s}  {g83:>10.6f} {'OK' if ok83 else 'X':>6s}  {g82:>10.6f} {'OK' if ok82 else 'X':>6s}  {target:>10.6f}  {'':>4s} {'':>6s}  {'':>4s} {'':>6s}  {'':>5s} {diff_c:>5s}")

print(f"\nCoef: cfg83={c83_pts}/{c_tot}, cfg82={c82_pts}/{c_tot}")
print(f"Sig:  cfg83={s83_pts}/{s_tot}, cfg82={s82_pts}/{s_tot}")
print(f"Score 83: {25*c83_pts/c_tot + 25*s83_pts/s_tot:.1f}")
print(f"Score 82: {25*c82_pts/c_tot + 25*s82_pts/s_tot:.1f}")
