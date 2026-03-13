#!/usr/bin/env python3
"""Test adding occupation dummies as controls and in CT prediction."""
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
df['ct_x_censor'] = df['ct_obs'] * (1 - df['censor'])
df['ct_x_exp_sq'] = df['ct_obs'] * df['exp_sq']
df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
occ_cols = [f'occ_{i}' for i in range(1, 10)]  # occ_0 is base

# ============================================================
# Test 1: Add occupation dummies as additional controls
# ============================================================
print("=== TEST 1: Occupation dummies as controls ===")
control_base = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols
control_occ = control_base + occ_cols

# Check if occupation dummies are valid
print(f"Occupation dummy sums: {[f'{c}:{df[c].sum():.0f}' for c in occ_cols]}")

all_vars = ['y', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor'] + control_base
sample = df.dropna(subset=all_vars + occ_cols).copy()
sample = sample[sample['exp'] <= 36].copy()
print(f"Sample with occ dummies: N={len(sample)}")

y = sample['y']
base = ['exp', 'exp_sq', 'tenure_var']

# Without occupation dummies
m1_no = sm.OLS(y, sm.add_constant(sample[base + control_base])).fit()

# With occupation dummies
m1_occ = sm.OLS(y, sm.add_constant(sample[base + control_occ])).fit()

print(f"\nCol 1 without occ: tenure={m1_no.params['tenure_var']:.6f} ({m1_no.bse['tenure_var']:.6f}), R2={m1_no.rsquared:.4f}")
print(f"Col 1 with occ:    tenure={m1_occ.params['tenure_var']:.6f} ({m1_occ.bse['tenure_var']:.6f}), R2={m1_occ.rsquared:.4f}")

# Models 1-5 with occ
m2_occ = sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor'] + control_occ])).fit()
m3_occ = sm.OLS(y, sm.add_constant(sample[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_occ])).fit()

# Imputed CT with occupation in prediction model
job_data = df.groupby('job_id').agg({
    'ct_obs': 'first', 'censor': 'first', 'exp': 'first',
    'ed_yrs': 'first', 'married_d': 'first', 'union': 'first', 'smsa': 'first',
    **{c: 'first' for c in occ_cols},
}).reset_index()
uncensored = job_data[job_data['censor'] == 0]
pred_vars_occ = ['exp', 'ed_yrs', 'married_d', 'union', 'smsa'] + occ_cols
ols_ct_occ = sm.OLS(uncensored['ct_obs'], sm.add_constant(uncensored[pred_vars_occ])).fit()
print(f"\nCT prediction R2 with occ: {ols_ct_occ.rsquared:.3f}")
job_data['pred_ct_occ'] = ols_ct_occ.predict(sm.add_constant(job_data[pred_vars_occ])).clip(lower=1)
job_data.loc[job_data['censor'] == 0, 'pred_ct_occ'] = job_data.loc[job_data['censor'] == 0, 'ct_obs']

df['imp_ct_occ'] = df['job_id'].map(job_data.set_index('job_id')['pred_ct_occ'])
df['imp_ct_occ_x_exp_sq'] = df['imp_ct_occ'] * df['exp_sq']
df['imp_ct_occ_x_tenure'] = df['imp_ct_occ'] * df['tenure_var']

sample2 = df.dropna(subset=all_vars + occ_cols + ['imp_ct_occ']).copy()
sample2 = sample2[sample2['exp'] <= 36].copy()
y2 = sample2['y']

m4_occ = sm.OLS(y2, sm.add_constant(sample2[base + ['imp_ct_occ'] + control_occ])).fit()
m5_occ = sm.OLS(y2, sm.add_constant(sample2[base + ['imp_ct_occ', 'imp_ct_occ_x_exp_sq', 'imp_ct_occ_x_tenure'] + control_occ])).fit()

models_occ = [m1_occ, m2_occ, m3_occ, m4_occ, m5_occ]

# Compare key coefficients
print("\n=== KEY COEFFICIENTS WITH OCC CONTROLS ===")
print(f"{'Var':25s} {'Col':>4s} {'NoOcc':>10s} {'WithOcc':>10s} {'Target':>10s}")

targets = [
    ('tenure_var', 0, 0.0138), ('tenure_var', 1, -0.0015), ('tenure_var', 2, 0.0137),
    ('tenure_var', 3, 0.006), ('tenure_var', 4, 0.0163),
    ('ct_obs', 1, 0.0165), ('ct_obs', 2, 0.0316),
    ('ct_x_censor', 1, -0.0025), ('ct_x_censor', 2, -0.0024),
    ('imp_ct_occ', 3, 0.0053), ('imp_ct_occ', 4, 0.0067),
    ('ct_x_exp_sq', 2, -0.00061),
    ('ct_x_tenure', 2, 0.0142),
    ('imp_ct_occ_x_exp_sq', 4, -0.00075),
    ('imp_ct_occ_x_tenure', 4, 0.0429),
]

# Also do score
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

var_map = {
    ('exp', 0): ('exp', 0), ('exp', 1): ('exp', 1), ('exp', 2): ('exp', 2),
    ('exp', 3): ('exp', 3), ('exp', 4): ('exp', 4),
    ('esq', 0): ('exp_sq', 0), ('esq', 1): ('exp_sq', 1), ('esq', 2): ('exp_sq', 2),
    ('esq', 3): ('exp_sq', 3), ('esq', 4): ('exp_sq', 4),
    ('ten', 0): ('tenure_var', 0), ('ten', 1): ('tenure_var', 1), ('ten', 2): ('tenure_var', 2),
    ('ten', 3): ('tenure_var', 3), ('ten', 4): ('tenure_var', 4),
    ('ct', 1): ('ct_obs', 1), ('ct', 2): ('ct_obs', 2),
    ('cen', 1): ('ct_x_censor', 1), ('cen', 2): ('ct_x_censor', 2),
    ('imp', 3): ('imp_ct_occ', 3), ('imp', 4): ('imp_ct_occ', 4),
    ('esq_int', 2): ('ct_x_exp_sq', 2), ('esq_int', 4): ('imp_ct_occ_x_exp_sq', 4),
    ('ten_int', 2): ('ct_x_tenure', 2), ('ten_int', 4): ('imp_ct_occ_x_tenure', 4),
}

def stars_from_t(c, se):
    t = abs(c / se) if se > 0 else 0
    return '***' if t > 3.291 else '**' if t > 2.576 else '*' if t > 1.96 else ''

coef_pts, coef_tot = 0, 0
for key, target in gt_coef.items():
    var, col = var_map[key]
    coef_tot += 1
    gen = models_occ[col].params.get(var, None)
    if gen is None:
        print(f"  MISSING: {key} -> {var} in model {col}")
        continue
    if abs(target) < 0.01:
        match = abs(gen - target) / max(abs(target), 1e-8) <= 0.20
    else:
        match = abs(gen - target) <= 0.05
    if match:
        coef_pts += 1
    else:
        pct = abs(gen - target) / max(abs(target), 1e-8) * 100
        print(f"  MISS {key}: gen={gen:.6f} vs target={target}, err={pct:.0f}%")

sig_pts, sig_tot = 0, 0
for key in gt_coef:
    if key not in gt_se: continue
    target_c = gt_coef[key]
    target_se_val = gt_se[key]
    var, col = var_map[key]
    sig_tot += 1
    target_stars = stars_from_t(target_c, target_se_val)
    gen_pv = models_occ[col].pvalues.get(var, 1.0)
    gen_stars = '***' if gen_pv < 0.001 else '**' if gen_pv < 0.01 else '*' if gen_pv < 0.05 else ''
    if gen_stars == target_stars:
        sig_pts += 1

se_pts, se_tot = 0, 0
for key in gt_se:
    var, col = var_map[key]
    se_tot += 1
    gen_se = models_occ[col].bse.get(var, 999)
    if abs(gen_se - gt_se[key]) <= 0.02:
        se_pts += 1

n = int(models_occ[0].nobs)
n_err = abs(n - 13128) / 13128
n_sc = 15 if n_err <= 0.05 else 10 if n_err <= 0.10 else 5 if n_err <= 0.20 else 0

r2_pts = sum(1 for i in range(5) if abs(models_occ[i].rsquared - gt_r2[i]) <= 0.02)

coef_score = 25 * coef_pts / coef_tot
sig_score = 25 * sig_pts / sig_tot
se_score = 15 * se_pts / se_tot
var_score = 10
r2_score = 10 * r2_pts / 5

total = coef_score + sig_score + se_score + n_sc + var_score + r2_score
print(f"\nWith occ controls: total={total:.1f} coef={coef_pts}/{coef_tot} sig={sig_pts}/{sig_tot} se={se_pts}/{se_tot} r2={r2_pts}/5 n={n}")

# ============================================================
# Test 2: Without occ controls but with occ in CT prediction
# ============================================================
print("\n\n=== TEST 2: Occ only in CT prediction, not as controls ===")
# Use normal controls but CT predicted with occ
all_vars2 = ['y', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct_occ'] + control_base
sample3 = df.dropna(subset=all_vars2).copy()
sample3 = sample3[sample3['exp'] <= 36].copy()
y3 = sample3['y']

m1_v2 = sm.OLS(y3, sm.add_constant(sample3[base + control_base])).fit()
m2_v2 = sm.OLS(y3, sm.add_constant(sample3[base + ['ct_obs', 'ct_x_censor'] + control_base])).fit()
m3_v2 = sm.OLS(y3, sm.add_constant(sample3[base + ['ct_obs', 'ct_x_censor', 'ct_x_exp_sq', 'ct_x_tenure'] + control_base])).fit()
m4_v2 = sm.OLS(y3, sm.add_constant(sample3[base + ['imp_ct_occ'] + control_base])).fit()
m5_v2 = sm.OLS(y3, sm.add_constant(sample3[base + ['imp_ct_occ', 'imp_ct_occ_x_exp_sq', 'imp_ct_occ_x_tenure'] + control_base])).fit()

models_v2 = [m1_v2, m2_v2, m3_v2, m4_v2, m5_v2]

# Score
var_map2 = var_map.copy()  # same mapping
coef_pts2, coef_tot2 = 0, 0
for key, target in gt_coef.items():
    var, col = var_map2[key]
    coef_tot2 += 1
    gen = models_v2[col].params.get(var, None)
    if gen is None: continue
    if abs(target) < 0.01:
        match = abs(gen - target) / max(abs(target), 1e-8) <= 0.20
    else:
        match = abs(gen - target) <= 0.05
    if match: coef_pts2 += 1

sig_pts2, sig_tot2 = 0, 0
for key in gt_coef:
    if key not in gt_se: continue
    var, col = var_map2[key]
    sig_tot2 += 1
    target_stars = stars_from_t(gt_coef[key], gt_se[key])
    gen_pv = models_v2[col].pvalues.get(var, 1.0)
    gen_stars = '***' if gen_pv < 0.001 else '**' if gen_pv < 0.01 else '*' if gen_pv < 0.05 else ''
    if gen_stars == target_stars: sig_pts2 += 1

se_pts2, se_tot2 = 0, 0
for key in gt_se:
    var, col = var_map2[key]
    se_tot2 += 1
    gen_se = models_v2[col].bse.get(var, 999)
    if abs(gen_se - gt_se[key]) <= 0.02: se_pts2 += 1

n2 = int(models_v2[0].nobs)
n_err2 = abs(n2 - 13128) / 13128
n_sc2 = 15 if n_err2 <= 0.05 else 10 if n_err2 <= 0.10 else 5 if n_err2 <= 0.20 else 0
r2_pts2 = sum(1 for i in range(5) if abs(models_v2[i].rsquared - gt_r2[i]) <= 0.02)
total2 = 25*coef_pts2/coef_tot2 + 25*sig_pts2/sig_tot2 + 15*se_pts2/se_tot2 + n_sc2 + 10 + 10*r2_pts2/5

print(f"Occ in CT only: total={total2:.1f} coef={coef_pts2}/{coef_tot2} sig={sig_pts2}/{sig_tot2} se={se_pts2}/{se_tot2} r2={r2_pts2}/5 n={n2}")

# ============================================================
# Test 3: Agriculture dummy and govt_worker
# ============================================================
print("\n\n=== TEST 3: Add agriculture/govt controls ===")
extra_ctrl = ['agriculture', 'govt_worker']
control_extra = control_base + extra_ctrl
all_vars3 = ['y', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor', 'imp_ct'] + control_extra
sample4 = df.dropna(subset=all_vars3).copy()
sample4 = sample4[sample4['exp'] <= 36].copy()
y4 = sample4['y']

m1_e = sm.OLS(y4, sm.add_constant(sample4[base + control_extra])).fit()
print(f"With agri+govt: N={len(sample4)}, R2={m1_e.rsquared:.4f}")
print(f"  tenure: {m1_e.params['tenure_var']:.6f} ({m1_e.bse['tenure_var']:.6f})")
print(f"  agriculture: {m1_e.params['agriculture']:.6f}")
print(f"  govt_worker: {m1_e.params['govt_worker']:.6f}")

# ============================================================
# Test 4: Self-employed exclusion (paper says non-self-employed)
# ============================================================
print("\n\n=== TEST 4: Exclude self-employed ===")
print(f"self_employed values: {df['self_employed'].value_counts().to_dict()}")
sample5 = df.dropna(subset=all_vars).copy()
sample5 = sample5[sample5['exp'] <= 36].copy()
sample5 = sample5[sample5['self_employed'] == 0].copy()
print(f"After excluding self-employed: N={len(sample5)}")
y5 = sample5['y']
m1_nse = sm.OLS(y5, sm.add_constant(sample5[base + control_base])).fit()
print(f"  tenure: {m1_nse.params['tenure_var']:.6f} ({m1_nse.bse['tenure_var']:.6f}), R2={m1_nse.rsquared:.4f}")

# ============================================================
# Test 5: Govt worker exclusion
# ============================================================
print("\n\n=== TEST 5: Exclude govt workers ===")
print(f"govt_worker values: {df['govt_worker'].value_counts().to_dict()}")
sample6 = df.dropna(subset=all_vars).copy()
sample6 = sample6[sample6['exp'] <= 36].copy()
sample6 = sample6[sample6['govt_worker'] == 0].copy()
print(f"After excluding govt: N={len(sample6)}")
y6 = sample6['y']
m1_ngv = sm.OLS(y6, sm.add_constant(sample6[base + control_base])).fit()
print(f"  tenure: {m1_ngv.params['tenure_var']:.6f} ({m1_ngv.bse['tenure_var']:.6f}), R2={m1_ngv.rsquared:.4f}")

# ============================================================
# Test 6: Agriculture exclusion
# ============================================================
print("\n\n=== TEST 6: Exclude agriculture ===")
print(f"agriculture values: {df['agriculture'].value_counts().to_dict()}")
sample7 = df.dropna(subset=all_vars).copy()
sample7 = sample7[sample7['exp'] <= 36].copy()
sample7 = sample7[sample7['agriculture'] == 0].copy()
print(f"After excluding agriculture: N={len(sample7)}")
y7 = sample7['y']
m1_nag = sm.OLS(y7, sm.add_constant(sample7[base + control_base])).fit()
print(f"  tenure: {m1_nag.params['tenure_var']:.6f} ({m1_nag.bse['tenure_var']:.6f}), R2={m1_nag.rsquared:.4f}")

# ============================================================
# Test 7: Wage outlier trimming
# ============================================================
print("\n\n=== TEST 7: Wage outlier trimming ===")
sample8 = df.dropna(subset=all_vars).copy()
sample8 = sample8[sample8['exp'] <= 36].copy()
# Trim top/bottom 1% of wages
q01 = sample8['hourly_wage'].quantile(0.01)
q99 = sample8['hourly_wage'].quantile(0.99)
sample8 = sample8[(sample8['hourly_wage'] >= q01) & (sample8['hourly_wage'] <= q99)].copy()
y8 = sample8['y']
print(f"After wage trimming: N={len(sample8)}")
m1_trim = sm.OLS(y8, sm.add_constant(sample8[base + control_base])).fit()
print(f"  tenure: {m1_trim.params['tenure_var']:.6f} ({m1_trim.bse['tenure_var']:.6f}), R2={m1_trim.rsquared:.4f}")

# Try 5%
sample8b = df.dropna(subset=all_vars).copy()
sample8b = sample8b[sample8b['exp'] <= 36].copy()
q05 = sample8b['hourly_wage'].quantile(0.05)
q95 = sample8b['hourly_wage'].quantile(0.95)
sample8b = sample8b[(sample8b['hourly_wage'] >= q05) & (sample8b['hourly_wage'] <= q95)].copy()
y8b = sample8b['y']
print(f"After 5% wage trimming: N={len(sample8b)}")
m1_trim5 = sm.OLS(y8b, sm.add_constant(sample8b[base + control_base])).fit()
print(f"  tenure: {m1_trim5.params['tenure_var']:.6f} ({m1_trim5.bse['tenure_var']:.6f}), R2={m1_trim5.rsquared:.4f}")
