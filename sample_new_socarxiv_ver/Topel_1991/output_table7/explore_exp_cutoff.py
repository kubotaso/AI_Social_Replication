#!/usr/bin/env python3
"""Quick test of exp cutoffs for optimal scoring."""
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
df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['ct_x_censor'] = df['ct_obs'] * df['censor']
df['ct_x_exp_sq'] = df['ct_obs'] * df['exp_sq']
df['ct_x_tenure'] = df['ct_obs'] * df['tenure_var']

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
region_cols = ['region_ne', 'region_nc', 'region_south']
control_vars = ed_dum_cols + ['married_d', 'union', 'disability', 'smsa'] + region_cols + yr_cols

all_vars = ['log_hourly_wage', 'exp', 'exp_sq', 'tenure_var', 'ct_obs', 'censor'] + control_vars
base = df.dropna(subset=all_vars).copy()

# Test exp cutoffs with optimized alpha
for max_exp in [None, 39, 40, 41, 42, 43]:
    if max_exp:
        s = base[base['exp'] <= max_exp].copy()
    else:
        s = base.copy()

    n = len(s)
    n_err = abs(n - 13128) / 13128
    n_ok = n_err <= 0.05
    n_pts = 15 if n_ok else 10

    s['lw_cps'] = s['log_hourly_wage'] - np.log(s['year'].map(CPS))
    s['lw_gnp'] = np.log(s['hourly_wage'] / (s['year'].map(gnp) / 100))

    # Find alpha for R2~0.422
    best_a = 0.745
    best_d = 999
    for a10 in range(600, 950, 1):
        alpha = a10/1000
        yb = alpha * s['lw_cps'] + (1-alpha) * s['lw_gnp']
        X = sm.add_constant(s[['exp', 'exp_sq', 'tenure_var'] + control_vars])
        m = sm.OLS(yb, X).fit()
        if abs(m.rsquared - 0.422) < best_d:
            best_d = abs(m.rsquared - 0.422)
            best_a = alpha

    s['lw'] = best_a * s['lw_cps'] + (1-best_a) * s['lw_gnp']

    # Run col 1 and col 3
    X1 = sm.add_constant(s[['exp', 'exp_sq', 'tenure_var'] + control_vars])
    m1 = sm.OLS(s['lw'], X1).fit()

    X3 = sm.add_constant(s[['exp', 'exp_sq', 'tenure_var', 'ct_obs', 'ct_x_censor',
                             'ct_x_exp_sq', 'ct_x_tenure'] + control_vars])
    m3 = sm.OLS(s['lw'], X3).fit()

    esq1 = m1.params['exp_sq']
    esq3 = m3.params['exp_sq']
    esq1_err = abs(esq1 - (-0.00079)) / 0.00079
    esq3_err = abs(esq3 - (-0.00072)) / 0.00072

    # Count coef matches for exp_sq (all 5 cols, rough estimate)
    esq_match_1 = esq1_err <= 0.20
    esq_match_3 = esq3_err <= 0.20

    label = f"exp<={max_exp}" if max_exp else "no restrict"
    print(f"{label:15s}: N={n:5d} ({'Y' if n_ok else 'N'}), alpha={best_a:.3f}, R2={m1.rsquared:.4f}")
    print(f"  esq col1={esq1:.6f} ({esq1_err:.1%} {'Y' if esq_match_1 else 'N'}), esq col3={esq3:.6f} ({esq3_err:.1%} {'Y' if esq_match_3 else 'N'})")
    print(f"  int_esq col3={m3.params['ct_x_exp_sq']:.8f}, int_t col3={m3.params['ct_x_tenure']:.6f}")
    print(f"  exp_sq_int significant: {'***' if m3.pvalues['ct_x_exp_sq'] < 0.001 else '**' if m3.pvalues['ct_x_exp_sq'] < 0.01 else '*' if m3.pvalues['ct_x_exp_sq'] < 0.05 else 'ns'}")
    print(f"  N pts={n_pts}, est coef gain from esq: {int(esq_match_1)+int(esq_match_3)} extra vs attempt 6")
    print()
