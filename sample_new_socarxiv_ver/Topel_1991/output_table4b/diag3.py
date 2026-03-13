"""
Diagnostic: Try different wage deflation for step 2.
Also check whether scaling issue in step 1 matters.
"""
import pandas as pd, numpy as np, statsmodels.api as sm
from linearmodels.iv import IV2SLS

df = pd.read_csv('data/psid_panel.csv')
df = df[~df['region'].isin([5, 6])]

EDUC = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ey'] = df['education_clean'].copy()
m = ~df['year'].isin([1975, 1976])
df.loc[m, 'ey'] = df.loc[m, 'education_clean'].map(EDUC)
df = df.dropna(subset=['ey'])

df['exp'] = (df['age'] - df['ey'] - 6).clip(lower=0)
df['ie'] = (df['exp'] - df['tenure_topel']).clip(lower=0)

CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
       1982:1.103,1983:1.089}
GNP = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,1973:44.4,1974:48.9,
       1975:53.6,1976:56.9,1977:60.6,1978:65.2,1979:72.6,1980:82.4,1981:90.9,1982:100.0}

df['cps'] = df['year'].map(CPS)
df['gnp'] = (df['year']-1).map(GNP)

# Three wage versions for step 2
df['lrw_nominal'] = df['log_hourly_wage']
df['lrw_cps'] = df['log_hourly_wage'] - np.log(df['cps'])
df['lrw_gnp_cps'] = df['log_hourly_wage'] - np.log(df['gnp']/100.0) - np.log(df['cps'])

for c in ['married','union_member','disabled','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

# Paper step 1 coefficients
beta_hat = 0.1258
gamma2 = -0.004592
gamma3 = 0.0001846
gamma4 = -0.00000245
delta2 = -0.004067
delta3 = 0.0000989
delta4 = 0.00000089

T = df['tenure_topel'].values.astype(float)
X = df['exp'].values.astype(float)

tenure_poly = gamma2*T**2 + gamma3*T**3 + gamma4*T**4
exp_poly = delta2*X**2 + delta3*X**3 + delta4*X**4

controls = ['ey', 'married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south']
yr = pd.get_dummies(df['year'], prefix='yr', dtype=float)
yr_cols = sorted(yr.columns.tolist())[1:]

df2 = df.reset_index(drop=True)

exog = df2[controls].copy()
for c in yr_cols:
    exog[c] = yr[c].values
exog = sm.add_constant(exog)

valid = exog.notna().all(axis=1)

for wage_name, wage_col in [('Nominal', 'lrw_nominal'), ('CPS-only', 'lrw_cps'), ('GNP+CPS', 'lrw_gnp_cps')]:
    w_star = df2[wage_col].values - beta_hat*T - tenure_poly - exp_poly
    df2['ws'] = w_star

    try:
        iv = IV2SLS(
            dependent=df2.loc[valid, 'ws'],
            exog=exog[valid],
            endog=df2.loc[valid, ['exp']],
            instruments=df2.loc[valid, ['ie']]
        ).fit()
        b1 = iv.params['exp']
        b1_se = iv.std_errors['exp']
        b2 = beta_hat - b1
        print(f"{wage_name}: beta_1={b1:.4f} ({b1_se:.4f}), beta_2={b2:.4f}")
    except Exception as e:
        print(f"{wage_name}: ERROR - {e}")

# Also try: what if year dummies are NOT included in step 2?
print("\n--- Without year dummies ---")
exog_noyr = df2[controls].copy()
exog_noyr = sm.add_constant(exog_noyr)
valid2 = exog_noyr.notna().all(axis=1)

for wage_name, wage_col in [('Nominal', 'lrw_nominal'), ('CPS-only', 'lrw_cps'), ('GNP+CPS', 'lrw_gnp_cps')]:
    w_star = df2[wage_col].values - beta_hat*T - tenure_poly - exp_poly
    df2['ws'] = w_star

    try:
        iv = IV2SLS(
            dependent=df2.loc[valid2, 'ws'],
            exog=exog_noyr[valid2],
            endog=df2.loc[valid2, ['exp']],
            instruments=df2.loc[valid2, ['ie']]
        ).fit()
        b1 = iv.params['exp']
        b1_se = iv.std_errors['exp']
        b2 = beta_hat - b1
        print(f"{wage_name}: beta_1={b1:.4f} ({b1_se:.4f}), beta_2={b2:.4f}")
    except Exception as e:
        print(f"{wage_name}: ERROR - {e}")

# Also try: what if we DON'T subtract exp polynomial from w*?
print("\n--- Without subtracting exp polynomial ---")
for wage_name, wage_col in [('GNP+CPS', 'lrw_gnp_cps')]:
    w_star = df2[wage_col].values - beta_hat*T - tenure_poly  # NO exp_poly subtracted
    df2['ws'] = w_star

    iv = IV2SLS(
        dependent=df2.loc[valid, 'ws'],
        exog=exog[valid],
        endog=df2.loc[valid, ['exp']],
        instruments=df2.loc[valid, ['ie']]
    ).fit()
    b1 = iv.params['exp']
    b1_se = iv.std_errors['exp']
    b2 = beta_hat - b1
    print(f"{wage_name} (no exp poly): beta_1={b1:.4f} ({b1_se:.4f}), beta_2={b2:.4f}")

# What if we subtract only tenure from exp to get initial exp?
# But allow negative initial experience?
print("\n--- Allow negative initial experience ---")
df2['ie_raw'] = df2['exp'] - df2['tenure_topel']  # No clipping
w_star = df2['lrw_gnp_cps'].values - beta_hat*T - tenure_poly - exp_poly
df2['ws'] = w_star

iv = IV2SLS(
    dependent=df2.loc[valid, 'ws'],
    exog=exog[valid],
    endog=df2.loc[valid, ['exp']],
    instruments=df2.loc[valid, ['ie_raw']]
).fit()
b1 = iv.params['exp']
b1_se = iv.std_errors['exp']
b2 = beta_hat - b1
print(f"Allow neg ie: beta_1={b1:.4f} ({b1_se:.4f}), beta_2={b2:.4f}")
