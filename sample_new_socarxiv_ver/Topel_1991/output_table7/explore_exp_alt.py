import pandas as pd, numpy as np, statsmodels.api as sm
df = pd.read_csv('data/psid_panel.csv')
EDUC_MAP = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ed_yr'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_yr'] = df.loc[m, 'education_clean'].map(EDUC_MAP)

# Standard experience
df['exp_std'] = (df['age'] - df['ed_yr'] - 6).clip(lower=1)

# Try different "starting age" assumptions
# Maybe 5 instead of 6?
df['exp_5'] = (df['age'] - df['ed_yr'] - 5).clip(lower=1)
df['exp_7'] = (df['age'] - df['ed_yr'] - 7).clip(lower=1)

# What if Topel uses a different education mapping?
# Try: 0->0, 1->2, 2->6, 3->9, 4->12, 5->13, 6->14, 7->16, 8->17
EDUC_MAP_B = {0:0, 1:2, 2:6, 3:9, 4:12, 5:13, 6:14, 7:16, 8:17, 9:17}
df['ed_yr_b'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_yr_b'] = df.loc[m, 'education_clean'].map(EDUC_MAP_B)
df['exp_b'] = (df['age'] - df['ed_yr_b'] - 6).clip(lower=1)

# What if experience = age - 18 (ignoring education)?
df['exp_age18'] = (df['age'] - 18).clip(lower=0)

df['union'] = df['union_member'].fillna(0)
df['dis'] = df['disabled'].fillna(0)
CPS = {1971:1.115,1972:1.113,1973:1.151,1974:1.167,1975:1.188,1976:1.117,
       1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,1982:1.103,1983:1.089}
df['lrw'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
ctrl = ['ed_yr','married','union','dis','lives_in_smsa',
        'region_ne','region_nc','region_south'] + yr_cols

s = df.dropna(subset=ctrl + ['lrw','tenure_topel']).copy()

for exp_col, label in [('exp_std','age-ed-6'), ('exp_5','age-ed-5'), ('exp_7','age-ed-7'),
                        ('exp_b','alt ed map'), ('exp_age18','age-18')]:
    s2 = s.dropna(subset=[exp_col]).copy()
    s2['esq'] = s2[exp_col] ** 2
    X = sm.add_constant(s2[[exp_col,'esq','tenure_topel'] + ctrl])
    m = sm.OLS(s2['lrw'], X).fit()
    print(f"{label:15s}: exp={m.params[exp_col]:.5f}, exp_sq={m.params['esq']:.7f}, "
          f"ten={m.params['tenure_topel']:.5f}, R2={m.rsquared:.4f}")

# Also try: what if we use LAGGED tenure (tenure at start of period, not end)?
# If wages are measured at interview time and tenure is as of interview,
# maybe we need tenure - 1
s['ten_lag'] = (s['tenure_topel'] - 1).clip(lower=0)
s['esq'] = s['exp_std'] ** 2
X = sm.add_constant(s[['exp_std','esq','ten_lag'] + ctrl])
m = sm.OLS(s['lrw'], X).fit()
print(f"\nLagged tenure:  exp={m.params['exp_std']:.5f}, exp_sq={m.params['esq']:.7f}, "
      f"ten={m.params['ten_lag']:.5f}, R2={m.rsquared:.4f}")

# One more: experience / 10 scaling?
# If exp is scaled differently, the coefficient changes
# exp_sq coef would be proportional to 1/scale^2
# For exp_sq coef to be -0.00079 instead of -0.00053,
# we'd need scale = sqrt(0.00053/0.00079) = 0.819
# That means experience should be about 82% of current value
# Current mean exp = 18.9, target exp = 18.9 * 0.82 = 15.5
# This could happen if education is higher by ~3 years on average
# Or if starting age is 9 instead of 6

df['exp_9'] = (df['age'] - df['ed_yr'] - 9).clip(lower=1)
s3 = df.dropna(subset=['exp_9'] + ctrl + ['lrw','tenure_topel']).copy()
s3['esq9'] = s3['exp_9'] ** 2
X = sm.add_constant(s3[['exp_9','esq9','tenure_topel'] + ctrl])
m = sm.OLS(s3['lrw'], X).fit()
print(f"\nage-ed-9:       exp={m.params['exp_9']:.5f}, exp_sq={m.params['esq9']:.7f}, "
      f"ten={m.params['tenure_topel']:.5f}, R2={m.rsquared:.4f}")

print(f"\nTargets: exp=0.0418, exp_sq=-0.00079, ten=0.0138, R2=0.422")
