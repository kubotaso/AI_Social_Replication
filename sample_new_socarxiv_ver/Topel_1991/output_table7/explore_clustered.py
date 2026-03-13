import pandas as pd, numpy as np, statsmodels.api as sm

df = pd.read_csv('data/psid_panel.csv')

# Education recoding
EDUC_MAP = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ed_yr'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_yr'] = df.loc[m, 'education_clean'].map(EDUC_MAP)

df['exp'] = (df['age'] - df['ed_yr'] - 6).clip(lower=1)
df['exp_sq'] = df['exp'] ** 2
df['union'] = df['union_member'].fillna(0)
df['dis'] = df['disabled'].fillna(0)

CPS = {1971:1.115,1972:1.113,1973:1.151,1974:1.167,1975:1.188,1976:1.117,
       1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,1982:1.103,1983:1.089}
df['lrw'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
ctrl = ['ed_yr','married','union','dis','lives_in_smsa',
        'region_ne','region_nc','region_south'] + yr_cols
base = ['exp','exp_sq','tenure_topel']

s = df.dropna(subset=ctrl + ['lrw'] + base).copy()

# Standard OLS
X = sm.add_constant(s[base + ctrl])
m_ols = sm.OLS(s['lrw'], X).fit()

# Clustered by person
m_cluster = sm.OLS(s['lrw'], X).fit(cov_type='cluster', cov_kwds={'groups': s['person_id']})

print("Standard OLS:")
for v in base:
    print(f"  {v}: coef={m_ols.params[v]:.6f}, se={m_ols.bse[v]:.6f}")
print(f"  R2={m_ols.rsquared:.4f}")

print("\nClustered by person:")
for v in base:
    print(f"  {v}: coef={m_cluster.params[v]:.6f}, se={m_cluster.bse[v]:.6f}")
print(f"  R2={m_cluster.rsquared:.4f}")

# HC1 robust
m_robust = sm.OLS(s['lrw'], X).fit(cov_type='HC1')
print("\nHC1 robust:")
for v in base:
    print(f"  {v}: coef={m_robust.params[v]:.6f}, se={m_robust.bse[v]:.6f}")

# Also try without education control but with person clusters
ctrl_noed = ['married','union','dis','lives_in_smsa',
             'region_ne','region_nc','region_south'] + yr_cols
X2 = sm.add_constant(s[base + ctrl_noed])
m2 = sm.OLS(s['lrw'], X2).fit(cov_type='cluster', cov_kwds={'groups': s['person_id']})
print("\nNo ed, clustered:")
for v in base:
    print(f"  {v}: coef={m2.params[v]:.6f}, se={m2.bse[v]:.6f}")
print(f"  R2={m2.rsquared:.4f}")
