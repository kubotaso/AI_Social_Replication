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

GNP = {1971:40.5,1972:41.8,1973:44.4,1974:48.9,1975:53.6,1976:56.9,
       1977:60.6,1978:65.2,1979:72.6,1980:82.4,1981:90.9,1982:100.0,1983:100.0}
CPS = {1971:1.115,1972:1.113,1973:1.151,1974:1.167,1975:1.188,1976:1.117,
       1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,1982:1.103,1983:1.089}

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971']

# GNP+CPS deflation
df['gnp_d'] = df['year'].map(GNP)
df['cps_d'] = df['year'].map(CPS)
df['lrw_gnpcps'] = df['log_hourly_wage'] - np.log(df['gnp_d']/100) - np.log(df['cps_d'])
df['lrw_gnp'] = df['log_hourly_wage'] - np.log(df['gnp_d']/100)
df['lrw_cps'] = df['log_hourly_wage'] - np.log(df['cps_d'])

ctrl_full = ['ed_yr','married','union','dis','lives_in_smsa',
             'region_ne','region_nc','region_south'] + yr_cols

s = df.dropna(subset=ctrl_full + ['lrw_gnpcps','exp','exp_sq','tenure_topel']).copy()
base = ['exp','exp_sq','tenure_topel']

# KEY: let's check how many region dummies we actually have in the data
print("Region columns and unique values:")
for c in ['region_ne','region_nc','region_south','region_west']:
    print(f"  {c}: mean={s[c].mean():.3f}")
print()

# Check if Table 4 mentions 8 region dummies
# The instruction says "census region dummies (8 regions, but we have 4 region dummies)"
# Paper has 8 census regions. We only have 4 (NE, NC, South, West).
# This missing granularity could explain lower R2 vs. paper.

# Let's also check occupation dummies
occ_cols = [c for c in df.columns if c.startswith('occ_')]
print(f"Occupation dummy columns: {occ_cols}")
print(f"Occupation values: {df['occ_1digit'].dropna().unique()}")

# Try with occupation dummies
ctrl_occ = ctrl_full + [c for c in occ_cols if c != 'occ_0']  # drop reference category

s2 = df.dropna(subset=ctrl_occ + ['lrw_cps','exp','exp_sq','tenure_topel']).copy()
X_occ = sm.add_constant(s2[base + ctrl_occ])
m_occ = sm.OLS(s2['lrw_cps'], X_occ).fit()
print(f"\nWith occ dummies, CPS+yr: R2={m_occ.rsquared:.4f}, exp={m_occ.params['exp']:.5f}, exp_sq={m_occ.params['exp_sq']:.7f}, tenure={m_occ.params['tenure_topel']:.5f}")

# Maybe the R2 of 0.422 refers to an adjusted R2?
# With year dummies and education, adjusted R2 would be lower
X_base = sm.add_constant(s[base + ctrl_full])
m_base = sm.OLS(s['lrw_cps'], X_base).fit()
print(f"\nCPS+yr (full): R2={m_base.rsquared:.4f}, adj_R2={m_base.rsquared_adj:.4f}")
m_gnp = sm.OLS(s['lrw_gnp'], sm.add_constant(s[base + ctrl_full])).fit()
print(f"GNP+yr (full): R2={m_gnp.rsquared:.4f}, adj_R2={m_gnp.rsquared_adj:.4f}")
m_gnpcps = sm.OLS(s['lrw_gnpcps'], sm.add_constant(s[base + ctrl_full])).fit()
print(f"GNP+CPS+yr:    R2={m_gnpcps.rsquared:.4f}, adj_R2={m_gnpcps.rsquared_adj:.4f}")

# Try: using the raw 'experience' column from the dataset
df['exp_raw'] = df['experience'].clip(lower=1)
df['exp_raw_sq'] = df['exp_raw'] ** 2
s3 = df.dropna(subset=['ed_yr','married','union','dis','lives_in_smsa',
                        'region_ne','region_nc','region_south','lrw_cps',
                        'exp_raw','exp_raw_sq','tenure_topel'] + yr_cols).copy()
X_raw = sm.add_constant(s3[['exp_raw','exp_raw_sq','tenure_topel'] + ctrl_full])
m_raw = sm.OLS(s3['lrw_cps'], X_raw).fit()
print(f"\nRaw exp, CPS+yr: R2={m_raw.rsquared:.4f}, exp={m_raw.params['exp_raw']:.5f}, exp_sq={m_raw.params['exp_raw_sq']:.7f}")

# Check: what's the paper's actual sample? They say 13,128.
# We have 13,922 after dropna. Maybe we need to trim harder.
print(f"\nOur N: {len(s)}")
print(f"Paper N: ~13,128")
