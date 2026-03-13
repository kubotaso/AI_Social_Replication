import pandas as pd, numpy as np, statsmodels.api as sm
df = pd.read_csv('data/psid_panel.csv')
EDUC_MAP = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ed_yr'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_yr'] = df.loc[m, 'education_clean'].map(EDUC_MAP)

df['exp'] = (df['age'] - df['ed_yr'] - 6).clip(lower=1)
df['exp_sq'] = df['exp'] ** 2

GNP = {1971:40.5,1972:41.8,1973:44.4,1974:48.9,1975:53.6,1976:56.9,
       1977:60.6,1978:65.2,1979:72.6,1980:82.4,1981:90.9,1982:100.0,1983:100.0}
CPS = {1971:1.115,1972:1.113,1973:1.151,1974:1.167,1975:1.188,1976:1.117,
       1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,1982:1.103,1983:1.089}

df['lrw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lrw_gnp'] = df['log_hourly_wage'] - np.log(df['year'].map(GNP)/100)
df['lrw_both'] = df['log_hourly_wage'] - np.log(df['year'].map(GNP)/100) - np.log(df['year'].map(CPS))
df['lrw_none'] = df['log_hourly_wage']

df['union'] = df['union_member'].fillna(0)
df['dis'] = df['disabled'].fillna(0)
yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971']
base = ['exp','exp_sq','tenure_topel','ed_yr','married','union','dis',
        'lives_in_smsa','region_ne','region_nc','region_south'] + yr_cols
s = df.dropna(subset=base + ['lrw_cps'])
X = sm.add_constant(s[base])

for name, dep in [('CPS only','lrw_cps'),('GNP only','lrw_gnp'),
                   ('GNP+CPS','lrw_both'),('No deflation','lrw_none')]:
    m = sm.OLS(s[dep], X).fit()
    print(f'{name:15s}: R2={m.rsquared:.4f}, exp={m.params["exp"]:.5f}, '
          f'exp_sq={m.params["exp_sq"]:.7f}, tenure={m.params["tenure_topel"]:.5f}')
