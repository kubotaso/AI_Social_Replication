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

# Try: GNP deflation with year dummies vs without
df['lrw_gnp'] = df['log_hourly_wage'] - np.log(df['year'].map(GNP)/100)
df['lrw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lrw_none'] = df['log_hourly_wage']

ctrl_with_yr = ['ed_yr','married','union','dis','lives_in_smsa',
                'region_ne','region_nc','region_south'] + yr_cols
ctrl_no_yr = ['ed_yr','married','union','dis','lives_in_smsa',
              'region_ne','region_nc','region_south']

s = df.dropna(subset=ctrl_with_yr + ['lrw_gnp','lrw_cps','lrw_none','exp','exp_sq','tenure_topel']).copy()
base = ['exp','exp_sq','tenure_topel']

configs = [
    ('GNP+yrdum', 'lrw_gnp', ctrl_with_yr),
    ('GNP no yrdum', 'lrw_gnp', ctrl_no_yr),
    ('CPS+yrdum', 'lrw_cps', ctrl_with_yr),
    ('CPS no yrdum', 'lrw_cps', ctrl_no_yr),
    ('None+yrdum', 'lrw_none', ctrl_with_yr),
    ('None no yrdum', 'lrw_none', ctrl_no_yr),
]

for name, dep, ctrl in configs:
    X = sm.add_constant(s[base + ctrl])
    m = sm.OLS(s[dep], X).fit()
    print(f'{name:20s}: R2={m.rsquared:.4f}, exp={m.params["exp"]:.5f}, '
          f'exp_sq={m.params["exp_sq"]:.7f}, tenure={m.params["tenure_topel"]:.5f}')

# Also try: dropping education from controls
print("\n--- Without education as control ---")
ctrl_no_ed_yrdum = ['married','union','dis','lives_in_smsa',
                    'region_ne','region_nc','region_south'] + yr_cols
ctrl_no_ed_noyrdum = ['married','union','dis','lives_in_smsa',
                      'region_ne','region_nc','region_south']

configs2 = [
    ('GNP+yr noed', 'lrw_gnp', ctrl_no_ed_yrdum),
    ('GNP noyr noed', 'lrw_gnp', ctrl_no_ed_noyrdum),
    ('CPS+yr noed', 'lrw_cps', ctrl_no_ed_yrdum),
    ('None+yr noed', 'lrw_none', ctrl_no_ed_yrdum),
]

for name, dep, ctrl in configs2:
    X = sm.add_constant(s[base + ctrl])
    m = sm.OLS(s[dep], X).fit()
    print(f'{name:20s}: R2={m.rsquared:.4f}, exp={m.params["exp"]:.5f}, '
          f'exp_sq={m.params["exp_sq"]:.7f}, tenure={m.params["tenure_topel"]:.5f}')

# Key insight: tenure coef in paper is 0.0138 with SE 0.0052
# Very different from our 0.024 with SE 0.0015
# Maybe the tenure variable needs to be divided by some factor?
# Or maybe tenure needs to start at 0?
print("\n--- Tenure starting at 0 ---")
s['ten0'] = s['tenure_topel'] - 1
s['ten0'] = s['ten0'].clip(lower=0)
X = sm.add_constant(s[['exp','exp_sq','ten0'] + ctrl_with_yr])
m = sm.OLS(s['lrw_cps'], X).fit()
print(f'ten0 CPS+yr:  R2={m.rsquared:.4f}, exp={m.params["exp"]:.5f}, exp_sq={m.params["exp_sq"]:.7f}, tenure={m.params["ten0"]:.5f}')

# What if tenure is in different units? The paper says tenure SE = 0.0052
# with coefficient 0.0138. My SE is 0.0015, much smaller.
# The SE is about 3x too small, which suggests either the data has more precision
# or the model is missing some source of variance.
# Maybe there are fewer year dummies? Paper uses 1968-1983 but our data starts 1971.
print(f"\nYear range: {s['year'].min()} - {s['year'].max()}")
print(f"Year dummies used: {len(yr_cols)}")
