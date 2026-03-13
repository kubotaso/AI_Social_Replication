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
GNP = {1971:40.5,1972:41.8,1973:44.4,1974:48.9,1975:53.6,1976:56.9,
       1977:60.6,1978:65.2,1979:72.6,1980:82.4,1981:90.9,1982:100.0,1983:100.0}

df['lrw_cps'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['lrw_gnp'] = df['log_hourly_wage'] - np.log(df['year'].map(GNP)/100)

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971']
ctrl = ['ed_yr','married','union','dis','lives_in_smsa',
        'region_ne','region_nc','region_south'] + yr_cols
base = ['exp','exp_sq','tenure_topel']

# 1. Try trimming hourly wage outliers more aggressively
s = df.dropna(subset=ctrl + ['lrw_cps'] + base).copy()
print(f"Base sample: N={len(s)}")
print(f"hourly_wage quantiles:")
print(s['hourly_wage'].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
print()

# Trim top/bottom 1% of wages
hw_01 = s['hourly_wage'].quantile(0.01)
hw_99 = s['hourly_wage'].quantile(0.99)
s_trim = s[(s['hourly_wage'] >= hw_01) & (s['hourly_wage'] <= hw_99)].copy()
print(f"After 1% trim: N={len(s_trim)}")
X = sm.add_constant(s_trim[base + ctrl])
m = sm.OLS(s_trim['lrw_cps'], X).fit()
print(f"  CPS+yr: R2={m.rsquared:.4f}, exp={m.params['exp']:.5f}, exp_sq={m.params['exp_sq']:.7f}")

# 2. Try experience >= 1 and <= 40
s_exp = s[(s['exp'] >= 1) & (s['exp'] <= 40)].copy()
print(f"\nAfter exp 1-40: N={len(s_exp)}")
X = sm.add_constant(s_exp[base + ctrl])
m = sm.OLS(s_exp['lrw_cps'], X).fit()
print(f"  CPS+yr: R2={m.rsquared:.4f}, exp={m.params['exp']:.5f}, exp_sq={m.params['exp_sq']:.7f}")

# 3. Try tenure >= 1 (already applied)
print(f"\nTenure range: {s['tenure_topel'].min()} - {s['tenure_topel'].max()}")

# 4. Let me check the Topel (1991) sample more carefully
# Paper says: "white male household heads" who are "employed full-time
# (at least 1500 hours per year)"
# Let's check hours
print(f"\nHours worked: mean={df['hours'].mean():.0f}, min={df['hours'].min()}, max={df['hours'].max()}")
s_ft = s[s['hours'] >= 1500].copy()
print(f"After hours >= 1500: N={len(s_ft)}")
X = sm.add_constant(s_ft[base + ctrl])
m = sm.OLS(s_ft['lrw_cps'], X).fit()
print(f"  CPS+yr: R2={m.rsquared:.4f}, exp={m.params['exp']:.5f}, exp_sq={m.params['exp_sq']:.7f}, tenure={m.params['tenure_topel']:.5f}")

# Also with GNP deflation
m2 = sm.OLS(s_ft['lrw_gnp'], sm.add_constant(s_ft[base + ctrl])).fit()
print(f"  GNP+yr: R2={m2.rsquared:.4f}, exp={m2.params['exp']:.5f}, exp_sq={m2.params['exp_sq']:.7f}")

# 5. Try hours >= 500
s_500 = s[s['hours'] >= 500].copy()
print(f"\nAfter hours >= 500: N={len(s_500)}")
X = sm.add_constant(s_500[base + ctrl])
m = sm.OLS(s_500['lrw_cps'], X).fit()
print(f"  CPS+yr: R2={m.rsquared:.4f}, exp={m.params['exp']:.5f}, exp_sq={m.params['exp_sq']:.7f}")

# 6. Full time AND wage trim
s_combo = s[(s['hours'] >= 1500) & (s['hourly_wage'] >= hw_01) & (s['hourly_wage'] <= hw_99)].copy()
print(f"\nFT + 1% trim: N={len(s_combo)}")
X = sm.add_constant(s_combo[base + ctrl])
m = sm.OLS(s_combo['lrw_cps'], X).fit()
print(f"  CPS+yr: R2={m.rsquared:.4f}, exp={m.params['exp']:.5f}, exp_sq={m.params['exp_sq']:.7f}")

# 7. Let's also check if maybe the issue is year_1968-1970 dummies
# Our data starts at 1971, so yr_1968 through yr_1970 are always 0
yr_active = [c for c in yr_cols if s[c].sum() > 0]
yr_zero = [c for c in yr_cols if s[c].sum() == 0]
print(f"\nActive year dummies ({len(yr_active)}): {yr_active}")
print(f"Zero year dummies ({len(yr_zero)}): {yr_zero}")

# 8. Try with only active year dummies
ctrl_active = ['ed_yr','married','union','dis','lives_in_smsa',
               'region_ne','region_nc','region_south'] + yr_active
X = sm.add_constant(s[base + ctrl_active])
m = sm.OLS(s['lrw_cps'], X).fit()
print(f"\nActive yr dummies: R2={m.rsquared:.4f}, N={int(m.nobs)}")
