import pandas as pd, numpy as np, statsmodels.api as sm

df = pd.read_csv('data/psid_panel.csv')

# Try different education approaches
EDUC_MAP_A = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
EDUC_MAP_B = {0:0,1:2,2:7,3:10,4:12,5:13,6:14,7:16,8:18,9:17}  # More spread

# Approach A: standard recoding, all years except 1975-1976
df['ed_a'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_a'] = df.loc[m, 'education_clean'].map(EDUC_MAP_A)

df['exp_a'] = (df['age'] - df['ed_a'] - 6).clip(lower=1)
df['exp_a_sq'] = df['exp_a'] ** 2

# Check experience statistics
print("Experience approach A:")
print(f"  Mean exp: {df['exp_a'].mean():.2f}")
print(f"  Mean ed: {df['ed_a'].mean():.2f}")
print(f"  Mean exp_sq: {df['exp_a_sq'].mean():.2f}")

# Approach B: use raw education for 1975-1976, map for others
# But 1975-1976 already in years

# Approach C: use experience as provided in panel (original)
df['exp_c'] = df['experience'].clip(lower=1)
df['exp_c_sq'] = df['exp_c'] ** 2
print("\nExperience approach C (as provided in panel):")
print(f"  Mean exp: {df['exp_c'].mean():.2f}")
print(f"  Mean exp_sq: {df['exp_c_sq'].mean():.2f}")

# Run regressions with each approach
CPS = {1971:1.115,1972:1.113,1973:1.151,1974:1.167,1975:1.188,1976:1.117,
       1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,1982:1.103,1983:1.089}
df['lrw'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['union'] = df['union_member'].fillna(0)
df['dis'] = df['disabled'].fillna(0)
yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971']
ctrl = ['ed_a','married','union','dis','lives_in_smsa','region_ne','region_nc','region_south'] + yr_cols
s = df.dropna(subset=ctrl + ['lrw','exp_a','exp_a_sq','tenure_topel']).copy()

# Approach A
Xa = sm.add_constant(s[['exp_a','exp_a_sq','tenure_topel'] + ctrl])
ma = sm.OLS(s['lrw'], Xa).fit()
print(f"\nApproach A: R2={ma.rsquared:.4f}, exp={ma.params['exp_a']:.5f}, exp_sq={ma.params['exp_a_sq']:.7f}")

# Approach C
ctrl_c = ctrl.copy()
Xc = sm.add_constant(s[['exp_c','exp_c_sq','tenure_topel'] + ctrl])
mc = sm.OLS(s['lrw'], Xc).fit()
print(f"Approach C: R2={mc.rsquared:.4f}, exp={mc.params['exp_c']:.5f}, exp_sq={mc.params['exp_c_sq']:.7f}")

# Check: what if we use NO education control (just experience)?
ctrl_noed = [c for c in ctrl if c != 'ed_a']
Xd = sm.add_constant(s[['exp_a','exp_a_sq','tenure_topel'] + ctrl_noed])
md = sm.OLS(s['lrw'], Xd).fit()
print(f"No ed ctrl: R2={md.rsquared:.4f}, exp={md.params['exp_a']:.5f}, exp_sq={md.params['exp_a_sq']:.7f}")

# What about different education year mappings?
# Category 3 = 9-11 grades. Try mapping to 9 instead of 10
EDUC_MAP_D = {0:0,1:3,2:7,3:9,4:12,5:13,6:14,7:16,8:17,9:17}
df['ed_d'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_d'] = df.loc[m, 'education_clean'].map(EDUC_MAP_D)
df['exp_d'] = (df['age'] - df['ed_d'] - 6).clip(lower=1)
df['exp_d_sq'] = df['exp_d'] ** 2
s2 = df.dropna(subset=['ed_d','lrw','exp_d','exp_d_sq','tenure_topel','married','union','dis','lives_in_smsa','region_ne','region_nc','region_south'] + yr_cols)
ctrl_d = ['ed_d','married','union','dis','lives_in_smsa','region_ne','region_nc','region_south'] + yr_cols
Xd2 = sm.add_constant(s2[['exp_d','exp_d_sq','tenure_topel'] + ctrl_d])
md2 = sm.OLS(s2['lrw'], Xd2).fit()
print(f"Alt ed map: R2={md2.rsquared:.4f}, exp={md2.params['exp_d']:.5f}, exp_sq={md2.params['exp_d_sq']:.7f}")
