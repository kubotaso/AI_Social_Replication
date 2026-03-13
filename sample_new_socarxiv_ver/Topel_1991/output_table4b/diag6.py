"""
Fix: Use person-level fixed education (e.g., mode or value from 1975/76 when actual years are available)
"""
import pandas as pd, numpy as np, statsmodels.api as sm

df = pd.read_csv('data/psid_panel.csv')
df = df[~df['region'].isin([5, 6])]

EDUC = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}

# Strategy: For each person, use education from 1975 or 1976 if available (actual years)
# Otherwise, use the modal categorical value and remap

# First, create education_years for each row
df['ey_raw'] = df['education_clean'].copy()

# For 1975-1976: already in years
mask_years = df['year'].isin([1975, 1976])
# For other years: categorical -> remap
mask_cat = ~mask_years
df.loc[mask_cat, 'ey_raw'] = df.loc[mask_cat, 'education_clean'].map(EDUC)

# Now get a fixed education per person
# Priority: use 1975/1976 value if available
educ_7576 = df[mask_years].groupby('person_id')['ey_raw'].first()

# For everyone else, use the mode of remapped values
educ_other = df[mask_cat].groupby('person_id')['ey_raw'].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)

# Merge
educ_fixed = educ_7576.reindex(df['person_id'].unique())
# Fill missing with the modal value from other years
missing = educ_fixed[educ_fixed.isna()].index
for pid in missing:
    if pid in educ_other.index:
        educ_fixed[pid] = educ_other[pid]

df['ey_fixed'] = df['person_id'].map(educ_fixed)
df = df.dropna(subset=['ey_fixed'])

df['exp'] = (df['age'] - df['ey_fixed'] - 6).clip(lower=0)
df['ie'] = (df['exp'] - df['tenure_topel']).clip(lower=0)

# Check d_exp consistency
df = df.sort_values(['person_id','year'])
df['prev_exp'] = df.groupby('person_id')['exp'].shift(1)
df['prev_yr'] = df.groupby('person_id')['year'].shift(1)
consec = df[(df['prev_yr']==df['year']-1)]
d_exp = consec['exp'] - consec['prev_exp']
print(f"With fixed education: d_exp != 1 count: {(d_exp!=1).sum()} out of {len(consec)}")
print(f"d_exp value counts:")
print(d_exp.value_counts().head(5))
print()

# Now try the step 2 regression with paper's coefficients
CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
       1982:1.103,1983:1.089}
GNP = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,1973:44.4,1974:48.9,
       1975:53.6,1976:56.9,1977:60.6,1978:65.2,1979:72.6,1980:82.4,1981:90.9,1982:100.0}

df['cps'] = df['year'].map(CPS)
df['gnp'] = (df['year']-1).map(GNP)
df['lrw'] = df['log_hourly_wage'] - np.log(df['gnp']/100.0) - np.log(df['cps'])

for c in ['married','union_member','disabled','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

beta_hat = 0.1258
gamma2 = -0.004592
gamma3 = 0.0001846
gamma4 = -0.00000245
delta2 = -0.004067
delta3 = 0.0000989
delta4 = 0.00000089

T = df['tenure_topel'].values.astype(float)
X = df['exp'].values.astype(float)

w_star = (df['lrw'].values - beta_hat*T - gamma2*T**2 - gamma3*T**3 - gamma4*T**4
          - delta2*X**2 - delta3*X**3 - delta4*X**4)
df['w_star'] = w_star

controls = ['ey_fixed', 'married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south']
yr = pd.get_dummies(df['year'], prefix='yr', dtype=float)
yr_cols = sorted(yr.columns.tolist())[1:]

df2 = df.reset_index(drop=True)
X_reg = df2[['ie'] + controls].copy()
for c in yr_cols:
    X_reg[c] = yr[c].values
X_reg = sm.add_constant(X_reg)

valid = X_reg.notna().all(axis=1) & df2['w_star'].notna()
model = sm.OLS(df2.loc[valid, 'w_star'], X_reg[valid]).fit()

b1 = model.params['ie']
b1_se = model.bse['ie']
b2 = beta_hat - b1
print(f"Fixed education: beta_1 = {b1:.4f} ({b1_se:.4f})")
print(f"beta_2 = {beta_hat} - {b1:.4f} = {b2:.4f}")
print(f"Expected: beta_1 = 0.0713 (0.0181), beta_2 = 0.0545")
print(f"N = {int(model.nobs)}")
