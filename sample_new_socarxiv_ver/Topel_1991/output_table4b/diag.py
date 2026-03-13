import pandas as pd, numpy as np, statsmodels.api as sm

df = pd.read_csv('data/psid_panel.csv')
df = df[~df['region'].isin([5, 6])]
EDUC = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ey'] = df['education_clean'].copy()
m = ~df['year'].isin([1975, 1976])
df.loc[m, 'ey'] = df.loc[m, 'education_clean'].map(EDUC)
df = df.dropna(subset=['ey'])
df['exp'] = (df['age'] - df['ey'] - 6).clip(lower=0)
df['ie'] = (df['exp'] - df['tenure_topel']).clip(lower=0)

print("Correlation(exp, ie):", df[['exp','ie']].corr().iloc[0,1])
print("exp stats:", df['exp'].describe().to_string())
print()
print("ie stats:", df['ie'].describe().to_string())
print()
print("tenure stats:", df['tenure_topel'].describe().to_string())
print()

# Run first stage: exp ~ ie + controls + const
controls = ['ey', 'married', 'union_member', 'disabled', 'region_ne', 'region_nc', 'region_south']
for c in controls:
    df[c] = df[c].fillna(0)

yr = pd.get_dummies(df['year'], prefix='yr', dtype=float)
yr_cols = sorted(yr.columns.tolist())[1:]

X = df[['ie'] + controls].copy()
for c in yr_cols:
    X[c] = yr[c].values
X = sm.add_constant(X)

valid = X.notna().all(axis=1)
fs = sm.OLS(df.loc[valid, 'exp'], X[valid]).fit()
print("\nFirst stage R-squared:", fs.rsquared)
print("First stage coef on ie:", fs.params['ie'])
print("First stage coef on ie SE:", fs.bse['ie'])
print("First stage F-stat:", fs.fvalue)
print()

# The issue: if exp = ie + tenure, and tenure is NOT in the first stage,
# then ie should predict exp well, but the residual variation is tenure.
# The IV estimate should be: E[y|ie] / E[x|ie] using the projected x.
print("R-squared of exp ~ ie alone:", sm.OLS(df.loc[valid,'exp'], sm.add_constant(df.loc[valid,'ie'])).fit().rsquared)
