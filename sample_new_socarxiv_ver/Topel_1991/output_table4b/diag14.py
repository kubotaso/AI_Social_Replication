"""
Fix education at person level to ensure d_exp=1, and check N=5790 issue.
Also try using paper's step 1 coefficients but with full sample for step 2.
"""
import pandas as pd, numpy as np, statsmodels.api as sm
from linearmodels.iv import IV2SLS

df = pd.read_csv('data/psid_panel.csv')
df['pnum'] = df['person_id'] % 1000
df = df[df['pnum'].isin([1, 170])].copy()
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
df['education_years'] = np.nan
for yr in df['year'].unique():
    mask = df['year'] == yr
    if yr in [1975, 1976]:
        df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
    else:
        df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)
df = df.dropna(subset=['education_years']).copy()

# Fix education at person level (use max/mode)
person_edu = df.groupby('person_id')['education_years'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.max())
df['edu_fixed'] = df['person_id'].map(person_edu)

df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df['exp_fixed'] = (df['age'] - df['edu_fixed'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'].copy()

CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
       1982:1.103,1983:1.089}
df['cps'] = df['year'].map(CPS)
df['lrw'] = df['log_hourly_wage'] - np.log(df['cps'])

df = df[(df['age'] >= 18) & (df['age'] <= 60)]
df = df[df['hourly_wage'] > 0]
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())]
df = df[df['tenure'] >= 1]
df = df.dropna(subset=['lrw', 'experience', 'tenure'])
df = df.sort_values(['person_id','job_id','year']).reset_index(drop=True)
df['init_exp'] = (df['experience'] - df['tenure']).clip(lower=0)
df['init_exp_fixed'] = (df['exp_fixed'] - df['tenure']).clip(lower=0)

print(f'Full sample: {len(df)} obs, {df["person_id"].nunique()} persons')

# Check the N issue
ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_cols = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_use = [c for c in yr_cols if df[c].std() > 1e-10][1:]

for c in ctrl + yr_use:
    na_count = df[c].isna().sum()
    if na_count > 0:
        print(f'  {c}: {na_count} NaN')

lev = df.dropna(subset=['lrw', 'experience', 'init_exp'] + ctrl).copy()
print(f'After dropna for step 2: {len(lev)} obs')

# Ah, let me check if the year dummies have NaN
print(f'Year dummy NaN check:')
for c in yr_use[:3]:
    print(f'  {c}: {df[c].isna().sum()} NaN')
print(f'Any yr NaN: {df[yr_use].isna().any().any()}')
print()

# Use paper step 1 values
bhat = 0.1258
g2 = -0.004592
g3 = 0.0001846
g4 = -0.00000245
d2 = -0.006051
d3 = 0.0002067
d4 = -0.00000238

T = lev['tenure'].values
X_exp = lev['experience'].values
lev['w_star'] = (lev['lrw'] - bhat*T - g2*T**2 - g3*T**3 - g4*T**4
                 - d2*X_exp**2 - d3*X_exp**3 - d4*X_exp**4)

# Check rank with full set
exog = sm.add_constant(lev[ctrl + yr_use])
r = np.linalg.matrix_rank(exog.values)
print(f'Rank of exog: {r}, ncols: {exog.shape[1]}')
current_yr = yr_use.copy()
while r < exog.shape[1] and current_yr:
    removed = current_yr.pop()
    exog = sm.add_constant(lev[ctrl + current_yr])
    r = np.linalg.matrix_rank(exog.values)
    print(f'  After removing {removed}: rank={r}, ncols={exog.shape[1]}')

# IV step 2
dep = lev['w_star']
exog_final = sm.add_constant(lev[ctrl + current_yr])
endog = lev[['experience']]
instruments = lev[['init_exp']]

iv = IV2SLS(dep, exog_final, endog, instruments).fit(cov_type='unadjusted')
print(f'\nStep 2 IV (paper step 1, CPS wages, full N):')
print(f'  N = {len(lev)}')
print(f'  beta_1 = {iv.params["experience"]:.4f} ({iv.std_errors["experience"]:.4f})')
print(f'  beta_2 = {bhat - iv.params["experience"]:.4f}')
print(f'  Paper: beta_1=0.0713, beta_2=0.0545')
print()

# Now try with fixed education for experience
X_exp_f = lev['exp_fixed'].values
lev['w_star_f'] = (lev['lrw'] - bhat*T - g2*T**2 - g3*T**3 - g4*T**4
                   - d2*X_exp_f**2 - d3*X_exp_f**3 - d4*X_exp_f**4)
dep_f = lev['w_star_f']
endog_f = lev[['exp_fixed']]
instruments_f = lev[['init_exp_fixed']]
iv_f = IV2SLS(dep_f, exog_final, endog_f, instruments_f).fit(cov_type='unadjusted')
print(f'Step 2 IV (paper step 1, CPS, fixed edu):')
print(f'  N = {len(lev)}')
print(f'  beta_1 = {iv_f.params["exp_fixed"]:.4f} ({iv_f.std_errors["exp_fixed"]:.4f})')
print(f'  beta_2 = {bhat - iv_f.params["exp_fixed"]:.4f}')
