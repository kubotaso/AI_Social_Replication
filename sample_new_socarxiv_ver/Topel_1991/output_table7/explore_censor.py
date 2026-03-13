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
df['union'] = df['union_member'].fillna(0)
df['dis'] = df['disabled'].fillna(0)

CPS = {1971:1.115,1972:1.113,1973:1.151,1974:1.167,1975:1.188,1976:1.117,
       1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,1982:1.103,1983:1.089}
df['lrw'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
ctrl = ['ed_yr','married','union','dis','lives_in_smsa',
        'region_ne','region_nc','region_south'] + yr_cols

# Completed tenure
df['ct'] = df.groupby('job_id')['tenure_topel'].transform('max')
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)

# Try x_censor = just the censor dummy (not interacted with ct)
s = df.dropna(subset=ctrl + ['lrw','exp','exp_sq','tenure_topel','ct','censor']).copy()

# Col (2) restricted: experience, exp_sq, tenure, ct, censor
X2a = sm.add_constant(s[['exp','exp_sq','tenure_topel','ct','censor'] + ctrl])
m2a = sm.OLS(s['lrw'], X2a).fit()
print("Col 2 with censor DUMMY (not ct*censor):")
for v in ['exp','exp_sq','tenure_topel','ct','censor']:
    print(f"  {v}: coef={m2a.params[v]:.6f}, se={m2a.bse[v]:.6f}")
print(f"  R2: {m2a.rsquared:.4f}")

# Compare with ct*censor
s['ct_x_censor'] = s['ct'] * s['censor']
X2b = sm.add_constant(s[['exp','exp_sq','tenure_topel','ct','ct_x_censor'] + ctrl])
m2b = sm.OLS(s['lrw'], X2b).fit()
print("\nCol 2 with ct*censor:")
for v in ['exp','exp_sq','tenure_topel','ct','ct_x_censor']:
    print(f"  {v}: coef={m2b.params[v]:.6f}, se={m2b.bse[v]:.6f}")
print(f"  R2: {m2b.rsquared:.4f}")

# The paper says SE for x_censor is 0.0073 in col 2
# With censor dummy: SE would be larger (binary variable)
# With ct*censor: SE would be smaller

# Now try col (3) unrestricted with censor dummy
s['ct_x_esq'] = s['ct'] * s['exp_sq']
s['ct_x_t'] = s['ct'] * s['tenure_topel']
X3a = sm.add_constant(s[['exp','exp_sq','tenure_topel','ct','censor','ct_x_esq','ct_x_t'] + ctrl])
m3a = sm.OLS(s['lrw'], X3a).fit()
print("\nCol 3 with censor DUMMY:")
for v in ['exp','exp_sq','tenure_topel','ct','censor','ct_x_esq','ct_x_t']:
    print(f"  {v}: coef={m3a.params[v]:.6f}, se={m3a.bse[v]:.6f}")
print(f"  R2: {m3a.rsquared:.4f}")

# Paper target for col(2):
# exp: .0379, exp_sq: -.00069, tenure: -.0015 (SE .0015)
# observed_ct: .0165 (SE .0016), x_censor: -.0025 (SE .0073)
# R2: .428

# Let's check: what if we SCALE tenure by 10?
# Paper reports tenure SE = 0.0015 but our SE = 0.002.
# Not a scaling issue since they report similar scale coefficients.

# Check raw ct (no interactions) SEs to see if censor dummy gives right SE
print("\n\nCompare censor SE across definitions:")
print(f"  censor dummy SE: {m2a.bse['censor']:.6f} (target ~0.0073)")
print(f"  ct*censor SE: {m2b.bse['ct_x_censor']:.6f}")
