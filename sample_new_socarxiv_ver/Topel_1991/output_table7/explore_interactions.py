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

# Completed tenure = max tenure for each job
ct = df.groupby('job_id')['tenure_topel'].transform('max')
df['ct'] = ct

# The key interaction variables
df['ct_x_esq'] = df['ct'] * df['exp_sq']
df['ct_x_t'] = df['ct'] * df['tenure_topel']

print('Variable statistics:')
print(f"ct: mean={df['ct'].mean():.2f}, std={df['ct'].std():.2f}")
print(f"exp_sq: mean={df['exp_sq'].mean():.2f}, std={df['exp_sq'].std():.2f}")
print(f"tenure: mean={df['tenure_topel'].mean():.2f}, std={df['tenure_topel'].std():.2f}")
print(f"ct_x_esq: mean={df['ct_x_esq'].mean():.2f}, std={df['ct_x_esq'].std():.2f}")
print(f"ct_x_t: mean={df['ct_x_t'].mean():.2f}, std={df['ct_x_t'].std():.2f}")

# Now the issue: the coefficient on ct_x_t is -0.0027 but paper says +0.0142
# This could mean the paper uses a DIFFERENT completed tenure variable
# Maybe the paper defines completed tenure DIFFERENTLY:
#   - As remaining tenure (completed - current)?
#   - As log completed tenure?
#   - As completed tenure - mean?

# Let's try: remaining tenure = completed - current
df['remaining'] = df['ct'] - df['tenure_topel']
df['remaining'] = df['remaining'].clip(lower=0)
df['rem_x_esq'] = df['remaining'] * df['exp_sq']
df['rem_x_t'] = df['remaining'] * df['tenure_topel']

# Censor
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['ct_x_censor'] = df['ct'] * df['censor']
df['rem_x_censor'] = df['remaining'] * df['censor']

CPS = {1971:1.115,1972:1.113,1973:1.151,1974:1.167,1975:1.188,1976:1.117,
       1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,1982:1.103,1983:1.089}
df['lrw'] = df['log_hourly_wage'] - np.log(df['year'].map(CPS))
df['union'] = df['union_member'].fillna(0)
df['dis'] = df['disabled'].fillna(0)
yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
ctrl = ['ed_yr','married','union','dis','lives_in_smsa',
        'region_ne','region_nc','region_south'] + yr_cols

s = df.dropna(subset=ctrl + ['lrw','exp','exp_sq','tenure_topel','ct','remaining']).copy()

# Col 3 with completed tenure
X3a = sm.add_constant(s[['exp','exp_sq','tenure_topel','ct','ct_x_censor','ct_x_esq','ct_x_t'] + ctrl])
m3a = sm.OLS(s['lrw'], X3a).fit()
print("\nCol 3 with completed tenure interactions:")
for v in ['exp','exp_sq','tenure_topel','ct','ct_x_censor','ct_x_esq','ct_x_t']:
    print(f"  {v}: coef={m3a.params[v]:.6f}, se={m3a.bse[v]:.6f}")

# Col 3 with REMAINING tenure
X3b = sm.add_constant(s[['exp','exp_sq','tenure_topel','remaining','rem_x_censor','rem_x_esq','rem_x_t'] + ctrl])
m3b = sm.OLS(s['lrw'], X3b).fit()
print("\nCol 3 with remaining tenure interactions:")
for v in ['exp','exp_sq','tenure_topel','remaining','rem_x_censor','rem_x_esq','rem_x_t']:
    print(f"  {v}: coef={m3b.params[v]:.6f}, se={m3b.bse[v]:.6f}")
print(f"R2: {m3b.rsquared:.4f}")

# What about interaction with JUST censor dummy (not ct*censor)?
df['just_censor'] = df['censor']
X3c = sm.add_constant(s[['exp','exp_sq','tenure_topel','ct','just_censor','ct_x_esq','ct_x_t'] + ctrl])
m3c = sm.OLS(s['lrw'], X3c).fit()
print("\nCol 3 with censor dummy (not interaction):")
for v in ['exp','exp_sq','tenure_topel','ct','just_censor','ct_x_esq','ct_x_t']:
    print(f"  {v}: coef={m3c.params[v]:.6f}, se={m3c.bse[v]:.6f}")

# Check: what if "x censor" means just censor dummy, not ct*censor?
# The paper says "x censor" with coefficient -0.0025 and SE 0.0073
# If it's just a censor dummy, SE of 0.0073 is reasonable for a binary variable
print("\nCensor dummy stats: mean={:.3f}, std={:.3f}".format(s['censor'].mean(), s['censor'].std()))
print("ct_x_censor stats: mean={:.3f}, std={:.3f}".format(s['ct_x_censor'].mean(), s['ct_x_censor'].std()))
