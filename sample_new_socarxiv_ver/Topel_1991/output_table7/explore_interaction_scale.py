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

# Education dummies
df['ed_cat'] = pd.cut(df['ed_yr'], bins=[-1,11,12,15,20], labels=['lt12','12','13_15','16p'])
ed_dummies = pd.get_dummies(df['ed_cat'], prefix='ed', drop_first=True, dtype=float)
for col in ed_dummies.columns:
    df[col] = ed_dummies[col]

yr_cols = [c for c in df.columns if c.startswith('year_') and c != 'year_1971' and df[c].sum() > 0]
ctrl = list(ed_dummies.columns) + ['married','union','dis','lives_in_smsa',
        'region_ne','region_nc','region_south'] + yr_cols

df['ct'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['ct_x_censor'] = df['ct'] * df['censor']

# Original interactions
df['ct_x_esq'] = df['ct'] * df['exp_sq']
df['ct_x_t'] = df['ct'] * df['tenure_topel']

# Scaled interactions: ct * (exp_sq/100)
df['ct_x_esq_100'] = df['ct'] * df['exp_sq'] / 100
df['ct_x_t_10'] = df['ct'] * df['tenure_topel'] / 10

s = df.dropna(subset=ctrl + ['lrw','exp','exp_sq','tenure_topel','ct']).copy()

# Original
X = sm.add_constant(s[['exp','exp_sq','tenure_topel','ct','ct_x_censor','ct_x_esq','ct_x_t'] + ctrl])
m = sm.OLS(s['lrw'], X).fit()
print("Original:")
print(f"  ct_x_esq coef: {m.params['ct_x_esq']:.8f}, se: {m.bse['ct_x_esq']:.8f}")
print(f"  ct_x_t coef: {m.params['ct_x_t']:.8f}, se: {m.bse['ct_x_t']:.8f}")
print(f"  exp_sq coef: {m.params['exp_sq']:.8f}")

# If paper reports -0.00061 for "Experience^2 (interaction)" and we get -0.0000014
# The ratio is 0.00061 / 0.0000014 = 435
# That's close to mean(exp_sq) = 500
# Maybe the paper's interaction is ct * (exp_sq - mean_exp_sq)?
# Or maybe the paper reports the MARGINAL EFFECT at some evaluation point?

# Actually: let me check if the paper might be reporting not the raw coefficient
# but the coefficient multiplied by something.
# In Table 2, the paper reports "Delta Experience^2 (x10^2)" meaning the coefficient
# is for the variable divided by 100.
# Maybe Table 7 also uses this convention for the interaction?

# If Table 7's "Experience^2 (interaction)" row uses the same scaling as Table 2,
# then the reported coefficient -0.00061 is actually for ct * (exp_sq / 100).
# My raw ct_x_esq coefficient is -0.0000014
# If I multiply by 100: -0.00014
# That's still not -0.00061

# But wait, in Table 7, the main Experience^2 coefficient is -0.00072 (raw, not scaled).
# In Table 2, Experience^2 scaled by 10^2 gives coefficients like -0.4592.
# So Table 7 uses UNscaled experience^2.
# Therefore the interaction coefficient should also be unscaled.

# Let me check correlation structure
print(f"\nCorr(ct_x_esq, exp_sq): {s['ct_x_esq'].corr(s['exp_sq']):.4f}")
print(f"Corr(ct_x_esq, ct): {s['ct_x_esq'].corr(s['ct']):.4f}")
print(f"Corr(ct_x_t, tenure): {s['ct_x_t'].corr(s['tenure_topel']):.4f}")
print(f"Corr(ct_x_t, ct): {s['ct_x_t'].corr(s['ct']):.4f}")
print(f"Corr(ct, tenure): {s['ct'].corr(s['tenure_topel']):.4f}")
print(f"Corr(ct, exp_sq): {s['ct'].corr(s['exp_sq']):.4f}")

# VIF analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = s[['exp_sq','tenure_topel','ct','ct_x_esq','ct_x_t']].copy()
vif_data = sm.add_constant(vif_data)
vifs = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
print(f"\nVIFs: {dict(zip(vif_data.columns, vifs))}")
