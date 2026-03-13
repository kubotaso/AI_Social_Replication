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

df['ct'] = df.groupby('job_id')['tenure_topel'].transform('max').astype(float)
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
df['ct_x_censor'] = df['ct'] * df['censor']

s = df.dropna(subset=ctrl + ['lrw','exp','exp_sq','tenure_topel','ct']).copy()

# INSIGHT: maybe the problem is that ct is highly correlated with tenure
# For a job of total duration T_bar, current tenure goes from 1 to T_bar
# So ct*tenure is essentially tenure^2 (for jobs of fixed length)
# This creates multicollinearity with exp_sq and creates sign instability

# Let me try the full unrestricted model but look at PARTIAL effects
print("Correlation between ct and tenure:", s['ct'].corr(s['tenure_topel']))
print("Correlation between ct*t and t^2:", (s['ct']*s['tenure_topel']).corr(s['tenure_topel']**2))
print()

# Maybe the paper defines the interaction NOT as ct * tenure but as
# (ct - tenure) * tenure? That is, REMAINING tenure * current tenure
s['remaining'] = s['ct'] - s['tenure_topel']
s['remaining'] = s['remaining'].clip(lower=0)

# Or maybe it's log(ct) * tenure?
s['log_ct'] = np.log(s['ct'].clip(lower=1))

# Or maybe it's 1/ct * tenure?
s['inv_ct'] = 1.0 / s['ct']

# Let me try "remaining tenure" approach more carefully
# "Observed completed tenure" = T_bar (total job duration)
# The interaction "Tenure (interaction)" = T_bar * T (current tenure)
#
# But what if the paper really means:
# Completed_tenure (as reported) = total job duration T_bar
# BUT the interactions use a FRACTION: T/T_bar (fraction of job completed)?

s['frac_done'] = s['tenure_topel'] / s['ct']  # fraction of job completed
s['frac_remain'] = 1 - s['frac_done']  # fraction remaining

# Or: what if "Tenure (interaction)" coefficient in the paper refers to
# the coefficient on T when evaluated at different T_bar values?
# The total tenure effect is: beta_t * T + gamma_t * T_bar * T
# At T_bar = 5: d_wage/d_T = beta_t + 5*gamma_t
# If beta_t = 0.0137 and gamma_t = 0.0142, then at T_bar = 5:
# d_wage/d_T = 0.0137 + 5 * 0.0142 = 0.0847
# That's an 8.5% annual return to tenure for a 5-year job!
# That seems unreasonably high...

# Unless gamma_t operates on a different scale. What if T_bar is in some
# other unit, like T_bar / 10?

# Let me try T_bar/10 interactions
s['ct_10'] = s['ct'] / 10
s['ct_10_x_esq'] = s['ct_10'] * s['exp_sq']
s['ct_10_x_t'] = s['ct_10'] * s['tenure_topel']

X = sm.add_constant(s[['exp','exp_sq','tenure_topel','ct','ct_x_censor','ct_10_x_esq','ct_10_x_t'] + ctrl])
m = sm.OLS(s['lrw'], X).fit()
print("With ct/10 interactions:")
for v in ['exp','exp_sq','tenure_topel','ct','ct_x_censor','ct_10_x_esq','ct_10_x_t']:
    print(f"  {v}: coef={m.params[v]:.6f}, se={m.bse[v]:.6f}")

# Of course scaling just scales the coefficient by 10...
# ct_10_x_t coef should be ct_x_t * 10 = -0.0271
# That's even more negative.

# KEY EXPERIMENT: What if the label is misleading and "Tenure (interaction)"
# is actually T * (1/T_bar) or T * (T_bar - T)?
# Let me try: interaction = tenure * remaining_time
s['t_x_remain'] = s['tenure_topel'] * s['remaining']
s['esq_x_remain'] = s['exp_sq'] * s['remaining']

X = sm.add_constant(s[['exp','exp_sq','tenure_topel','ct','ct_x_censor','esq_x_remain','t_x_remain'] + ctrl])
m = sm.OLS(s['lrw'], X).fit()
print("\nWith tenure*remaining interactions:")
for v in ['exp','exp_sq','tenure_topel','ct','ct_x_censor','esq_x_remain','t_x_remain']:
    print(f"  {v}: coef={m.params[v]:.6f}, se={m.bse[v]:.6f}")

# Also try: interaction = 1/(ct) * exp_sq and 1/(ct) * tenure
s['inv_ct_x_esq'] = s['inv_ct'] * s['exp_sq']
s['inv_ct_x_t'] = s['inv_ct'] * s['tenure_topel']

X = sm.add_constant(s[['exp','exp_sq','tenure_topel','ct','ct_x_censor','inv_ct_x_esq','inv_ct_x_t'] + ctrl])
m = sm.OLS(s['lrw'], X).fit()
print("\nWith 1/ct interactions:")
for v in ['exp','exp_sq','tenure_topel','ct','ct_x_censor','inv_ct_x_esq','inv_ct_x_t']:
    print(f"  {v}: coef={m.params[v]:.6f}, se={m.bse[v]:.6f}")

print("\nPaper targets (col 3):")
print("exp=.0345, exp_sq=-.00072, ten=.0137, ct=.0316, ct_x_esq=-.00061, ct_x_t=+.0142")
