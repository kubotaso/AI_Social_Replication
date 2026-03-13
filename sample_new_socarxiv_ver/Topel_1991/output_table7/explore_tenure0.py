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

# Tenure starting at 0
df['ten0'] = df['tenure_topel'] - 1

# Different completed tenure definitions:
# A) max(tenure_topel) = max years (starts at 1)
df['ct_A'] = df.groupby('job_id')['tenure_topel'].transform('max')
# B) max(ten0) = max years starting at 0
df['ct_B'] = df.groupby('job_id')['ten0'].transform('max')
# C) n_observations in job = number of years observed
df['ct_C'] = df.groupby('job_id')['year'].transform('count').astype(float)
# D) max_year - min_year + 1
df['ct_D'] = (df.groupby('job_id')['year'].transform('max') - df.groupby('job_id')['year'].transform('min') + 1).astype(float)

df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)

s = df.dropna(subset=ctrl + ['lrw','exp','exp_sq','ten0','tenure_topel']).copy()

# Test all combinations
for ct_name in ['ct_A', 'ct_B', 'ct_C', 'ct_D']:
    for ten_name, ten_label in [('tenure_topel', 'ten1'), ('ten0', 'ten0')]:
        ct = s[ct_name]
        ten = s[ten_name]

        s['_ct'] = ct
        s['_ct_x_censor'] = ct * s['censor']
        s['_ct_x_esq'] = ct * s['exp_sq']
        s['_ct_x_t'] = ct * ten

        X = sm.add_constant(s[['exp','exp_sq',ten_name,'_ct','_ct_x_censor','_ct_x_esq','_ct_x_t'] + ctrl])
        m = sm.OLS(s['lrw'], X).fit()
        print(f"{ct_name}+{ten_label}: R2={m.rsquared:.4f}, "
              f"exp={m.params['exp']:.5f}, "
              f"exp_sq={m.params['exp_sq']:.7f}, "
              f"ten={m.params[ten_name]:.5f}, "
              f"ct={m.params['_ct']:.5f}, "
              f"ct_x_esq={m.params['_ct_x_esq']:.7f}, "
              f"ct_x_t={m.params['_ct_x_t']:.6f}")

# Paper targets (col 3):
# exp=.0345, exp_sq=-.00072, ten=.0137, ct=.0316
# ct_x_esq=-.00061, ct_x_t=+.0142
print("\nPaper targets (col 3):")
print("exp=.0345, exp_sq=-.00072, ten=.0137, ct=.0316, ct_x_esq=-.00061, ct_x_t=+.0142")
