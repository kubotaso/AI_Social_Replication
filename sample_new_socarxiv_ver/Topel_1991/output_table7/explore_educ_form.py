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

# Try different forms of education control:
# A: Continuous years (current)
# B: Education categories as dummies
# C: Education squared
# D: Log education
# E: No education

s = df.dropna(subset=['lrw','exp','exp_sq','tenure_topel','ed_yr','married',
                       'union','dis','lives_in_smsa',
                       'region_ne','region_nc','region_south'] + yr_cols).copy()

base = ['exp','exp_sq','tenure_topel']
ctrl_base = ['married','union','dis','lives_in_smsa','region_ne','region_nc','region_south'] + yr_cols

# A: Continuous education
X_a = sm.add_constant(s[base + ['ed_yr'] + ctrl_base])
m_a = sm.OLS(s['lrw'], X_a).fit()
print(f"A(continuous ed): R2={m_a.rsquared:.4f}, exp_sq={m_a.params['exp_sq']:.7f}, exp={m_a.params['exp']:.5f}, ten={m_a.params['tenure_topel']:.5f}")

# B: Education dummies (based on categories)
ed_dummies = pd.get_dummies(s['education_clean'].astype(int), prefix='ed_cat', drop_first=True, dtype=float)
s_b = pd.concat([s, ed_dummies], axis=1)
ed_dum_cols = list(ed_dummies.columns)
X_b = sm.add_constant(s_b[base + ed_dum_cols + ctrl_base])
m_b = sm.OLS(s_b['lrw'], X_b).fit()
print(f"B(ed dummies):   R2={m_b.rsquared:.4f}, exp_sq={m_b.params['exp_sq']:.7f}, exp={m_b.params['exp']:.5f}, ten={m_b.params['tenure_topel']:.5f}")

# C: Education + Education^2
s['ed_sq'] = s['ed_yr'] ** 2
X_c = sm.add_constant(s[base + ['ed_yr','ed_sq'] + ctrl_base])
m_c = sm.OLS(s['lrw'], X_c).fit()
print(f"C(ed+ed^2):      R2={m_c.rsquared:.4f}, exp_sq={m_c.params['exp_sq']:.7f}, exp={m_c.params['exp']:.5f}, ten={m_c.params['tenure_topel']:.5f}")

# D: Log education
s['log_ed'] = np.log(s['ed_yr'].clip(lower=1))
X_d = sm.add_constant(s[base + ['log_ed'] + ctrl_base])
m_d = sm.OLS(s['lrw'], X_d).fit()
print(f"D(log ed):       R2={m_d.rsquared:.4f}, exp_sq={m_d.params['exp_sq']:.7f}, exp={m_d.params['exp']:.5f}, ten={m_d.params['tenure_topel']:.5f}")

# E: No education
X_e = sm.add_constant(s[base + ctrl_base])
m_e = sm.OLS(s['lrw'], X_e).fit()
print(f"E(no ed):        R2={m_e.rsquared:.4f}, exp_sq={m_e.params['exp_sq']:.7f}, exp={m_e.params['exp']:.5f}, ten={m_e.params['tenure_topel']:.5f}")

# F: Education as single category variable (raw 0-8)
X_f = sm.add_constant(s[base + ['education_clean'] + ctrl_base])
m_f = sm.OLS(s['lrw'], X_f).fit()
print(f"F(raw ed cat):   R2={m_f.rsquared:.4f}, exp_sq={m_f.params['exp_sq']:.7f}, exp={m_f.params['exp']:.5f}, ten={m_f.params['tenure_topel']:.5f}")

# Target: exp_sq = -0.00079, exp = 0.0418, ten = 0.0138, R2 = 0.422
print(f"\nTargets: exp_sq=-0.00079, exp=0.0418, ten=0.0138, R2=0.422")
