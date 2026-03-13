"""Debug: Try using paper's b1+b2 values and IV2SLS for step 2."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

try:
    from linearmodels.iv import IV2SLS
    HAS_LM = True
except:
    HAS_LM = False

df = pd.read_csv('data/psid_panel.csv')

em = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17}
df['ed'] = df['education_clean'].map(em).fillna(12)
df['exp'] = (df['age'] - df['ed'] - 6).clip(lower=0)
df['ten'] = df['tenure_topel']

gnp = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,
       1973:44.4,1974:48.9,1975:53.6,1976:56.9,1977:60.6,1978:65.2,
       1979:72.6,1980:82.4,1981:90.9,1982:100.0}
cps = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,
       1974:1.167,1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,
       1980:1.128,1981:1.109,1982:1.103,1983:1.089}

df['lrw'] = df['log_hourly_wage'] - np.log((df['year']-1).map(gnp)/100.0) - np.log(df['year'].map(cps))

occ = df['occ_1digit'].copy()
m3 = occ > 9
three = occ[m3]
mapped = pd.Series(0, index=three.index, dtype=int)
mapped[(three>=1)&(three<=195)]=1
mapped[(three>=201)&(three<=245)]=2
mapped[(three>=260)&(three<=395)]=4
mapped[(three>=401)&(three<=580)]=5
mapped[(three>=601)&(three<=695)]=6
mapped[(three>=701)&(three<=785)]=7
mapped[(three>=801)&(three<=824)]=9
mapped[(three>=900)&(three<=965)]=8
occ[m3] = mapped
df['occ'] = occ

df = df.sort_values(['person_id','job_id','year'])
ju = df.groupby(['person_id','job_id'])['union_member'].agg(
    lambda x: (x.mean()>0.5) if x.notna().any() else np.nan
).reset_index().rename(columns={'union_member':'job_union'})
df = df.merge(ju, on=['person_id','job_id'], how='left')

df = df[df['ten']>=1].dropna(subset=['log_hourly_wage','lrw']).copy()
df = df[~df['occ'].isin([0,3,9])].copy()
df = df[(df['self_employed']==0)|(df['self_employed'].isna())].copy()
df['x0'] = (df['exp'] - df['ten']).clip(lower=0)
for c in ['married','disabled','lives_in_smsa','union_member','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

# Paper values
g2 = -0.004592; g3 = 0.0001846; g4 = -0.00000245
d2 = -0.006051; d3 = 0.0002067; d4 = -0.00000238

yr = [f'year_{y}' for y in range(1969, 1984)]
ct = ['ed','married','disabled','lives_in_smsa','region_ne','region_nc','region_south']

paper_vals = {
    'PS': {'b1b2': 0.1309, 'occ': [1,2,4,8], 'extra': ['union_member']},
    'BC_NU': {'b1b2': 0.1520, 'occ': [5,6,7], 'extra': []},
    'BC_U': {'b1b2': 0.0992, 'occ': [5,6,7], 'extra': []},
}

print("=== Testing different approaches ===\n")

for name, info in paper_vals.items():
    if name == 'BC_NU':
        mask = df['occ'].isin(info['occ']) & (df['job_union']==0)
    elif name == 'BC_U':
        mask = df['occ'].isin(info['occ']) & (df['job_union']==1)
    else:
        mask = df['occ'].isin(info['occ'])

    sub = df[mask].copy()
    b1b2 = info['b1b2']

    # Construct w*
    sub['ws'] = (sub['lrw'] - b1b2*sub['ten']
                 - g2*sub['ten']**2 - g3*sub['ten']**3 - g4*sub['ten']**4
                 - d2*sub['exp']**2 - d3*sub['exp']**3 - d4*sub['exp']**4)

    all_ctrl = ct + info['extra'] + yr
    ctrl_use = [c for c in all_ctrl if c in sub.columns and sub[c].std() > 0]

    # Approach 1: OLS on X_0
    X_ols = sm.add_constant(sub[['x0'] + ctrl_use].astype(float))
    m_ols = sm.OLS(sub['ws'], X_ols).fit()
    b1_ols = m_ols.params['x0']
    b1_ols_se = m_ols.bse['x0']

    # Approach 2: IV2SLS (experience instrumented by x0)
    if HAS_LM:
        try:
            exog = sm.add_constant(sub[ctrl_use].astype(float))
            dep = sub['ws']
            endog_var = sub[['exp']]
            instr = sub[['x0']]
            iv_m = IV2SLS(dep, exog, endog_var, instr).fit(cov_type='unadjusted')
            b1_iv = iv_m.params['exp']
            b1_iv_se = iv_m.std_errors['exp']
        except Exception as e:
            b1_iv = np.nan
            b1_iv_se = np.nan
            print(f"  {name} IV2SLS failed: {e}")
    else:
        b1_iv = np.nan
        b1_iv_se = np.nan

    # Approach 3: OLS on exp directly (biased but for comparison)
    X_exp = sm.add_constant(sub[['exp'] + ctrl_use].astype(float))
    m_exp = sm.OLS(sub['ws'], X_exp).fit()
    b1_exp = m_exp.params['exp']

    b2_ols = b1b2 - b1_ols
    b2_iv = b1b2 - b1_iv if not np.isnan(b1_iv) else np.nan

    print(f"{name}: N={len(sub)}, b1b2={b1b2:.4f}")
    print(f"  OLS on X_0: beta_1={b1_ols:.4f} (SE={b1_ols_se:.4f}), beta_2={b2_ols:.4f}")
    print(f"  IV2SLS:     beta_1={b1_iv:.4f} (SE={b1_iv_se:.4f}), beta_2={b2_iv:.4f}")
    print(f"  OLS on exp: beta_1={b1_exp:.4f}")

    # Cumulative returns
    for T in [5, 10, 15, 20]:
        cum_ols = b2_ols * T + g2 * T**2 + g3 * T**3 + g4 * T**4
        cum_iv = (b2_iv * T + g2 * T**2 + g3 * T**3 + g4 * T**4) if not np.isnan(b2_iv) else np.nan
        print(f"  cum{T}: OLS={cum_ols:.4f}, IV={cum_iv:.4f}")
    print()

print("\nPaper targets:")
print("PS:    beta_1=0.0707, beta_2=0.0601, cum5=0.1887, cum10=0.2400, cum15=0.2527, cum20=0.2841")
print("BC_NU: beta_1=0.1066, beta_2=0.0513, cum5=0.1577, cum10=0.2073, cum15=0.2480, cum20=0.3295")
print("BC_U:  beta_1=0.0592, beta_2=0.0399, cum5=0.1401, cum10=0.2033, cum15=0.2384, cum20=0.2733")
