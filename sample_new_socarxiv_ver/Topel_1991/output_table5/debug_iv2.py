"""
Debug: Correct implementation of Topel two-step estimator.

Key insight from re-reading the paper:
Equation (10): y - x'Gamma_hat = X_0 * beta_1 + F*gamma + e

Where x'Gamma_hat contains ONLY the higher-order terms (T^2, T^3, T^4, X^2, X^3, X^4)
NOT the linear tenure term. The linear part (beta_1+beta_2)*T stays in y.

So:
w* = log_real_wage - [g2*T^2/100 + g3*T^3/1000 + g4*T^4/10000
                     + d2*X^2/100 + d3*X^3/1000 + d4*X^4/10000]

Then OLS of w* on X_0 + controls gives beta_1.
"""
import pandas as pd, numpy as np, statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/psid_panel.csv')

# Education recoding
cat_to_years = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17}
df['education_years'] = df['education_clean'].copy()
for mask_cond in [df['year'].between(1968,1975), df['year']>=1976]:
    if mask_cond.any() and df.loc[mask_cond,'education_clean'].max() <= 8:
        df.loc[mask_cond,'education_years'] = df.loc[mask_cond,'education_clean'].map(cat_to_years)
df['education_years'] = df['education_years'].fillna(12)
df['experience'] = df['age'] - df['education_years'] - 6
df.loc[df['experience']<0,'experience'] = 0

# Wage deflation
cps={1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,
     1974:1.167,1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,
     1980:1.128,1981:1.109,1982:1.103,1983:1.089}
gnp={1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,
     1973:44.4,1974:48.9,1975:53.6,1976:56.9,1977:60.6,1978:65.2,
     1979:72.6,1980:82.4,1981:90.9,1982:100.0}
df['cps_index']=df['year'].map(cps)
df['gnp_defl']=(df['year']-1).map(gnp)
df['log_real_wage']=df['log_hourly_wage']-np.log(df['gnp_defl']/100)-np.log(df['cps_index'])
df['tenure']=df['tenure_topel']

# Occupation
occ=df['occ_1digit'].copy(); m=occ>9; three=occ[m]
mapped=pd.Series(0,index=three.index,dtype=int)
mapped[(three>=1)&(three<=195)]=1;mapped[(three>=201)&(three<=245)]=2
mapped[(three>=260)&(three<=285)]=4;mapped[(three>=301)&(three<=395)]=4
mapped[(three>=401)&(three<=580)]=5;mapped[(three>=601)&(three<=695)]=6
mapped[(three>=701)&(three<=785)]=7;mapped[(three>=801)&(three<=824)]=9
mapped[(three>=900)&(three<=965)]=8
occ[m]=mapped; df['occ_broad']=occ

# Filters
filt=df[(df['tenure']>=1)&(~df['occ_broad'].isin([0,3,9]))].copy()
filt=filt[(filt['self_employed']==0)|(filt['self_employed'].isna())]
filt=filt[(filt['agriculture']==0)|(filt['agriculture'].isna())]
filt=filt[(filt['govt_worker']==0)|(filt['govt_worker'].isna())]
filt['initial_experience']=filt['experience']-filt['tenure']
filt.loc[filt['initial_experience']<0,'initial_experience']=0
for c in ['married','disabled','lives_in_smsa','region_ne','region_nc','region_south']:
    filt[c]=filt[c].fillna(0)

# Full sample
full = filt.copy().sort_values(['person_id','job_id','year'])

# ============================================================
# Step 1 on full sample
# ============================================================
full['prev_lrw']=full.groupby(['person_id','job_id'])['log_real_wage'].shift(1)
full['d_lrw']=full['log_real_wage']-full['prev_lrw']
full['prev_t']=full.groupby(['person_id','job_id'])['tenure'].shift(1)
full['d_t']=full['tenure']-full['prev_t']
wj=full.dropna(subset=['d_lrw','prev_t'])
wj=wj[wj['d_t']==1].copy()

T=wj['tenure'].values;Tp=T-1;X=wj['experience'].values;Xp=X-1
wj['d_t2']=(T**2-Tp**2)/100;wj['d_t3']=(T**3-Tp**3)/1000;wj['d_t4']=(T**4-Tp**4)/10000
wj['d_x2']=(X**2-Xp**2)/100;wj['d_x3']=(X**3-Xp**3)/1000;wj['d_x4']=(X**4-Xp**4)/10000

sv=['d_t2','d_t3','d_t4','d_x2','d_x3','d_x4']
for y in range(1969,1984):
    c=f'year_{y}';dc=f'd_{c}'
    wj[dc]=wj[c].values-wj.groupby(['person_id','job_id'])[c].shift(1).values
    wj[dc]=wj[dc].fillna(0)
    if wj[dc].std()>0: sv.append(dc)

X1=sm.add_constant(wj[sv]);y1=wj['d_lrw']
v=X1.notna().all(axis=1)&y1.notna()
m1=sm.OLS(y1[v],X1[v]).fit()
bh=m1.params['const'];bh_se=m1.bse['const']
g2=m1.params.get('d_t2',0);g3=m1.params.get('d_t3',0);g4=m1.params.get('d_t4',0)
d2=m1.params.get('d_x2',0);d3=m1.params.get('d_x3',0);d4=m1.params.get('d_x4',0)
print(f"FULL SAMPLE Step 1:")
print(f"  beta_hat (b1+b2)={bh:.4f} ({bh_se:.4f})")
print(f"  g2={g2:.4f} g3={g3:.4f} g4={g4:.4f}")
print(f"  d2={d2:.4f} d3={d3:.4f} d4={d4:.4f}")
print(f"  N_wj={v.sum()}")

# ============================================================
# Step 2: CORRECT - subtract only higher-order terms, NOT linear beta_hat*T
# ============================================================
# w* = log_real_wage - [higher-order terms only]
full['w_star'] = (full['log_real_wage']
                  - g2*full['tenure']**2/100 - g3*full['tenure']**3/1000 - g4*full['tenure']**4/10000
                  - d2*full['experience']**2/100 - d3*full['experience']**3/1000 - d4*full['experience']**4/10000)

# OLS of w* on X_0 + controls
ctrls=['education_years','married','disabled','lives_in_smsa','region_ne','region_nc','region_south']
ctrls_use = [c for c in ctrls if full[c].std()>0]
yr_use=[f'year_{y}' for y in range(1969,1984) if full[f'year_{y}'].std()>0]

# Check rank
all_vars = ['initial_experience']+ctrls_use+yr_use
test_X = sm.add_constant(full[all_vars].astype(float))
r = np.linalg.matrix_rank(test_X.values)
print(f"\nStep 2 full sample: N={len(full)}")
print(f"  Rank: {r}, Cols: {test_X.shape[1]}")

# If rank deficient, trim year dummies
while r < test_X.shape[1] and len(yr_use) > 0:
    yr_use = yr_use[:-1]
    all_vars = ['initial_experience']+ctrls_use+yr_use
    test_X = sm.add_constant(full[all_vars].astype(float))
    r = np.linalg.matrix_rank(test_X.values)

m_ols = sm.OLS(full['w_star'].astype(float), test_X).fit()
b1_full = m_ols.params['initial_experience']
b1_se_full = m_ols.bse['initial_experience']
print(f"  beta_1 (OLS on X_0): {b1_full:.4f} ({b1_se_full:.4f})")
print(f"  beta_2 = {bh:.4f} - {b1_full:.4f} = {bh-b1_full:.4f}")
print(f"  Paper Table 3: beta_1=0.0713, beta_2=0.0545")

# ============================================================
# Now try on Professional/Service subsample
# ============================================================
ps = filt[filt['occ_broad'].isin([1,2,4,8])].copy().sort_values(['person_id','job_id','year'])

# Step 1 on subsample
ps['prev_lrw']=ps.groupby(['person_id','job_id'])['log_real_wage'].shift(1)
ps['d_lrw']=ps['log_real_wage']-ps['prev_lrw']
ps['prev_t']=ps.groupby(['person_id','job_id'])['tenure'].shift(1)
ps['d_t']=ps['tenure']-ps['prev_t']
wj_ps=ps.dropna(subset=['d_lrw','prev_t'])
wj_ps=wj_ps[wj_ps['d_t']==1].copy()

T=wj_ps['tenure'].values;Tp=T-1;X=wj_ps['experience'].values;Xp=X-1
wj_ps['d_t2']=(T**2-Tp**2)/100;wj_ps['d_t3']=(T**3-Tp**3)/1000;wj_ps['d_t4']=(T**4-Tp**4)/10000
wj_ps['d_x2']=(X**2-Xp**2)/100;wj_ps['d_x3']=(X**3-Xp**3)/1000;wj_ps['d_x4']=(X**4-Xp**4)/10000
sv_ps=['d_t2','d_t3','d_t4','d_x2','d_x3','d_x4']
for y in range(1969,1984):
    c=f'year_{y}';dc=f'd_{c}'
    wj_ps[dc]=wj_ps[c].values-wj_ps.groupby(['person_id','job_id'])[c].shift(1).values
    wj_ps[dc]=wj_ps[dc].fillna(0)
    if wj_ps[dc].std()>0: sv_ps.append(dc)
X1_ps=sm.add_constant(wj_ps[sv_ps]);y1_ps=wj_ps['d_lrw']
v_ps=X1_ps.notna().all(axis=1)&y1_ps.notna()
m1_ps=sm.OLS(y1_ps[v_ps],X1_ps[v_ps]).fit()
bh_ps=m1_ps.params['const'];bh_ps_se=m1_ps.bse['const']
g2_ps=m1_ps.params.get('d_t2',0);g3_ps=m1_ps.params.get('d_t3',0);g4_ps=m1_ps.params.get('d_t4',0)
d2_ps=m1_ps.params.get('d_x2',0);d3_ps=m1_ps.params.get('d_x3',0);d4_ps=m1_ps.params.get('d_x4',0)

print(f"\nPS SUBSAMPLE Step 1:")
print(f"  beta_hat={bh_ps:.4f} ({bh_ps_se:.4f})")
print(f"  g2={g2_ps:.4f} g3={g3_ps:.4f} g4={g4_ps:.4f}")

# Step 2 on PS - CORRECT (subtract only higher-order terms)
ps['w_star'] = (ps['log_real_wage']
                - g2_ps*ps['tenure']**2/100 - g3_ps*ps['tenure']**3/1000 - g4_ps*ps['tenure']**4/10000
                - d2_ps*ps['experience']**2/100 - d3_ps*ps['experience']**3/1000 - d4_ps*ps['experience']**4/10000)

ps2 = ps.dropna(subset=['w_star','experience','initial_experience']).copy()
ctrls_ps = [c for c in ctrls if ps2[c].std()>0]

# Add union control for PS
ps2['union_ctrl'] = ps2.get('union_member', pd.Series(0, index=ps2.index)).fillna(0)
ctrls_ps_all = ctrls_ps + ['union_ctrl'] if ps2['union_ctrl'].std() > 0 else ctrls_ps

yr_ps = [f'year_{y}' for y in range(1969,1984) if ps2[f'year_{y}'].std()>0]
all_vars_ps = ['initial_experience']+ctrls_ps_all+yr_ps
test_ps = sm.add_constant(ps2[all_vars_ps].astype(float))
r_ps = np.linalg.matrix_rank(test_ps.values)
while r_ps < test_ps.shape[1] and len(yr_ps) > 0:
    yr_ps = yr_ps[:-1]
    all_vars_ps = ['initial_experience']+ctrls_ps_all+yr_ps
    test_ps = sm.add_constant(ps2[all_vars_ps].astype(float))
    r_ps = np.linalg.matrix_rank(test_ps.values)

m_ps = sm.OLS(ps2['w_star'].astype(float), test_ps).fit()
b1_ps = m_ps.params['initial_experience']
b1_se_ps = m_ps.bse['initial_experience']
b2_ps = bh_ps - b1_ps
print(f"\n  Step 2 N: {len(ps2)}")
print(f"  beta_1: {b1_ps:.4f} ({b1_se_ps:.4f})")
print(f"  beta_2: {b2_ps:.4f}")
print(f"  Paper Table 5 col 1: beta_1=0.0707 (0.0288), beta_2=0.0601 (0.0127)")

# Cumulative returns
for T_val in [5, 10, 15, 20]:
    cum = (b2_ps * T_val
           + g2_ps * T_val**2/100 + g3_ps * T_val**3/1000 + g4_ps * T_val**4/10000)
    print(f"  Cum return {T_val}yr: {cum:.4f}")
