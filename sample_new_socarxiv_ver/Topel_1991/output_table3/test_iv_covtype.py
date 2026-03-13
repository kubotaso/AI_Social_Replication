"""Test different cov_type in step 2 IV regression"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
PAPER_S1 = {
    'b1b2': 0.1258, 'g2': -0.004592, 'g3': 0.0001846, 'g4': -0.00000245,
    'd2': -0.005871, 'd3': 0.0002067, 'd4': -0.00000238
}

df = pd.read_csv('data/psid_panel.csv')
df['pnum'] = df['person_id'] % 1000
df = df[df['pnum'].isin([1, 170, 3])].copy()
df['education_years'] = np.nan
for yr in df['year'].unique():
    mask = df['year'] == yr
    if yr in [1975, 1976]:
        df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
    else:
        df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)
df = df.dropna(subset=['education_years']).copy()
df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
df['tenure'] = df['tenure_topel'].copy()
df['gnp_def'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, np.nan))
df['log_real_gnp'] = df['log_hourly_wage'] - np.log(df['gnp_def'] / 100.0)
df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
df = df[df['hourly_wage'] > 0].copy()
df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
df = df[df['tenure'] >= 1].copy()
df = df.dropna(subset=['log_real_gnp', 'experience', 'tenure']).copy()
df['union_member'] = df['union_member'].fillna(0)
df['disabled'] = df['disabled'].fillna(0)
df['married'] = df['married'].fillna(0)
for k in [2, 3, 4]:
    df[f'tenure_{k}'] = df['tenure'] ** k
    df[f'exp_{k}'] = df['experience'] ** k
for yr in range(1970, 1984):
    col = f'year_{yr}'
    if col not in df.columns:
        df[col] = (df['year'] == yr).astype(int)

ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]
wage_var = 'log_real_gnp'

# w* construction
b1b2 = PAPER_S1['b1b2']
g2, g3, g4 = PAPER_S1['g2'], PAPER_S1['g3'], PAPER_S1['g4']
d2, d3, d4 = PAPER_S1['d2'], PAPER_S1['d3'], PAPER_S1['d4']
levels = df.copy()
levels['w_star'] = (levels[wage_var]
                    - b1b2 * levels['tenure']
                    - g2 * levels['tenure_2']
                    - g3 * levels['tenure_3']
                    - g4 * levels['tenure_4']
                    - d2 * levels['exp_2']
                    - d3 * levels['exp_3']
                    - d4 * levels['exp_4'])
levels['init_exp'] = (levels['experience'] - levels['tenure']).clip(lower=0)
all_ctrl = ctrl + yr_dums
levels = levels.dropna(subset=['w_star', 'experience', 'init_exp'] + all_ctrl).copy()
exog_df = sm.add_constant(levels[all_ctrl])
rank = np.linalg.matrix_rank(exog_df.values)
active_yr = yr_dums.copy()
while rank < exog_df.shape[1] and active_yr:
    active_yr.pop()
    all_ctrl = ctrl + active_yr
    exog_df = sm.add_constant(levels[all_ctrl])
    rank = np.linalg.matrix_rank(exog_df.values)

dep = levels['w_star']
exog = sm.add_constant(levels[all_ctrl])
endog = levels[['experience']]
instruments = levels[['init_exp']]

for cov_type in ['unadjusted', 'robust', 'kernel']:
    try:
        if cov_type == 'kernel':
            iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type=cov_type,
                                                            kernel='bartlett',
                                                            bandwidth=5)
        else:
            iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type=cov_type)
        b1 = iv.params['experience']
        se = iv.std_errors['experience']
        print(f"cov_type={cov_type}: beta_1={b1:.4f}, SE_naive={se:.6f}")
    except Exception as e:
        print(f"cov_type={cov_type}: ERROR: {e}")

# Also try clustered by person
try:
    iv_clus = IV2SLS(dep, exog, endog, instruments).fit(
        cov_type='clustered', clusters=levels['person_id'])
    print(f"cov_type=clustered(person): beta_1={iv_clus.params['experience']:.4f}, SE={iv_clus.std_errors['experience']:.6f}")
except Exception as e:
    print(f"cov_type=clustered(person): ERROR: {e}")
