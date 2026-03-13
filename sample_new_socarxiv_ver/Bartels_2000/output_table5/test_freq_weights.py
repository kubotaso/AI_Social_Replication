"""Test freq_weights effect on Probit LL."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

# Simple test: create data and compare expanded vs freq_weights
np.random.seed(42)
n = 100
x = np.random.randn(n)
y = (x + np.random.randn(n) > 0).astype(float)
X = sm.add_constant(x)

# Unweighted
mod1 = Probit(y, X).fit(disp=0)
print(f'Unweighted: LL={mod1.llf:.4f}, nobs={mod1.nobs}')

# With freq_weights = 2 for all
wt = np.full(n, 2.0)
mod2 = Probit(y, X).fit(disp=0, freq_weights=wt)
print(f'freq_weights=2: LL={mod2.llf:.4f}, nobs={mod2.nobs}')

# Expanded (double the data)
y2 = np.concatenate([y, y])
X2 = np.vstack([X, X])
mod3 = Probit(y2, X2).fit(disp=0)
print(f'Expanded 2x: LL={mod3.llf:.4f}, nobs={mod3.nobs}')

# So freq_weights multiplies the LL contribution of each obs by weight
# LL_weighted = sum(w_i * log(p_i)) vs LL_expanded = sum(log(p_i)) for expanded
# These should be the same since both are just summing log(p_i) twice

print(f'\nCoefficients:')
print(f'Unweighted: {mod1.params}')
print(f'freq_wt=2:  {mod2.params}')
print(f'Expanded:   {mod3.params}')

# Now test with 1960 actual data
cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf58 = cdf[cdf['VCF0004']==1958].copy()
panel60 = cdf60[cdf60['VCF0006a'] < 19600000].copy()
merged = panel60.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
merged['house_vote'] = merged['VCF0707']
mask = merged['house_vote'].isna() & merged['VCF0706'].isin([1.0, 2.0])
merged.loc[mask, 'house_vote'] = merged.loc[mask, 'VCF0706']
valid = merged[
    merged['house_vote'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
valid = construct_pid_vars(valid, 'VCF0301', 'curr')

X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
wt = valid['VCF0009x'].fillna(1.0).values

# Unweighted
mod_uw = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
print(f'\n1960 unweighted: N={len(valid)}, LL={mod_uw.llf:.1f}, R2={mod_uw.prsquared:.4f}')

# freq_weights
mod_fw = Probit(valid['house_rep'].astype(float), X).fit(disp=0, freq_weights=wt)
print(f'1960 freq_weights: N_eff={wt.sum():.0f}, LL={mod_fw.llf:.1f}, R2={mod_fw.prsquared:.4f}')
print(f'  nobs={mod_fw.nobs}')
print(f'  Strong={mod_fw.params["strong_curr"]:.3f}({mod_fw.bse["strong_curr"]:.3f})')

# Expanded
wt_int = wt.astype(int)
expanded = valid.loc[valid.index.repeat(wt_int)].reset_index(drop=True)
X_exp = sm.add_constant(expanded[['strong_curr','weak_curr','lean_curr']])
mod_exp = Probit(expanded['house_rep'].astype(float), X_exp).fit(disp=0)
print(f'1960 expanded: N={len(expanded)}, LL={mod_exp.llf:.1f}, R2={mod_exp.prsquared:.4f}')
print(f'  Strong={mod_exp.params["strong_curr"]:.3f}({mod_exp.bse["strong_curr"]:.3f})')

print(f'\nTarget: N=911, LL=-372.7, R2=0.41')
