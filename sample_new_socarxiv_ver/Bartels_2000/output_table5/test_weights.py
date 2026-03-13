"""Test whether sample weights improve results."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

# Check weight variables
print('Weight variables:')
for col in ['VCF0009x', 'VCF0009y', 'VCF0009z']:
    if col in cdf.columns:
        # Check for 1992 panel
        cdf92 = cdf[cdf['VCF0004']==1992]
        print(f'{col} (1992): {cdf92[col].describe()}')
        print(f'  unique values: {cdf92[col].nunique()}')
        print(f'  NaN: {cdf92[col].isna().sum()}')

# Check for 1960
print('\n1960 weights:')
cdf60 = cdf[cdf['VCF0004']==1960]
for col in ['VCF0009x', 'VCF0009y', 'VCF0009z']:
    if col in cdf.columns:
        print(f'{col}: {cdf60[col].describe()}')

# Try weighted probit for 1992
def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

cdf92_panel = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92_panel[cdf92_panel['VCF0006a'] < 19920000]
cdf90 = cdf[cdf['VCF0004']==1990].copy()
merged = cdf92_panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
valid = merged[
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
valid['house_rep'] = (valid['VCF0707'] == 2.0).astype(int)
valid = construct_pid_vars(valid, 'VCF0301', 'curr')
valid = construct_pid_vars(valid, 'VCF0301_lag', 'lag')

print(f'\n1992 panel valid: {len(valid)}')

for wt_col in ['VCF0009x', 'VCF0009y', 'VCF0009z']:
    if wt_col in valid.columns and valid[wt_col].notna().any():
        wt = valid[wt_col].fillna(1.0)
        if wt.std() > 0.01:  # Non-trivial weights
            # Weighted probit - statsmodels supports freq_weights
            X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
            try:
                mod_wt = Probit(valid['house_rep'].astype(float), X).fit(disp=0, freq_weights=wt)
                print(f'\n1992 weighted ({wt_col}): LL={mod_wt.llf:.1f}, R2={mod_wt.prsquared:.4f}')
                print(f'  Strong={mod_wt.params["strong_curr"]:.3f}({mod_wt.bse["strong_curr"]:.3f})')
                print(f'  Weak={mod_wt.params["weak_curr"]:.3f}({mod_wt.bse["weak_curr"]:.3f})')
                print(f'  Lean={mod_wt.params["lean_curr"]:.3f}({mod_wt.bse["lean_curr"]:.3f})')
                print(f'  Int={mod_wt.params["const"]:.3f}({mod_wt.bse["const"]:.3f})')
            except Exception as e:
                print(f'{wt_col}: error - {e}')

            # Try var_weights instead
            try:
                mod_vw = Probit(valid['house_rep'].astype(float), X).fit(disp=0, var_weights=wt)
                print(f'\n1992 var_weights ({wt_col}): LL={mod_vw.llf:.1f}, R2={mod_vw.prsquared:.4f}')
                print(f'  Strong={mod_vw.params["strong_curr"]:.3f}({mod_vw.bse["strong_curr"]:.3f})')
            except Exception as e:
                print(f'var_weights {wt_col}: error - {e}')
        else:
            print(f'{wt_col}: no variation in weights')
    else:
        print(f'{wt_col}: not available or all NaN')

# Unweighted for comparison
X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
mod_uw = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
print(f'\nUnweighted: LL={mod_uw.llf:.1f}, R2={mod_uw.prsquared:.4f}')
print(f'  Strong={mod_uw.params["strong_curr"]:.3f}({mod_uw.bse["strong_curr"]:.3f})')
print(f'  Weak={mod_uw.params["weak_curr"]:.3f}({mod_uw.bse["weak_curr"]:.3f})')
print(f'  Lean={mod_uw.params["lean_curr"]:.3f}({mod_uw.bse["lean_curr"]:.3f})')
print(f'  Int={mod_uw.params["const"]:.3f}({mod_uw.bse["const"]:.3f})')
print('Target: N=760, LL=-408.2, R2=0.20, Strong=0.975, Weak=0.627, Lean=0.472, Int=-0.211')
