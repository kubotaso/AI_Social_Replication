import pandas as pd
import numpy as np

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf58 = cdf[cdf['VCF0004']==1958].copy()
cdf60_panel = cdf60[cdf60['VCF0006a'] < 19600000].copy()
merged = cdf60_panel.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
print('Merged:', len(merged))
print('VCF0707 valid:', merged['VCF0707'].isin([1.0,2.0]).sum())
print('VCF0706 valid:', merged['VCF0706'].isin([1.0,2.0]).sum())

# Union
merged['hv'] = merged['VCF0707']
m = merged['hv'].isna() & merged['VCF0706'].isin([1.0,2.0])
merged.loc[m, 'hv'] = merged.loc[m, 'VCF0706']
print('Union valid:', merged['hv'].isin([1.0,2.0]).sum())
print('PID cur valid:', merged['VCF0301'].isin([1,2,3,4,5,6,7]).sum())
print('PID lag valid:', merged['VCF0301_lag'].isin([1,2,3,4,5,6,7]).sum())

v = merged[merged['hv'].isin([1.0,2.0]) & merged['VCF0301'].isin([1,2,3,4,5,6,7]) & merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])]
print('All valid:', len(v))

# What about just vote + current PID (for current PID model)?
v2 = merged[merged['hv'].isin([1.0,2.0]) & merged['VCF0301'].isin([1,2,3,4,5,6,7])]
print('Vote + current PID only:', len(v2))

# VCF0707 distribution
print('\nVCF0707 dist:', merged['VCF0707'].value_counts(dropna=False).sort_index().to_dict())
print('VCF0706 dist:', merged['VCF0706'].value_counts(dropna=False).sort_index().to_dict())

# Check VCF0301_lag distribution - maybe we need to allow 0 or 8/9?
print('\nVCF0301_lag dist:', merged['VCF0301_lag'].value_counts(dropna=False).sort_index().to_dict())
print('VCF0301 dist:', merged['VCF0301'].value_counts(dropna=False).sort_index().to_dict())

# What if lagged PID includes apolitical (0)?
v3 = merged[merged['hv'].isin([1.0,2.0]) & merged['VCF0301'].isin(range(8)) & merged['VCF0301_lag'].isin(range(8))]
print('\nWith PID 0 allowed:', len(v3))

# What if we use VCF0301 from 1958 wave directly from the CDF?
# Check VCF0302 (alternative PID)
if 'VCF0302' in merged.columns:
    print('\nVCF0302 dist:', merged['VCF0302'].value_counts(dropna=False).sort_index().to_dict())

# Try including 3rd party voters (VCF0707 == 3 or other values)?
print('\nAll VCF0707 values:', merged['VCF0707'].value_counts(dropna=False).sort_index().to_dict())

# Check if VCF0706 has values that VCF0707 doesn't and vice versa
both_valid = merged[merged['VCF0707'].isin([1.0,2.0]) & merged['VCF0706'].isin([1.0,2.0])]
agree = (both_valid['VCF0707'] == both_valid['VCF0706']).sum()
print(f'\nBoth valid: {len(both_valid)}, agree: {agree}, disagree: {len(both_valid)-agree}')

# 1992 investigation
print('\n\n=== 1992 ===')
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()
cdf90 = cdf[cdf['VCF0004']==1990].copy()
m92 = cdf92_panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
print('Merged:', len(m92))
v92 = m92[m92['VCF0707'].isin([1.0,2.0]) & m92['VCF0301'].isin([1,2,3,4,5,6,7]) & m92['VCF0301_lag'].isin([1,2,3,4,5,6,7])]
print('Valid:', len(v92))
print('House vote dist:', v92['VCF0707'].value_counts().sort_index().to_dict())
print('Dem pct:', (v92['VCF0707']==1.0).mean())
print('PID dist:', v92['VCF0301'].value_counts().sort_index().to_dict())
print('PID lag dist:', v92['VCF0301_lag'].value_counts().sort_index().to_dict())

# Maybe Bartels used VCF0706 for 1992 as well?
v92b = m92[m92['VCF0706'].isin([1.0,2.0]) & m92['VCF0301'].isin([1,2,3,4,5,6,7]) & m92['VCF0301_lag'].isin([1,2,3,4,5,6,7])]
print('\n1992 with VCF0706:', len(v92b))
# Union
m92['hv'] = m92['VCF0707']
mask = m92['hv'].isna() & m92['VCF0706'].isin([1.0,2.0])
m92.loc[mask, 'hv'] = m92.loc[mask, 'VCF0706']
v92c = m92[m92['hv'].isin([1.0,2.0]) & m92['VCF0301'].isin([1,2,3,4,5,6,7]) & m92['VCF0301_lag'].isin([1,2,3,4,5,6,7])]
print('1992 union:', len(v92c))

# Try running probit with VCF0706 for 1992
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

for name, vv, vote_col in [('VCF0707', v92, 'VCF0707'), ('VCF0706', v92b, 'VCF0706'), ('Union', v92c, 'hv')]:
    df = vv.copy() if name != 'Union' else v92c.copy()
    vc = vote_col
    df['house_rep'] = (df[vc] == 2.0).astype(int)
    df['strong'] = np.where(df['VCF0301']==7, 1, np.where(df['VCF0301']==1, -1, 0))
    df['weak'] = np.where(df['VCF0301']==6, 1, np.where(df['VCF0301']==2, -1, 0))
    df['lean'] = np.where(df['VCF0301']==5, 1, np.where(df['VCF0301']==3, -1, 0))
    X = sm.add_constant(df[['strong','weak','lean']])
    mod = Probit(df['house_rep'], X).fit(disp=0)
    print(f'\n1992 {name}: N={len(df)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}')
    print(f'  Strong={mod.params["strong"]:.3f}({mod.bse["strong"]:.3f})')
    print(f'  Weak={mod.params["weak"]:.3f}({mod.bse["weak"]:.3f})')
    print(f'  Lean={mod.params["lean"]:.3f}({mod.bse["lean"]:.3f})')
    print(f'  Int={mod.params["const"]:.3f}({mod.bse["const"]:.3f})')
print('Target: N=760, LL=-408.2, R2=0.20, Strong=0.975, Weak=0.627, Lean=0.472, Int=-0.211')
