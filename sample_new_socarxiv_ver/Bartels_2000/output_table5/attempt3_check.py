import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

# Check all data details for improvements

# 1960 panel
df60 = pd.read_csv('panel_1960.csv')
print("=== 1960 Panel Details ===")
print(f"Total respondents: {len(df60)}")
print(f"VCF0707 valid (1 or 2): {df60['VCF0707'].isin([1.0,2.0]).sum()}")
print(f"VCF0707 NaN: {df60['VCF0707'].isna().sum()}")
print(f"VCF0301 valid (1-7): {df60['VCF0301'].isin([1,2,3,4,5,6,7]).sum()}")
print(f"VCF0301 NaN: {df60['VCF0301'].isna().sum()}")
print(f"VCF0301_lagged valid (1-7): {df60['VCF0301_lagged'].isin([1,2,3,4,5,6,7]).sum()}")
print(f"VCF0301_lagged NaN: {df60['VCF0301_lagged'].isna().sum()}")

# What if we require only House vote + current PID for current model?
# And only House vote + lagged PID for lagged model?
# Rather than requiring ALL to be valid for both models
valid_curr = df60[
    df60['VCF0707'].isin([1.0,2.0]) &
    df60['VCF0301'].isin([1,2,3,4,5,6,7])
].copy()
print(f"\n1960 valid for current PID model: {len(valid_curr)}")

valid_lag = df60[
    df60['VCF0707'].isin([1.0,2.0]) &
    df60['VCF0301_lagged'].isin([1,2,3,4,5,6,7])
].copy()
print(f"1960 valid for lagged PID model: {len(valid_lag)}")

valid_both = df60[
    df60['VCF0707'].isin([1.0,2.0]) &
    df60['VCF0301'].isin([1,2,3,4,5,6,7]) &
    df60['VCF0301_lagged'].isin([1,2,3,4,5,6,7])
].copy()
print(f"1960 valid for both: {len(valid_both)}")

# If we use separate samples per model row, would that match better?
# The paper shows N=911 for all three rows
# Our valid_both=634, valid_curr may be higher

# 1976 panel
df76 = pd.read_csv('panel_1976.csv')
print(f"\n=== 1976 Panel Details ===")
print(f"Total: {len(df76)}")
valid_curr76 = df76[df76['VCF0707'].isin([1.0,2.0]) & df76['VCF0301'].isin([1,2,3,4,5,6,7])]
valid_lag76 = df76[df76['VCF0707'].isin([1.0,2.0]) & df76['VCF0301_lagged'].isin([1,2,3,4,5,6,7])]
valid_both76 = df76[df76['VCF0707'].isin([1.0,2.0]) & df76['VCF0301'].isin([1,2,3,4,5,6,7]) & df76['VCF0301_lagged'].isin([1,2,3,4,5,6,7])]
print(f"Valid current: {len(valid_curr76)}")
print(f"Valid lagged: {len(valid_lag76)}")
print(f"Valid both: {len(valid_both76)}")

# Check: maybe each row uses its own sample?
# Current PID row: vote + current PID valid
# Lagged PID row: vote + lagged PID valid
# IV row: vote + both PIDs valid (for first stage)

# Try with separate samples
print("\n\n=== Testing separate samples per model row ===")

# 1960 current PID only
vc60 = valid_curr.copy()
vc60['house_rep'] = (vc60['VCF0707'] == 2.0).astype(int)
vc60['strong'] = np.where(vc60['VCF0301']==7, 1, np.where(vc60['VCF0301']==1, -1, 0))
vc60['weak'] = np.where(vc60['VCF0301']==6, 1, np.where(vc60['VCF0301']==2, -1, 0))
vc60['lean'] = np.where(vc60['VCF0301']==5, 1, np.where(vc60['VCF0301']==3, -1, 0))
X = sm.add_constant(vc60[['strong','weak','lean']])
mod = Probit(vc60['house_rep'], X).fit(disp=0)
print(f"1960 current (separate sample N={len(vc60)}): Strong={mod.params['strong']:.3f}, Weak={mod.params['weak']:.3f}, Lean={mod.params['lean']:.3f}, Int={mod.params['const']:.3f}")
print(f"  LL={mod.llf:.1f}, R2={mod.prsquared:.4f}")
print(f"  Target: Strong=1.358, Weak=1.028, Lean=0.855, Int=0.035, LL=-372.7, R2=0.41")

# 1960 lagged PID only
vl60 = valid_lag.copy()
vl60['house_rep'] = (vl60['VCF0707'] == 2.0).astype(int)
vl60['strong'] = np.where(vl60['VCF0301_lagged']==7, 1, np.where(vl60['VCF0301_lagged']==1, -1, 0))
vl60['weak'] = np.where(vl60['VCF0301_lagged']==6, 1, np.where(vl60['VCF0301_lagged']==2, -1, 0))
vl60['lean'] = np.where(vl60['VCF0301_lagged']==5, 1, np.where(vl60['VCF0301_lagged']==3, -1, 0))
X2 = sm.add_constant(vl60[['strong','weak','lean']])
mod2 = Probit(vl60['house_rep'], X2).fit(disp=0)
print(f"\n1960 lagged (separate sample N={len(vl60)}): Strong={mod2.params['strong']:.3f}, Weak={mod2.params['weak']:.3f}, Lean={mod2.params['lean']:.3f}, Int={mod2.params['const']:.3f}")
print(f"  LL={mod2.llf:.1f}, R2={mod2.prsquared:.4f}")
print(f"  Target: Strong=1.363, Weak=0.842, Lean=0.564, Int=0.068, LL=-403.9, R2=0.36")
