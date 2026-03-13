import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

# 1976 panel
cdf76 = cdf[cdf['VCF0004']==1976].copy()
cdf76_panel = cdf76[cdf76['VCF0006a'] < 19760000].copy()
cdf74 = cdf[cdf['VCF0004']==1974].copy()

# Agreement between VCF0706 and VCF0707
both = cdf76_panel[cdf76_panel['VCF0706'].isin([1.0,2.0]) & cdf76_panel['VCF0707'].isin([1.0,2.0])]
agree = (both['VCF0706'] == both['VCF0707']).sum()
print(f"1976 VCF0706 vs VCF0707 agreement: {agree}/{len(both)} = {agree/len(both):.1%}")
print(pd.crosstab(both['VCF0706'], both['VCF0707']))

# Try using VCF0706 + VCF0707 union for 1976
merged76 = cdf76_panel.merge(cdf74[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# Create combined House vote variable
merged76['house_vote'] = merged76['VCF0707']
# Fill missing with VCF0706
mask_fill = merged76['house_vote'].isna() & merged76['VCF0706'].isin([1.0, 2.0])
merged76.loc[mask_fill, 'house_vote'] = merged76.loc[mask_fill, 'VCF0706']

# Filter
valid = merged76[
    merged76['house_vote'].isin([1.0, 2.0]) &
    merged76['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged76['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()

print(f"\n1976 union N: {len(valid)} (target 682)")

valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
valid['strong'] = np.where(valid['VCF0301']==7, 1, np.where(valid['VCF0301']==1, -1, 0))
valid['weak'] = np.where(valid['VCF0301']==6, 1, np.where(valid['VCF0301']==2, -1, 0))
valid['lean'] = np.where(valid['VCF0301']==5, 1, np.where(valid['VCF0301']==3, -1, 0))

X = sm.add_constant(valid[['strong','weak','lean']])
mod = Probit(valid['house_rep'], X).fit(disp=0)
print(f"N={len(valid)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}")
print(f"Strong={mod.params['strong']:.3f}({mod.bse['strong']:.3f})")
print(f"Weak={mod.params['weak']:.3f}({mod.bse['weak']:.3f})")
print(f"Lean={mod.params['lean']:.3f}({mod.bse['lean']:.3f})")
print(f"Int={mod.params['const']:.3f}({mod.bse['const']:.3f})")
print(f"Target: N=682, LL=-358.2, R2=0.24, Strong=1.087, Weak=0.624, Lean=0.622, Int=-0.123")

# Same for 1960
print("\n=== 1960 ===")
cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf60_panel = cdf60[cdf60['VCF0006a'] < 19600000].copy()
cdf58 = cdf[cdf['VCF0004']==1958].copy()

both60 = cdf60_panel[cdf60_panel['VCF0706'].isin([1.0,2.0]) & cdf60_panel['VCF0707'].isin([1.0,2.0])]
agree60 = (both60['VCF0706'] == both60['VCF0707']).sum()
print(f"1960 VCF0706 vs VCF0707 agreement: {agree60}/{len(both60)} = {agree60/len(both60):.1%}")

merged60 = cdf60_panel.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
merged60['house_vote'] = merged60['VCF0707']
mask_fill60 = merged60['house_vote'].isna() & merged60['VCF0706'].isin([1.0, 2.0])
merged60.loc[mask_fill60, 'house_vote'] = merged60.loc[mask_fill60, 'VCF0706']

valid60 = merged60[
    merged60['house_vote'].isin([1.0, 2.0]) &
    merged60['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged60['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()

print(f"1960 union N: {len(valid60)} (target 911)")

valid60['house_rep'] = (valid60['house_vote'] == 2.0).astype(int)
valid60['strong'] = np.where(valid60['VCF0301']==7, 1, np.where(valid60['VCF0301']==1, -1, 0))
valid60['weak'] = np.where(valid60['VCF0301']==6, 1, np.where(valid60['VCF0301']==2, -1, 0))
valid60['lean'] = np.where(valid60['VCF0301']==5, 1, np.where(valid60['VCF0301']==3, -1, 0))

X60 = sm.add_constant(valid60[['strong','weak','lean']])
mod60 = Probit(valid60['house_rep'], X60).fit(disp=0)
print(f"N={len(valid60)}, LL={mod60.llf:.1f}, R2={mod60.prsquared:.4f}")
print(f"Strong={mod60.params['strong']:.3f}({mod60.bse['strong']:.3f})")
print(f"Weak={mod60.params['weak']:.3f}({mod60.bse['weak']:.3f})")
print(f"Lean={mod60.params['lean']:.3f}({mod60.bse['lean']:.3f})")
print(f"Int={mod60.params['const']:.3f}({mod60.bse['const']:.3f})")
print(f"Target: N=911, LL=-372.7, R2=0.41, Strong=1.358, Weak=1.028, Lean=0.855, Int=0.035")
