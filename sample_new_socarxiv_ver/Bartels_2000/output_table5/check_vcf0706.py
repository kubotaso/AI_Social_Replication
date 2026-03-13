import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()
cdf90 = cdf[cdf['VCF0004']==1990].copy()

merged = cdf92_panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# VCF0706 = House vote 2-category (pre-election intent?)
# VCF0707 = House vote 2-category (post-election report)
# Let me check both
print("VCF0706 distribution:")
print(merged['VCF0706'].value_counts(dropna=False).sort_index())
print("\nVCF0707 distribution:")
print(merged['VCF0707'].value_counts(dropna=False).sort_index())

# Also check VCF0709 (House vote validated)
for col in ['VCF0709', 'VCF0710', 'VCF0711', 'VCF0712']:
    if col in merged.columns:
        vals = merged[col].value_counts(dropna=False)
        if not vals.empty:
            print(f"\n{col}:")
            print(vals.sort_index())

# Try VCF0706 instead of VCF0707
# VCF0706: 1=Dem, 2=Rep, 3=other, 4=wouldn't vote, 7=don't know
# This is pre-election, so probably not what Bartels used
# But let's check how many valid values
valid_706 = merged[
    merged['VCF0706'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
]
print(f"\nValid with VCF0706: {len(valid_706)}")

# Compare with VCF0707
valid_707 = merged[
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
]
print(f"Valid with VCF0707: {len(valid_707)}")

# Check the overlap
both_valid = merged[
    merged['VCF0706'].isin([1.0, 2.0]) &
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
]
print(f"Valid with both: {len(both_valid)}")

# Do the VCF0706 and VCF0707 values agree?
agree = (both_valid['VCF0706'] == both_valid['VCF0707']).sum()
print(f"Agreement between VCF0706 and VCF0707: {agree}/{len(both_valid)} ({agree/len(both_valid):.1%})")

# What if we use the UNION of VCF0706 and VCF0707?
# For respondents with valid VCF0707, use that. For those without, use VCF0706.
merged_copy = merged.copy()
merged_copy['house_vote'] = merged_copy['VCF0707']
# Fill NaN with VCF0706 where available
mask_fill = merged_copy['house_vote'].isna() & merged_copy['VCF0706'].isin([1.0, 2.0])
merged_copy.loc[mask_fill, 'house_vote'] = merged_copy.loc[mask_fill, 'VCF0706']
valid_union = merged_copy[
    merged_copy['house_vote'].isin([1.0, 2.0]) &
    merged_copy['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged_copy['VCF0301_lag'].isin([1,2,3,4,5,6,7])
]
print(f"\nValid with union (VCF0707 + VCF0706 fallback): {len(valid_union)}")

if len(valid_union) > len(valid_707):
    # Test if this changes results
    df = valid_union.copy()
    df['house_rep'] = (df['house_vote'] == 2.0).astype(int)
    df['strong'] = np.where(df['VCF0301']==7, 1, np.where(df['VCF0301']==1, -1, 0))
    df['weak'] = np.where(df['VCF0301']==6, 1, np.where(df['VCF0301']==2, -1, 0))
    df['lean'] = np.where(df['VCF0301']==5, 1, np.where(df['VCF0301']==3, -1, 0))

    X = sm.add_constant(df[['strong','weak','lean']])
    mod = Probit(df['house_rep'], X).fit(disp=0)
    print(f"Union approach: N={len(df)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}")
    print(f"Strong={mod.params['strong']:.3f}, Weak={mod.params['weak']:.3f}, Lean={mod.params['lean']:.3f}, Int={mod.params['const']:.3f}")
    print(f"Target: N=760, LL=-408.2, R2=0.20")
