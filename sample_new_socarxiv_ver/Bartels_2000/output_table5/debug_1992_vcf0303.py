import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

# 1992 panel
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()
cdf90 = cdf[cdf['VCF0004']==1990].copy()

merged = cdf92_panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# Check VCF0303 distribution
print('VCF0303 dist:')
print(merged['VCF0303'].value_counts(dropna=False).sort_index())

# VCF0303 might be coded differently - check the codebook
# VCF0303 is probably pre-election party ID with different coding
# Let's check what values it has
v303_vals = merged['VCF0303'].dropna().unique()
print(f'\nUnique VCF0303 values: {sorted(v303_vals)}')

# Check VCF0301 vs VCF0303 cross-tab for 1992
ct = pd.crosstab(merged['VCF0301'], merged['VCF0303'], margins=True)
print('\nVCF0301 vs VCF0303 crosstab:')
print(ct)

# The issue might be that VCF0303 has fewer categories or different coding
# Try treating VCF0303 as a 3-point scale: 1=Dem, 3=Ind, 5=Rep
# Or maybe VCF0303 is party identification strength (not the 7-point)

# Let me try something different: what if the 1992 difference is because
# Bartels used a slightly different sample?
# Target intercept is -0.211, ours is -0.239
# Intercept = Phi^{-1}(p_rep among independents)
# For 1992: if 4=pure independent, let's check their vote
valid = merged[
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()

ind = valid[valid['VCF0301'] == 4]
print(f'\nIndependents (VCF0301=4): N={len(ind)}')
print(f'Dem: {(ind["VCF0707"]==1.0).sum()}, Rep: {(ind["VCF0707"]==2.0).sum()}')
rep_rate = (ind['VCF0707']==2.0).mean()
print(f'Rep rate: {rep_rate:.4f}')
from scipy.stats import norm
print(f'Phi^-1({rep_rate:.4f}) = {norm.ppf(rep_rate):.3f} (should be close to -0.211)')

# Let's also check: what N would give LL=-408.2 for current PID model?
# LL scales roughly as N * average_log_prob
# Our LL=-393.9 with N=759, so avg_log_prob = -393.9/759 = -0.519
# Target LL=-408.2, so N_target = -408.2 / -0.519 = 786
# But target N is 760, so avg_log_prob_target = -408.2/760 = -0.537
# Our model fits better (lower avg negative log prob) → our R2 is higher
# This means our sample is more partisan-consistent

# What if we include respondents with PID=0 (apolitical)?
valid0 = merged[
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([0,1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([0,1,2,3,4,5,6,7])
].copy()
print(f'\nWith PID 0 allowed: N={len(valid0)}')

# Check PID=0 respondents
pid0 = merged[(merged['VCF0301'] == 0) & merged['VCF0707'].isin([1.0,2.0])]
print(f'PID=0 with valid vote: {len(pid0)}')

# What about including VCF0301 = 8 or 9?
for val in [0, 8, 9]:
    subset = merged[(merged['VCF0301'] == val) & merged['VCF0707'].isin([1.0,2.0])]
    print(f'VCF0301={val} with valid vote: {len(subset)}')
    subset_lag = merged[(merged['VCF0301_lag'] == val) & merged['VCF0707'].isin([1.0,2.0])]
    print(f'VCF0301_lag={val} with valid vote: {len(subset_lag)}')

# Run probit with N=760 by including ONE more respondent
# Target N=760 vs our N=759 - could there be 1 more valid respondent?
# Check if we're missing someone due to PID=0 treated as valid
valid_with_0 = merged[
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([0,1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()
print(f'\nN with PID=0 allowed for current (but not lagged): {len(valid_with_0)}')

valid_with_0b = merged[
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([0,1,2,3,4,5,6,7])
].copy()
print(f'N with PID=0 allowed for lagged (but not current): {len(valid_with_0b)}')

# Run probit with this extra respondent
if len(valid_with_0b) == 760:
    valid_with_0b['house_rep'] = (valid_with_0b['VCF0707'] == 2.0).astype(int)
    valid_with_0b['strong'] = np.where(valid_with_0b['VCF0301']==7, 1, np.where(valid_with_0b['VCF0301']==1, -1, 0))
    valid_with_0b['weak'] = np.where(valid_with_0b['VCF0301']==6, 1, np.where(valid_with_0b['VCF0301']==2, -1, 0))
    valid_with_0b['lean'] = np.where(valid_with_0b['VCF0301']==5, 1, np.where(valid_with_0b['VCF0301']==3, -1, 0))
    X = sm.add_constant(valid_with_0b[['strong','weak','lean']])
    mod = Probit(valid_with_0b['house_rep'], X).fit(disp=0)
    print(f'  LL={mod.llf:.1f}, R2={mod.prsquared:.4f}')
    print(f'  Strong={mod.params["strong"]:.3f}({mod.bse["strong"]:.3f})')
