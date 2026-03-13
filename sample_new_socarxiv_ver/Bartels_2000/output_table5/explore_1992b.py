import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

# Try using the panel_1992.csv with correct House vote coding
# The issue is that vote_house in panel_1992 maps V925701 1->1, 5->2
# But V925701 codes ballot position, not party
#
# What if we need to use the PARTY of the candidate?
# V925609 might encode the party of house candidates in the district

p92 = pd.read_csv('panel_1992.csv')
cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

# Get CDF data for 1992 panel respondents
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()
cdf90 = cdf[cdf['VCF0004']==1990].copy()

# Merge 1992 panel with 1990 for lagged PID
merged = cdf92_panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# Valid sample
mask = (
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
)
df = merged[mask].copy()
print(f"N with all valid: {len(df)}")

# Check vote distribution
print("\nHouse vote distribution:")
print(df['VCF0707'].value_counts().sort_index())
print(f"Dem proportion: {(df['VCF0707']==1.0).sum()/len(df):.3f}")
print(f"Rep proportion: {(df['VCF0707']==2.0).sum()/len(df):.3f}")

# The paper target LL=-408.2 for current PID model
# Our LL=-393.9 is more negative (in absolute terms, less negative = better fit)
# This could happen if our sample has a different partisan composition

# Let's check what the null model LL would be
from scipy.stats import norm
p_rep = (df['VCF0707']==2.0).sum() / len(df)
null_ll = len(df) * (p_rep * np.log(p_rep) + (1-p_rep) * np.log(1-p_rep))
print(f"\nNull model LL: {null_ll:.1f}")
print(f"Probit LL: -393.9")
print(f"Pseudo-R2 = 1 - (-393.9 / {null_ll:.1f}) = {1 - (-393.9/null_ll):.4f}")

# For target: null_ll that gives R2=0.20 with LL=-408.2
# 0.20 = 1 - (-408.2/null_ll) => null_ll = -408.2 / (1-0.20) = -510.25
target_null = -408.2 / (1-0.20)
print(f"\nTarget null LL (from R2=0.20, LL=-408.2): {target_null:.1f}")
# Our null_ll would need to be around -510 for the numbers to work
# With N=760 and p_rep around 0.39, null_ll ≈ -510

# Check what p_rep gives null_ll ≈ -510
# null_ll = N * [p*log(p) + (1-p)*log(1-p)]
# -510 = 760 * [p*log(p) + (1-p)*log(1-p)]
# => p*log(p) + (1-p)*log(1-p) = -510/760 = -0.6711
# This is the entropy function. For p=0.39: 0.39*log(0.39) + 0.61*log(0.61) = -0.6665
# For p=0.386: 0.386*log(0.386) + 0.614*log(0.614) =
for p in [0.35, 0.36, 0.37, 0.38, 0.385, 0.386, 0.387, 0.388, 0.389, 0.39, 0.40]:
    ent = p*np.log(p) + (1-p)*np.log(1-p)
    null = 760 * ent
    print(f"  p={p:.3f}: null_LL={null:.1f}, R2 from LL=-408.2: {1-(-408.2/null):.4f}")

# Our actual data
print(f"\nOur p_rep: {p_rep:.4f}")
print(f"Our null LL: {null_ll:.1f}")

# Check: with VCF0707, what if we also look at VCF0706?
# VCF0706 is the House vote intention (pre-election)
# VCF0707 is the reported House vote (post-election)
print("\n=== Check VCF0706 ===")
if 'VCF0706' in merged.columns:
    print("VCF0706 available")
    print(merged['VCF0706'].value_counts().sort_index())
else:
    print("VCF0706 not in merged data")

# The 1992 LL difference might be because Bartels had slightly different
# vote data (perhaps from the individual 1992 study rather than CDF)
#
# Let's try: what if we use the individual 1992 study variables instead?
# The panel_1992.csv has V925701 and V925609
# V925609 might be party of the candidate
# If V925609=1 for the candidate the respondent voted for => Dem vote
# If V925609=2 => Rep vote

# But we need the party of the VOTED-FOR candidate, not just any candidate
# V925609 might be the party of candidate 1 in the district
# Then we'd need V925610 for candidate 2's party

# Let me check: for voters who chose candidate 1 (V925701=1), is V925609
# the party of candidate 1? And for V925701=5, is there another variable
# for candidate 2's party?

# Actually, let me check whether the CDF panel IDs exactly match the
# panel_1992 respondents
# The CDF has 1359 panel respondents, panel_1992 has 1336

# We can try an alternative approach: directly match panel_1992 to CDF
# using all demographic variables to find the best match

# Actually, let me try: sort both datasets and assume row-by-row match
# No, that's unreliable

# Better: use a combination approach
# For the 1992 current PID model, use the CDF VCF0707 and CDF VCF0301
# For the lagged PID, use CDF 1990 VCF0301
# This is what we're already doing

# The 14-point LL gap might just be due to slightly different sample composition
# Let's accept this and focus on other improvements

print("\n=== Probit with current approach ===")
df['house_rep'] = (df['VCF0707'] == 2.0).astype(int)
df['strong_curr'] = np.where(df['VCF0301']==7, 1, np.where(df['VCF0301']==1, -1, 0))
df['weak_curr'] = np.where(df['VCF0301']==6, 1, np.where(df['VCF0301']==2, -1, 0))
df['lean_curr'] = np.where(df['VCF0301']==5, 1, np.where(df['VCF0301']==3, -1, 0))
df['strong_lag'] = np.where(df['VCF0301_lag']==7, 1, np.where(df['VCF0301_lag']==1, -1, 0))
df['weak_lag'] = np.where(df['VCF0301_lag']==6, 1, np.where(df['VCF0301_lag']==2, -1, 0))
df['lean_lag'] = np.where(df['VCF0301_lag']==5, 1, np.where(df['VCF0301_lag']==3, -1, 0))

X = sm.add_constant(df[['strong_curr','weak_curr','lean_curr']])
mod = Probit(df['house_rep'], X).fit(disp=0)
print(mod.summary2().tables[1])
print(f"LL: {mod.llf:.1f}, R2: {mod.prsquared:.4f}")

# Compare: try probit directly with statsmodels Probit endog
# Using Probit from statsmodels should give standard MLE
print("\nNull model:")
null_mod = Probit(df['house_rep'], np.ones(len(df))).fit(disp=0)
print(f"Null LL: {null_mod.llf:.1f}")
print(f"Computed R2: {1 - mod.llf/null_mod.llf:.4f}")
