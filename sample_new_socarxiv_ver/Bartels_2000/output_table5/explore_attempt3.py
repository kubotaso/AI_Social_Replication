import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

# The CDF VCF0301 for the lagged wave should be the same as the raw PID
# from the original study, harmonized to the CDF coding.
# But let's check: for the 1992 panel, the panel_1992.csv has
# pid_lagged derived from V900320 (1990 raw PID, 0-6 scale)
# CDF VCF0301_lag comes from the 1990 CDF entry

# Are these the same for matched respondents?
cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
p92 = pd.read_csv('panel_1992.csv')

# CDF 1992 panel respondents
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()
cdf90 = cdf[cdf['VCF0004']==1990].copy()
merged = cdf92_panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# The panel_1992.csv V900320 has values 0-6
# CDF VCF0301 for 1990 has values 1-7
# So V900320 + 1 = VCF0301 (for the 1990 wave)
# But V900320 is actually: 0=Strong Dem, ..., 6=Strong Rep
# While VCF0301: 1=Strong Dem, ..., 7=Strong Rep
# So the mapping V900320+1 = VCF0301 should hold

# However, V900320 in the 1992 study might have been recoded differently
# In the ANES 1992 individual study, PID might be coded:
# 0=Strong Rep, 1=Not-so-strong Rep, ..., 6=Strong Dem
# (REVERSED from standard!)

# The Table 4 analysis confirmed this: pid_current_std = 8 - pid_current
# This means pid_current=1 is Strong Rep (not Strong Dem)

# For V900320 (1990 raw PID), the coding might also be reversed:
# 0=Strong Rep, ..., 6=Strong Dem

# In the panel_1992.csv, pid_lagged = V900320 + 1
# If V900320: 0=StrongRep, ..., 6=StrongDem
# Then pid_lagged: 1=StrongRep, ..., 7=StrongDem

# But in the CDF, VCF0301 for 1990: 1=StrongDem, ..., 7=StrongRep

# So the mapping would be: pid_lagged = 8 - VCF0301_lag
# or equivalently: VCF0301_lag = 8 - pid_lagged

# Let me verify this
# Compare distributions
print("panel_1992 pid_lagged distribution:")
print(p92['pid_lagged'].value_counts().sort_index())
print("\nCDF 1990 VCF0301 distribution (for panel respondents):")
print(merged['VCF0301_lag'].value_counts().sort_index())

# If reversed: pid_lagged=1 (277 cases) should correspond to VCF0301_lag=7 (122 cases)
# But 277 != 122, so they're NOT simply reversed

# If same: pid_lagged=1 (277) should correspond to VCF0301_lag=1 (277)
# Let me check
print("\nCompare counts:")
for i in range(1, 8):
    p_count = (p92['pid_lagged']==i).sum()
    c_count = (merged['VCF0301_lag']==i).sum()
    r_count = (merged['VCF0301_lag']==(8-i)).sum()
    print(f"  pid_lagged={i}: {p_count}, VCF0301_lag={i}: {c_count}, VCF0301_lag={8-i}: {r_count}")

# Based on the counts, pid_lagged and VCF0301_lag should be the same coding
# (both 1=StrongDem), OR one is reversed

# pid_lagged: 1=277, 2=263, 3=170, 4=126, 5=170, 6=208, 7=122
# VCF0301_lag: 1=277, 2=262, 3=170, 4=141, 5=170, 6=208, 7=122

# These match almost perfectly (262 vs 263 for value 2, 141 vs 126 for value 4)
# Small differences because CDF has 1359 panel respondents vs 1336 in panel_1992
# The pid_lagged=1 count (277) matches VCF0301_lag=1 count (277), NOT VCF0301_lag=7 (122)
# So pid_lagged and VCF0301_lag have the SAME coding: 1=StrongDem, 7=StrongRep

# Wait, but Table 4 says pid_current is REVERSED. Let me check pid_current too
print("\n\npanel_1992 pid_current distribution:")
print(p92['pid_current'].value_counts(dropna=False).sort_index())
print("\nCDF 1992 VCF0301 distribution (for panel respondents):")
print(merged['VCF0301'].value_counts(dropna=False).sort_index())

# pid_current: 1=265, 2=234, 3=192, 4=139, 5=142, 6=200, 7=144
# VCF0301:     1=267, 2=236, 3=193, 4=166, 5=143, 6=202, 7=145

# These also match closely. pid_current=1 (265) ≈ VCF0301=1 (267)
# So pid_current and VCF0301 have the SAME coding

# BUT the Table 4 analysis says pid_current is reversed (1=StrongRep)!
# Let me check: if VCF0301=1 is StrongDem and pid_current=1 matches VCF0301=1,
# then pid_current=1 is ALSO StrongDem, and the Table 4 analysis was WRONG
# to reverse it.

# OR maybe the Table 4 analysis worked because of the vote_pres coding
# In panel_1992, vote_pres: 1=Dem, 2=Rep
# If the Table 4 code reverses PID and keeps vote=Rep,
# it would flip all coefficient signs... unless both are reversed

# Let me check the Table 4 results more carefully
# Table 4 for 1992 current PID: Strong=1.853, positive sign
# A positive Strong coefficient with vote=Rep means Strong Rep PID → more Rep vote
# If pid_current=1=StrongDem (standard) and vote=Rep, then Strong=7→+1, Strong=1→-1
# Strong Reps (7) → +1 → positive coef → more Rep vote. That's correct.

# If Table 4 reverses PID (new_pid = 8 - old_pid), then:
# old_pid=1→new_pid=7=StrongRep, Strong=7→+1
# old_pid=7→new_pid=1=StrongDem, Strong=1→-1
# Result would be SAME as not reversing!

# Actually no. The construct_pid_dummies function uses:
# Strong: +1 if pid==7, -1 if pid==1
# If pid_current=1=StrongDem (standard) and NOT reversed:
#   StrongDem(1) → Strong=-1, StrongRep(7) → Strong=+1
#   Positive coef = Rep direction → correct
# If reversed (new_pid = 8-pid): StrongDem(7) → Strong=+1, StrongRep(1) → Strong=-1
#   Positive coef would mean Dem direction → WRONG

# So Table 4 reversing PID would give wrong signs... unless vote is also reversed
# In Table 4 for 1992: vote = (vote_pres == 2) = Rep
# If reversed PID: StrongDem has Strong=+1, positive coef → more Rep vote → WRONG

# This is contradictory. Let me just test both approaches and see which matches
print("\n\n=== Test both PID approaches for 1992 House vote ===")

# Valid data
valid = merged[
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
].copy()

valid['house_rep'] = (valid['VCF0707'] == 2.0).astype(int)

# Approach 1: Standard coding (VCF0301 1=StrongDem, 7=StrongRep)
valid['strong_std'] = np.where(valid['VCF0301']==7, 1, np.where(valid['VCF0301']==1, -1, 0))
valid['weak_std'] = np.where(valid['VCF0301']==6, 1, np.where(valid['VCF0301']==2, -1, 0))
valid['lean_std'] = np.where(valid['VCF0301']==5, 1, np.where(valid['VCF0301']==3, -1, 0))

X_std = sm.add_constant(valid[['strong_std','weak_std','lean_std']])
mod_std = Probit(valid['house_rep'], X_std).fit(disp=0)
print(f"\nStandard coding (VCF0301 as-is):")
print(f"N={len(valid)}, LL={mod_std.llf:.1f}, R2={mod_std.prsquared:.4f}")
print(f"Strong={mod_std.params['strong_std']:.3f}({mod_std.bse['strong_std']:.3f})")
print(f"Weak={mod_std.params['weak_std']:.3f}({mod_std.bse['weak_std']:.3f})")
print(f"Lean={mod_std.params['lean_std']:.3f}({mod_std.bse['lean_std']:.3f})")
print(f"Int={mod_std.params['const']:.3f}({mod_std.bse['const']:.3f})")

# Approach 2: Reversed coding (VCF0301: 8-pid)
valid['pid_rev'] = 8 - valid['VCF0301']
valid['strong_rev'] = np.where(valid['pid_rev']==7, 1, np.where(valid['pid_rev']==1, -1, 0))
valid['weak_rev'] = np.where(valid['pid_rev']==6, 1, np.where(valid['pid_rev']==2, -1, 0))
valid['lean_rev'] = np.where(valid['pid_rev']==5, 1, np.where(valid['pid_rev']==3, -1, 0))

X_rev = sm.add_constant(valid[['strong_rev','weak_rev','lean_rev']])
mod_rev = Probit(valid['house_rep'], X_rev).fit(disp=0)
print(f"\nReversed coding (8 - VCF0301):")
print(f"N={len(valid)}, LL={mod_rev.llf:.1f}, R2={mod_rev.prsquared:.4f}")
print(f"Strong={mod_rev.params['strong_rev']:.3f}({mod_rev.bse['strong_rev']:.3f})")
print(f"Weak={mod_rev.params['weak_rev']:.3f}({mod_rev.bse['weak_rev']:.3f})")
print(f"Lean={mod_rev.params['lean_rev']:.3f}({mod_rev.bse['lean_rev']:.3f})")
print(f"Int={mod_rev.params['const']:.3f}({mod_rev.bse['const']:.3f})")

print(f"\nTarget: Strong=0.975(0.094), Weak=0.627(0.084), Lean=0.472(0.098), Int=-0.211(0.051)")
print(f"Target: LL=-408.2, R2=0.20")
