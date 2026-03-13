import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

p92 = pd.read_csv('panel_1992.csv')

# V925609 values: 0 (417), 1 (291), 2 (437), 3 (172), 7 (3), 9 (16)
# These might be: party of House candidate #1
# 0 = no candidate / NA
# 1 = Democrat
# 2 = Republican
# 3 = Independent

# V925701: 0 (no vote), 1 (candidate 1), 5 (candidate 2)

# If V925609 is party of candidate 1, then:
# V925701=1 and V925609=1 => voted for Dem candidate 1
# V925701=1 and V925609=2 => voted for Rep candidate 1
# V925701=5 => voted for candidate 2, whose party is NOT V925609

# But we don't have V925610 (party of candidate 2)

# Let me think differently. Maybe V925609 is NOT party of candidate 1 only.
# Let me check if it correlates with PID in a way that makes sense
# as "party of the candidate voted for"

# Cross-tab V925609 with pid_current
print("V925609 vs pid_current (for V925701=1 voters):")
v1 = p92[p92['V925701']==1]
print(pd.crosstab(v1['pid_current'], v1['V925609']))

print("\nV925609 vs pid_current (for V925701=5 voters):")
v5 = p92[p92['V925701']==5]
print(pd.crosstab(v5['pid_current'], v5['V925609']))

# If V925609 is party of the candidate voted for:
# Strong Dems (pid=1) should have V925609=1 (Dem) mostly
# Strong Reps (pid=7) should have V925609=2 (Rep) mostly

# But wait - in the Table 4 analysis, pid_current is REVERSED
# So pid_current=1 is actually Strong Rep and pid_current=7 is Strong Dem
# Let me check with the reversed coding

print("\n\nWith reversed PID (1=StrongRep -> 7=StrongDem):")
# pid_current in panel_1992: 1=Strong Rep, 7=Strong Dem
# So pid=7 (Strong Dem) should correlate with Dem vote
# pid=1 (Strong Rep) should correlate with Rep vote

# For V925701=1 voters:
# pid=7 (Strong Dem) with V925609=1 (Dem) should be common
# pid=1 (Strong Rep) with V925609=2 (Rep) should be common
print("For candidate 1 voters (V925701=1):")
print("  pid=7 (StrongDem) and V925609=1 (Dem):",
      len(v1[(v1['pid_current']==7) & (v1['V925609']==1)]))
print("  pid=7 (StrongDem) and V925609=2 (Rep):",
      len(v1[(v1['pid_current']==7) & (v1['V925609']==2)]))
print("  pid=1 (StrongRep) and V925609=1 (Dem):",
      len(v1[(v1['pid_current']==1) & (v1['V925609']==1)]))
print("  pid=1 (StrongRep) and V925609=2 (Rep):",
      len(v1[(v1['pid_current']==1) & (v1['V925609']==2)]))

print("\nFor candidate 2 voters (V925701=5):")
print("  pid=7 (StrongDem) and V925609=1 (Dem):",
      len(v5[(v5['pid_current']==7) & (v5['V925609']==1)]))
print("  pid=7 (StrongDem) and V925609=2 (Rep):",
      len(v5[(v5['pid_current']==7) & (v5['V925609']==2)]))
print("  pid=1 (StrongRep) and V925609=1 (Dem):",
      len(v5[(v5['pid_current']==1) & (v5['V925609']==1)]))
print("  pid=1 (StrongRep) and V925609=2 (Rep):",
      len(v5[(v5['pid_current']==1) & (v5['V925609']==2)]))

# What if V925609 = party of candidate 1?
# Then for candidate 1 voters: their party = V925609
# For candidate 2 voters: their party = the OTHER major party
# (assuming 2-party races)

# Let me try this approach
print("\n\n=== Approach: V925609 = party of candidate 1 ===")
# For V925701=1: vote_party = V925609
# For V925701=5: vote_party = opposite of V925609
# Only for 2-party races (V925609 in [1,2])

# Create party-of-voted-candidate variable
df = p92.copy()
df['vote_party'] = np.nan

# Candidate 1 voters: party = V925609
mask1 = (df['V925701'] == 1) & (df['V925609'].isin([1, 2]))
df.loc[mask1, 'vote_party'] = df.loc[mask1, 'V925609']

# Candidate 2 voters: party = opposite of V925609 (assuming 2-party race)
mask5_dem = (df['V925701'] == 5) & (df['V925609'] == 1)
df.loc[mask5_dem, 'vote_party'] = 2  # If cand1 is Dem, cand2 is Rep
mask5_rep = (df['V925701'] == 5) & (df['V925609'] == 2)
df.loc[mask5_rep, 'vote_party'] = 1  # If cand1 is Rep, cand2 is Dem

print("vote_party distribution:")
print(df['vote_party'].value_counts(dropna=False).sort_index())

# Check if this makes sense with PID
# Remember: pid=1=StrongRep, pid=7=StrongDem (reversed!)
valid = df[df['vote_party'].isin([1,2]) & df['pid_current'].isin([1,2,3,4,5,6,7])]
# Reverse PID: new_pid = 8 - pid
valid_copy = valid.copy()
valid_copy['pid_std'] = 8 - valid_copy['pid_current']
print("\nCrosstab (standard PID vs vote_party):")
print(pd.crosstab(valid_copy['pid_std'], valid_copy['vote_party']))
# pid_std=1 (Strong Dem) should vote party=1 (Dem)
# pid_std=7 (Strong Rep) should vote party=2 (Rep)

# Now try probit
valid_copy['house_rep'] = (valid_copy['vote_party'] == 2).astype(int)
valid_copy['strong'] = np.where(valid_copy['pid_std']==7, 1, np.where(valid_copy['pid_std']==1, -1, 0))
valid_copy['weak'] = np.where(valid_copy['pid_std']==6, 1, np.where(valid_copy['pid_std']==2, -1, 0))
valid_copy['lean'] = np.where(valid_copy['pid_std']==5, 1, np.where(valid_copy['pid_std']==3, -1, 0))

# Also need lagged PID - reverse it too
valid_copy['pid_lag_std'] = 8 - valid_copy['pid_lagged']
valid_copy = valid_copy[valid_copy['pid_lag_std'].isin([1,2,3,4,5,6,7])]

X = sm.add_constant(valid_copy[['strong','weak','lean']])
if len(valid_copy) > 10:
    mod = Probit(valid_copy['house_rep'], X).fit(disp=0)
    print(f"\nProbit with party-based vote coding:")
    print(f"N={len(valid_copy)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}")
    print(f"Strong={mod.params['strong']:.3f}, Weak={mod.params['weak']:.3f}, Lean={mod.params['lean']:.3f}, Int={mod.params['const']:.3f}")
    print(f"Target: N=760, LL=-408.2, R2=0.20, Strong=0.975, Weak=0.627, Lean=0.472, Int=-0.211")
