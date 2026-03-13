import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

# Load the newly extracted 1992 data
df_1992_vars = pd.read_csv('output_table5/anes_1992_house_vars.csv')
print("1992 individual study variables:")
print(f"N: {len(df_1992_vars)}")

# V925608: Party of House candidate 1
# V925610: Party of House candidate 2
# V925701: Voted for which candidate (1=cand1, 5=cand2)
# V925609: Something else (previously checked - maybe party of House candidate that was listed first?)

print("\nV925608 (party cand 1):", df_1992_vars['V925608'].value_counts().sort_index().to_dict())
print("V925610 (party cand 2):", df_1992_vars['V925610'].value_counts().sort_index().to_dict())
print("V925701 (vote):", df_1992_vars['V925701'].value_counts().sort_index().to_dict())
print("V925609:", df_1992_vars['V925609'].value_counts().sort_index().to_dict())
print("V925611:", df_1992_vars['V925611'].value_counts().sort_index().to_dict())

# V925608 codes: 0(1043), 1(667), 5(672), 8(74), 9(29)
# V925610 codes: 0(1041), 1(627), 5(702), 8(84), 9(31)
# Assuming: 1=Democrat, 5=Republican (matching V925701 coding!)

# For V925701=1 (voted cand1): party = V925608
# For V925701=5 (voted cand2): party = V925610

# Construct party-of-voted-candidate
df = df_1992_vars.copy()
df['vote_party'] = np.nan

# Candidate 1 voters
mask1 = df['V925701'] == 1
df.loc[mask1 & (df['V925608'] == 1), 'vote_party'] = 1  # Voted Dem
df.loc[mask1 & (df['V925608'] == 5), 'vote_party'] = 2  # Voted Rep

# Candidate 2 voters
mask5 = df['V925701'] == 5
df.loc[mask5 & (df['V925610'] == 1), 'vote_party'] = 1  # Voted Dem
df.loc[mask5 & (df['V925610'] == 5), 'vote_party'] = 2  # Voted Rep

print("\nvote_party (party of candidate voted for):")
print(df['vote_party'].value_counts(dropna=False).sort_index())

# Now load the panel_1992 data and match
p92 = pd.read_csv('panel_1992.csv')
print(f"\npanel_1992 N: {len(p92)}")

# Both datasets have V925701 - we can check if they're aligned
# panel_1992 also has V925609 but NOT V925608/V925610

# Actually, the individual study has 2485 respondents.
# The panel_1992 has 1336 respondents (panel subset).
# We need to match them by some key.

# The individual study has V923634 (raw PID 0-9)
# panel_1992 also has V923634

# But V923634 is NOT a case ID - it's PID values 0-9.
# We need a different approach.

# Check: do the datasets share any unique identifiers?
print("\npanel_1992 columns:", p92.columns.tolist())

# The V925608, V925609, V925610, V925611 are in both?
# No - panel_1992 only has: pid_current, pid_lagged, vote_pres, vote_house, V923634, V900320, V925609, V925701

# The individual study has V925608 and V925610 (party of each candidate)
# But panel_1992 does NOT have these.

# Can we match by (V923634, V925609, V925701) as a composite key?
# V923634: PID (0-9 values)
# V925609: party of something (0,1,2,3,7,9 values)
# V925701: vote (0,1,5,8,9 values)

# Create match keys
df['match_key'] = df['V923634'].astype(str) + '_' + df['V925609'].astype(str) + '_' + df['V925701'].astype(str)
p92['match_key'] = p92['V923634'].astype(str) + '_' + p92['V925609'].astype(str) + '_' + p92['V925701'].astype(str)

# Check uniqueness
print(f"\nIndividual study unique keys: {df['match_key'].nunique()} / {len(df)}")
print(f"Panel unique keys: {p92['match_key'].nunique()} / {len(p92)}")

# Since keys aren't unique, direct matching won't work.
# But we CAN use the party-of-candidate information from the individual study
# to create a proper House vote variable for ALL respondents (including panel).

# Actually, the individual study has 2485 respondents, same as the CDF for 1992.
# The CDF panel respondents (VCF0006a < 19920000) number 1359.
# Among these, some have valid V925608/V925610.

# Let me just check: for the full 2485 respondents, how many have valid
# party-of-voted-candidate?
valid_vote = df['vote_party'].isin([1, 2])
valid_pid = df['V923634'].isin([0,1,2,3,4,5,6])
print(f"\nValid vote_party: {valid_vote.sum()}")
print(f"Valid PID: {valid_pid.sum()}")
print(f"Both valid: {(valid_vote & valid_pid).sum()}")

# Cross-tab PID with vote_party
df['pid_std'] = 8 - df['V923634']  # Reverse: 0=StrongRep→8, ..., 6=StrongDem→2
# Wait, if V923634 0-6 maps to pid_current 1-7 via +1,
# and pid_current in 1992 is: 1=StrongRep...7=StrongDem (from Table 4 analysis)
# Then V923634=0 → pid=1=StrongRep → standard PID=7=StrongRep → reversed!
# So V923634: 0=StrongRep, ..., 6=StrongDem
# Standard CDF: 1=StrongDem, 7=StrongRep
# Mapping: standard = 8 - (V923634 + 1) = 7 - V923634
df['pid_standard'] = 7 - df['V923634']
# Check: V923634=0 → pid_standard=7=StrongRep. V923634=6 → pid_standard=1=StrongDem.

valid_df = df[(df['vote_party'].isin([1,2])) & (df['V923634'].isin([0,1,2,3,4,5,6]))].copy()
valid_df['house_rep'] = (valid_df['vote_party'] == 2).astype(int)
print("\nCrosstab PID_standard vs house_rep:")
print(pd.crosstab(valid_df['pid_standard'], valid_df['house_rep']))

# This should show: pid_standard=1 (StrongDem) → mostly house_rep=0 (Dem)
# pid_standard=7 (StrongRep) → mostly house_rep=1 (Rep)

# Now run probit with this data
valid_df['strong'] = np.where(valid_df['pid_standard']==7, 1, np.where(valid_df['pid_standard']==1, -1, 0))
valid_df['weak'] = np.where(valid_df['pid_standard']==6, 1, np.where(valid_df['pid_standard']==2, -1, 0))
valid_df['lean'] = np.where(valid_df['pid_standard']==5, 1, np.where(valid_df['pid_standard']==3, -1, 0))

X = sm.add_constant(valid_df[['strong','weak','lean']])
mod = Probit(valid_df['house_rep'], X).fit(disp=0)
print(f"\nProbit with V925608/V925610 party coding (full 1992 sample):")
print(f"N={len(valid_df)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}")
print(f"Strong={mod.params['strong']:.3f}({mod.bse['strong']:.3f})")
print(f"Weak={mod.params['weak']:.3f}({mod.bse['weak']:.3f})")
print(f"Lean={mod.params['lean']:.3f}({mod.bse['lean']:.3f})")
print(f"Int={mod.params['const']:.3f}({mod.bse['const']:.3f})")
print(f"Target: N=760, LL=-408.2, R2=0.20, Strong=0.975, Weak=0.627, Lean=0.472, Int=-0.211")

# Also compare with CDF VCF0707 for the same respondents
print("\n=== Compare vote_party with CDF VCF0707 ===")
# For the full sample, VCF0707 in CDF gives:
# Dem=812, Rep=558 for all 2485 respondents
# Our vote_party gives different numbers
print(f"vote_party=1 (Dem): {(df['vote_party']==1).sum()}")
print(f"vote_party=2 (Rep): {(df['vote_party']==2).sum()}")
print("CDF VCF0707: Dem=812, Rep=558")
