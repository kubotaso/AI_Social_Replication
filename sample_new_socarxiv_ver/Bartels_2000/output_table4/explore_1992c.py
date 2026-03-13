import pandas as pd
df = pd.read_csv('panel_1992.csv')

# Look at all columns for vote info
print("V925609 value counts (vote):")
print(df['V925609'].value_counts().sort_index())
print()

print("V925701 value counts:")
print(df['V925701'].value_counts().sort_index())
print()

# Cross-tab V925609 vs V925701
ct = pd.crosstab(df['V925609'], df['V925701'])
print("Crosstab V925609 vs V925701:")
print(ct)
print()

# Cross-tab pid_current=1 with V925609
mask_valid = df['pid_current'].isin([1,2,3,4,5,6,7]) & df['vote_pres'].isin([1,2])
df_v = df[mask_valid].copy()
print(f"Valid N: {len(df_v)}")

# Check V923634 values
print("\nV923634 coding (raw PID from 1992 study):")
print("0 = Strong Republican")
print("1 = Not very strong Republican")
print("2 = Independent, close to Republican")
print("3 = Independent (neither/other)")
print("4 = Independent, close to Democrat")
print("5 = Not very strong Democrat")
print("6 = Strong Democrat")
print()

# So pid_current = V923634 + 1:
# 1 = Strong Republican
# 2 = Not very strong Republican (Weak Rep)
# 3 = Independent-Republican (Lean Rep)
# 4 = Independent
# 5 = Independent-Democrat (Lean Dem)
# 6 = Not very strong Democrat (Weak Dem)
# 7 = Strong Democrat

# With reversed PID (8-pid):
# 1->7=StrongRep, 7->1=StrongDem
# This is STANDARD VCF0301 coding now

# Vote: V925609=1->vote_pres=1 (Bush), V925609=2->vote_pres=2 (Clinton)
# Bush is Republican. So vote_pres=1 = Republican, vote_pres=2 = Democrat?
# NO: Looking at the cross-tab from explore_1992.py:
#   pid_current=1 (Strong Rep) -> vote_pres=2 (191) >> vote_pres=1 (6)
# This means Strong Reps vote for vote_pres=2, so vote_pres=2 = Republican!

# BUT: V925609=1 maps to vote_pres=1, and V925609=2 maps to vote_pres=2
# In ANES 1992:
# V925609 = 0: Didn't vote / registration form not completed
# V925609 = 1: Bill Clinton
# V925609 = 2: George Bush
# V925609 = 3: Ross Perot
# V925609 = 7: Other
# V925609 = 9: DK

# So V925609=1=Clinton(Dem), V925609=2=Bush(Rep)
# vote_pres=1 maps to V925609=1=Clinton=Dem
# vote_pres=2 maps to V925609=2=Bush=Rep

# So vote_pres=1=Dem, vote_pres=2=Rep IS CORRECT after all.
# And vote=(vote_pres==2)=Rep is correct.

# But then why is the intercept positive (0.080) when it should be negative (-0.073)?

# Let's check: Of the 725 valid cases, how many are Rep voters?
print(f"\nvote_pres=1 (Dem/Clinton): {(df_v['vote_pres']==1).sum()}")
print(f"vote_pres=2 (Rep/Bush): {(df_v['vote_pres']==2).sum()}")
print(f"Rep share: {(df_v['vote_pres']==2).sum()/len(df_v):.3f}")

# In Bartels' data (N=729), maybe the Rep share is slightly different
# Ground truth intercept = -0.073 -> base rate < 50% after controlling for PID
# Our intercept = 0.080 -> base rate > 50% after controlling for PID

# This is about the intercept in the PROBIT with PID controls, not the raw share.
# The intercept reflects the mean tendency after accounting for PID.
# With PID dummies that are directional (+1/-1), the intercept represents
# the expected probit index for someone with all PID dummies = 0 (pure independent).

# Check PID distribution
for i in range(1, 8):
    n_i = (df_v['pid_current'] == i).sum()
    rep_i = ((df_v['pid_current'] == i) & (df_v['vote_pres'] == 2)).sum()
    print(f"pid_current={i}: N={n_i}, Rep voters={rep_i}")
