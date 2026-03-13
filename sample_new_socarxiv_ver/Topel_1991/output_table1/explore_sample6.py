import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# With ALL displacement reasons (1-6), plant closing = 0.389, matching paper's 0.390
# Now find the right N

# DWREAS 1-6, excluding 98 (DK/Ref)
m_all = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
        (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & \
        (df['DWYEARS']<99)
s = df[m_all]
print(f"DWREAS 1-6, all: N = {len(s)}")

# With full-time
sft = s[s['DWFULLTIME']==2]
print(f"+ full-time: N = {len(sft)}")

# With DWYEARS >= 1
sy1 = s[s['DWYEARS']>=1]
print(f"+ DWYEARS>=1: N = {len(sy1)}")

# Full-time + DWYEARS>=1
sft_y1 = sft[sft['DWYEARS']>=1]
print(f"+ full-time + DWYEARS>=1: N = {len(sft_y1)}")

# DWCLASS private + various
m_priv = m_all & (df['DWCLASS'].isin([1,2]))
print(f"+ DWCLASS private: N = {len(df[m_priv])}")

m_priv_ft = m_priv & (df['DWFULLTIME']==2)
print(f"+ DWCLASS private + full-time: N = {len(df[m_priv_ft])}")

m_priv_y1 = m_priv & (df['DWYEARS']>=1)
print(f"+ DWCLASS private + DWYEARS>=1: N = {len(df[m_priv_y1])}")

m_priv_ft_y1 = m_priv_ft & (df['DWYEARS']>=1)
print(f"+ DWCLASS private + full-time + DWYEARS>=1: N = {len(df[m_priv_ft_y1])}")

# excluding DWREAS==98 only
m_no98 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & \
         (df['DWYEARS']<99) & (df['DWREAS']!=98)
print(f"\nDWREAS 1-6 (excl 98): N = {len(df[m_no98])}")

# Try including DWREAS 98
m_inc98 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
          (df['DWREAS'].isin([1,2,3,4,5,6,98])) & (df['EMPSTAT'].isin([10,12])) & \
          (df['DWYEARS']<99)
print(f"DWREAS 1-6,98: N = {len(df[m_inc98])}")

# DWLASTWRK filter
print(f"\nDWLASTWRK distribution for DWREAS 1-6 sample:")
print(s['DWLASTWRK'].value_counts().sort_index())

# With DWLASTWRK <= 5
s_lw5 = s[s['DWLASTWRK']<=5]
print(f"+ DWLASTWRK<=5: N = {len(s_lw5)}")

# Try: DWREAS 1-6, DWLASTWRK<=5, to get ~4367
s_lw5_ft = s_lw5[s_lw5['DWFULLTIME']==2]
print(f"+ DWLASTWRK<=5 + full-time: N = {len(s_lw5_ft)}")

# The paper surveys 1984 and 1986 about past 5 years
# But in IPUMS, DWLASTWRK is probably already restricted to 5 years
# Let me check DWYEARS >= 1 exclusion

# Actually: N = 6738, we need ~4367
# 6738 / 4367 = 1.54. That's a big gap.
# Maybe the paper excludes DWREAS 4,5,6 after all but has different DWREAS coding
# Or maybe DWREAS in original CPS is different from IPUMS

# Let me reconsider: the paper says the sample is men who were
# "displaced from a job in the past 5 years, for economic reasons"
# and "were employed at the time of the survey"
# "Economic reasons" is standard terminology meaning layoffs and plant closings

# Actually, in the original CPS DWS supplement:
# Q: Why did you leave that job?
# 1 = Plant closed down or moved
# 2 = Slack work or business conditions
# 3 = Position/shift abolished
# This is the ORIGINAL coding. IPUMS harmonized it.

# In IPUMS DWS:
# 1 = Plant or company closed down or moved
# 2 = Insufficient work (= "slack work or business conditions" in original)
# 3 = Position or shift abolished
# 4 = Slack work or business conditions (DIFFERENT YEAR coding?)
# 5 = Seasonal job completed
# 6 = Other

# Maybe codes 4,5,6 are from different survey years or have different meanings
# For 1984 and 1986 DWS, all displacements are likely coded 1,2,3
# The 4,5,6 may be from other years in the IPUMS extract

# Let me check by year
print(f"\nDWREAS by YEAR:")
for y in [1984, 1986]:
    sub = df[(df['YEAR']==y) & (df['DWREAS'].between(1,6))]
    print(f"\nYEAR {y}:")
    print(sub['DWREAS'].value_counts().sort_index())

# Hmm wait - if IPUMS includes more reason codes in certain years,
# that could explain the discrepancy. Let me see what's in the codebook
