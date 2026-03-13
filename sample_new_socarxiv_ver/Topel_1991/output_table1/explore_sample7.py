import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# With DWREAS 1-6, plant_closing = 0.389, matching paper's .390
# Now need to find the right N = 4367

# DWREAS 1-6 gives N=6738. We need filters to bring it down.

# Key insight: the paper says "wages rise with job seniority" - they study
# wage changes, so maybe the sample is restricted to those with VALID wage data

# Base: DWREAS 1-6, male, 20-60, employed, valid tenure
m_all = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
        (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & \
        (df['DWYEARS']<99)
s = df[m_all]

# Check valid wages
vw = (s['DWWEEKL']>0) & (s['DWWEEKL']<9000) & \
     (s['DWWEEKC']>0) & (s['DWWEEKC']<9000)
print(f"DWREAS 1-6, valid wages: N = {vw.sum()}")

# Full-time + valid wages
ft = s['DWFULLTIME'] == 2
print(f"DWREAS 1-6, full-time + valid wages: N = {(ft & vw).sum()}")

# DWCLASS private + valid wages
priv = s['DWCLASS'].isin([1,2])
print(f"DWREAS 1-6, DWCLASS private + valid wages: N = {(priv & vw).sum()}")

# Valid DWLASTWRK + valid wages
lw = s['DWLASTWRK'] < 99
print(f"DWREAS 1-6, DWLASTWRK<99 + valid wages: N = {(lw & vw).sum()}")

# Full-time
print(f"DWREAS 1-6, full-time: N = {ft.sum()}")

# DWYEARS >= 1
y1 = s['DWYEARS'] >= 1
print(f"DWREAS 1-6, DWYEARS>=1: N = {y1.sum()}")

# Maybe: full-time + DWYEARS >= 3 (paper focuses on 3+ years tenure in later analysis)
y3 = s['DWYEARS'] >= 3
print(f"DWREAS 1-6, full-time + DWYEARS>=3: N = {(ft & y3).sum()}")

# Let me try many combinations systematically
print("\n=== Systematic combinations ===")
filters = {
    'ft': ft,
    'priv': priv,
    'y1': y1,
    'lw': lw,
    'y3': y3,
    'vw': vw,
}

import itertools
for r in range(1, 4):
    for combo in itertools.combinations(filters.keys(), r):
        mask = pd.Series(True, index=s.index)
        for f in combo:
            mask &= filters[f]
        n = mask.sum()
        if 4300 <= n <= 4500:
            # Check plant closing
            sub = s[mask]
            pc = (sub['DWREAS']==1).mean()
            label = ' + '.join(combo)
            print(f"  {label}: N = {n}, plant_closing = {pc:.3f}")

# Since no simple combination gives N=4367 with DWREAS 1-6,
# maybe the original paper uses different codes (not IPUMS harmonized)
# Or maybe the DWS from the original CPS tape has slightly different
# universe definitions

# Actually, let me check: the paper uses data from 1984 and 1986 DWS
# In the original 1984 CPS supplement, DWREAS might have different codes
# than in the 1986 supplement. IPUMS harmonizes them.

# Let me try: DWREAS 1-3 (standard economic displacement) but check
# if the plant closing matches better by bin
# The discrepancy is systematic: .460 vs .390 for total
# That's about .070 too high
# This could mean we're counting some non-plant-closing as plant-closing
# Or missing observations that are NOT plant closings

# Actually, wait. Let me reconsider the denominator.
# If sample includes DWREAS 1-6, the denominator is larger (more people)
# and many of the added people (DWREAS 4,5,6) are NOT plant closings
# So the percentage drops

# With DWREAS 1-3 only: 2619 plant closings / 5691 = 0.460
# With DWREAS 1-6: 2619 plant closings / 6738 = 0.389

# The plant closing numerator is the SAME (2619 from DWREAS==1)
# Only the denominator changes

# So if the paper's .390 matches DWREAS 1-6 denominator,
# that means the paper includes ALL displaced workers (1-6), not just economic reasons!

# This makes sense if we read the paper more carefully:
# The DWS asks about workers who "lost or left" a job in the past 5 years
# The paper may include everyone who was displaced, regardless of reason
# "Displaced for economic reasons" might be the paper's description of the survey,
# not an actual filter

# Now: how to get N = 4367 from DWREAS 1-6 with N = 6738?
# Need to remove about 2371 observations
# That's 35% of the sample

# Key filters to try:
# - Full-time on lost job (DWFULLTIME==2): removes ~371 -> 6367
# - DWYEARS >= 1: removes ~1355 -> 5383
# - Both: 5173

# These are still too many. Let me think about what else...

# Maybe the paper restricts to workers with 3+ years tenure on lost job?
# Paper says later: "I estimate the returns to seniority... among a sample
# of workers with at least 3 years of tenure on the current job"
# But Table 1 shows "0-5 years" bin, so tenure >= 0

# Maybe employed = currently working (not just "has a job")
# EMPSTAT 10 = At work only (not 12 = Has job, not at work)
m_10 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
       (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT']==10) & (df['DWYEARS']<99)
print(f"\nEMPSTAT==10 only: N = {len(df[m_10])}")
print(f"  + full-time: N = {len(df[m_10 & (df['DWFULLTIME']==2)])}")
print(f"  + DWYEARS>=1: N = {len(df[m_10 & (df['DWYEARS']>=1)])}")
print(f"  + ft + y1: N = {len(df[m_10 & (df['DWFULLTIME']==2) & (df['DWYEARS']>=1)])}")

# Check UHRSWORK1 (hours worked last week)
# Maybe currently working full-time (35+ hours)
hrs = s['UHRSWORK1'] >= 35
print(f"\nUHRSWORK1 >= 35: N = {hrs.sum()}")
print(f"+ ft: N = {(hrs & ft).sum()}")
print(f"+ y1: N = {(hrs & y1).sum()}")
print(f"+ ft + y1: N = {(hrs & ft & y1).sum()}")

# Check current full-time employment
# Actually UHRSWORK1 == 999 might mean NIU
hrs2 = (s['UHRSWORK1'] >= 35) & (s['UHRSWORK1'] < 999)
print(f"\nUHRSWORK1 35-998: N = {hrs2.sum()}")
print(f"+ ft: N = {(hrs2 & ft).sum()}")

# Maybe the key is that the paper's sample also has valid earnings data
# N=4367 might be those with valid BOTH prior and current earnings
# Let me check if that's close
vw_both = vw & (s['DWLASTWRK'] < 99)
print(f"\nValid wages + DWLASTWRK<99: N = {vw_both.sum()}")
print(f"+ ft: N = {(vw_both & ft).sum()}")
print(f"+ y1: N = {(vw_both & y1).sum()}")
print(f"+ ft + y1: N = {(vw_both & ft & y1).sum()}")

# Maybe valid DWWEEKL OR DWWEEKC (not both required)
vw_either = ((s['DWWEEKL']>0) & (s['DWWEEKL']<9000)) | \
            ((s['DWWEEKC']>0) & (s['DWWEEKC']<9000))
print(f"\nValid either wage: N = {vw_either.sum()}")

# Only valid DWWEEKL
vwl = (s['DWWEEKL']>0) & (s['DWWEEKL']<9000)
print(f"Valid DWWEEKL only: N = {vwl.sum()}")
print(f"+ ft: N = {(vwl & ft).sum()}")
print(f"+ y1: N = {(vwl & y1).sum()}")

# Only valid DWWEEKC
vwc = (s['DWWEEKC']>0) & (s['DWWEEKC']<9000)
print(f"Valid DWWEEKC only: N = {vwc.sum()}")
print(f"+ ft: N = {(vwc & ft).sum()}")
