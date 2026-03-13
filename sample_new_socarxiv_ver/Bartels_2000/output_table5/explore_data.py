import pandas as pd
import numpy as np

# Explore 1992 panel data matching to CDF
p92 = pd.read_csv('panel_1992.csv')
print("=== Panel 1992 ===")
print("V923634 - stats:", p92['V923634'].min(), p92['V923634'].max(), p92['V923634'].nunique())
print("V900320 - stats:", p92['V900320'].min(), p92['V900320'].max(), p92['V900320'].nunique())

# Try to find these respondents in the CDF
cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
cdf92 = cdf[cdf['VCF0004']==1992].copy()
print("\n=== CDF 1992 ===")
print("N:", len(cdf92))
print("VCF0006a range:", cdf92['VCF0006a'].min(), cdf92['VCF0006a'].max(), "nunique:", cdf92['VCF0006a'].nunique())
print("VCF0006 range:", cdf92['VCF0006'].min(), cdf92['VCF0006'].max(), "nunique:", cdf92['VCF0006'].nunique())

# V923634 in panel_1992 is likely a PID variable (values 0-9)
# It looks like current PID from the 1992 study
# pid_current matches V923634?
print("\n=== Checking V923634 vs pid_current ===")
print("V923634 value counts:")
print(p92['V923634'].value_counts().sort_index())
print("\npid_current value counts:")
print(p92['pid_current'].value_counts().sort_index())

# V900320 is likely from the 1990 wave
print("\n=== V900320 vs pid_lagged ===")
print("V900320 value counts:")
print(p92['V900320'].value_counts().sort_index())
print("\npid_lagged value counts:")
print(p92['pid_lagged'].value_counts().sort_index())

# Try matching panel respondents to CDF by pid_current value
# Actually, the CDF uses VCF0006a as the respondent ID (case number)
# V923634 looks like a recoded PID, not an ID

# Let me check: can we use the CDF for 1992 panel respondents?
# CDF has VCF0707 coded correctly (1=Dem, 2=Rep)
# But we need to identify which CDF respondents are panel respondents

# Check VCF0707 cross-tabbed with VCF0301 in the CDF
print("\n=== CDF 1992: VCF0707 vs VCF0301 ===")
print(pd.crosstab(cdf92['VCF0301'], cdf92['VCF0707']))

# Check if panel_1992 pid_current maps to V923634 differently
# Actually V923634 has values 0-9, likely ANES 7pt PID with missing codes
# 0=NA/missing, 1-7=PID, 8=DK, 9=NA
# Same for V900320 from 1990 wave

# So pid_current was derived from V923634 (mapping 1-7 directly)
# And pid_lagged was derived from V900320

# For the House vote, we need a proper party-of-candidate variable
# V925609 looks like it could be party of the district's candidates
# Let me check what V925609 represents more carefully
print("\n=== V925609 value distribution ===")
print(p92['V925609'].value_counts().sort_index())

# Maybe we need to determine party of each candidate
# V925701: 1=candidate 1, 5=candidate 2
# What if the party of the voted-for candidate depends on the district?
# We don't have a direct party-of-voted-candidate variable

# Alternative: use the CDF VCF0707 for 1992 panel members
# We need to match on some ID
# The CDF might contain the panel respondents identified by case number

# Check if VCF0006a in CDF matches some pattern in panel data
print("\nCDF VCF0006a first 20:", cdf92['VCF0006a'].head(20).tolist())
print("\nPanel N=", len(p92))
print("CDF 1992 N=", len(cdf92))

# The 1992 ANES timeseries had 2485 respondents
# The panel had ~1336 respondents who also appeared in 1990
# So panel respondents are a subset of CDF 1992 respondents

# We need to figure out how to match them
# Perhaps we can match on pid_current (from panel) = VCF0301 (from CDF)
# and V925701 (from panel) = some CDF variable

# Actually, a simpler approach: just use the CDF for ALL 1992 analysis
# We need:
# - Current PID (1992): VCF0301 from CDF
# - Lagged PID (1990): from panel_1992.csv pid_lagged
# - House vote: VCF0707 from CDF
# The issue is matching CDF rows to panel rows

# Since panel_1992 has 1336 rows and the CDF has 2485 for 1992,
# we might be able to assume they're in the same order for the first 1336?
# That's unlikely to be correct.

# Actually maybe we should look at whether the CDF panel variable exists
print("\n=== CDF panel-related variables ===")
# VCF0013 might indicate panel membership
if 'VCF0013' in cdf92.columns:
    print("VCF0013:", cdf92['VCF0013'].value_counts().sort_index())
