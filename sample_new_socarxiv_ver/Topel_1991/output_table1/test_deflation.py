import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

deflator = {
    1977: 66.7, 1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
    1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8
}

mask = (df['SEX'] == 1) & (df['AGE'] >= 20) & (df['AGE'] <= 60) & \
       df['DWREAS'].isin([1, 2, 3, 4, 5, 6]) & \
       df['EMPSTAT'].isin([10, 12]) & \
       (df['DWYEARS'] < 99) & (df['DWFULLTIME'] == 2) & \
       (df['DWWEEKL'] > 0) & (df['DWWEEKL'] < 9000) & \
       (df['DWWEEKC'] > 0) & (df['DWWEEKC'] < 9000) & \
       (df['DWLASTWRK'] < 99)
sample = df[mask].copy()
sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']
sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1, 5, 10, 20, 100],
                              labels=['0-5', '6-10', '11-20', '21+'])

# Test different DWWEEKL wage year assumptions
for shift_label, shift in [('disp_yr-1', -1), ('disp_yr', 0), ('disp_yr+1', 1)]:
    ws = sample.copy()
    ws['wage_year'] = ws['disp_year'] + shift
    ws['def_cur'] = ws['YEAR'].map(deflator)
    ws['def_pri'] = ws['wage_year'].map(deflator)
    ws = ws.dropna(subset=['def_cur', 'def_pri'])
    ws['lwc'] = np.log(ws['DWWEEKC'] / ws['def_cur']) - np.log(ws['DWWEEKL'] / ws['def_pri'])

    total = ws['lwc'].mean()
    bins_list = ['0-5', '6-10', '11-20', '21+']
    vals = [ws[ws['tenure_bin'] == b]['lwc'].mean() for b in bins_list]
    print(f"{shift_label}: {vals[0]:.3f} {vals[1]:.3f} {vals[2]:.3f} {vals[3]:.3f} Total={total:.3f}")

print("\nPaper:  -.095  -.223  -.282  -.439  Total=-.135")

# Also test: what if DWLASTWRK refers to the number of full years ago,
# not calendar years? E.g., DWLASTWRK=1 from Jan 1984 means displaced in
# calendar year 1983, but wages were from 1982 (last full year of work)
# In this case: wage_year = YEAR - DWLASTWRK - 1
print("\n\nAlternative: wage_year = YEAR - DWLASTWRK - 1")
ws = sample.copy()
ws['wage_year'] = ws['YEAR'] - ws['DWLASTWRK'] - 1
ws['def_cur'] = ws['YEAR'].map(deflator)
ws['def_pri'] = ws['wage_year'].map(deflator)
ws = ws.dropna(subset=['def_cur', 'def_pri'])
ws['lwc'] = np.log(ws['DWWEEKC'] / ws['def_cur']) - np.log(ws['DWWEEKL'] / ws['def_pri'])
total = ws['lwc'].mean()
vals = [ws[ws['tenure_bin'] == b]['lwc'].mean() for b in bins_list]
print(f"Result: {vals[0]:.3f} {vals[1]:.3f} {vals[2]:.3f} {vals[3]:.3f} Total={total:.3f}")

# What about: for DWLASTWRK=0 (this year), use survey year for both
# For DWLASTWRK=1, use disp_year for prior earnings
# But for DWLASTWRK=0, there's no inflation adjustment
# This is already what standard approach does (disp_year = YEAR, so deflators cancel)

# What if we use the MID-YEAR deflator?
# The survey is in January. DWWEEKL from 1983 would be mid-1983 level.
# DWWEEKC is January 1984 level. Using annual averages might be inaccurate.

# Let me try: what if we deflate DWWEEKC by the PREVIOUS year's deflator
# (since current earnings might refer to previous year's income)
print("\n\nAlternative: deflate current by YEAR-1")
ws = sample.copy()
ws['def_cur'] = (ws['YEAR'] - 1).map(deflator)
ws['def_pri'] = ws['disp_year'].map(deflator)
ws = ws.dropna(subset=['def_cur', 'def_pri'])
ws['lwc'] = np.log(ws['DWWEEKC'] / ws['def_cur']) - np.log(ws['DWWEEKL'] / ws['def_pri'])
total = ws['lwc'].mean()
vals = [ws[ws['tenure_bin'] == b]['lwc'].mean() for b in bins_list]
print(f"Result: {vals[0]:.3f} {vals[1]:.3f} {vals[2]:.3f} {vals[3]:.3f} Total={total:.3f}")

# What about: deflate current to YEAR, prior to disp_year, but using
# a different deflator series?
# Try: PCE chain-type price index from FRED (Q4 values for better timing)
# Q4 values would be closer to January of next year
pce_q4 = {
    1977: 67.2, 1978: 73.7, 1979: 80.3, 1980: 87.6, 1981: 95.3,
    1982: 100.7, 1983: 104.9, 1984: 108.7, 1985: 111.8, 1986: 114.0
}
print("\n\nPCE Q4 deflators:")
ws = sample.copy()
ws['def_cur'] = ws['YEAR'].map(pce_q4)
ws['def_pri'] = ws['disp_year'].map(pce_q4)
ws = ws.dropna(subset=['def_cur', 'def_pri'])
ws['lwc'] = np.log(ws['DWWEEKC'] / ws['def_cur']) - np.log(ws['DWWEEKL'] / ws['def_pri'])
total = ws['lwc'].mean()
vals = [ws[ws['tenure_bin'] == b]['lwc'].mean() for b in bins_list]
print(f"Result: {vals[0]:.3f} {vals[1]:.3f} {vals[2]:.3f} {vals[3]:.3f} Total={total:.3f}")

# Try: what if the paper deflates to a COMMON year (like 1984) using the
# survey-year adjustment only? I.e., for 1986 workers, deflate to 1984:
# 1986 worker: log(DWWEEKC * defl[1984]/defl[1986]) - log(DWWEEKL * defl[1984]/defl[disp])
# = log(DWWEEKC/DWWEEKL) + log(defl[disp]/defl[1986])
# This is different from deflating to 1982!
# Wait no: log(a/P_a) - log(b/P_b) = log(a/b) + log(P_b/P_a)
# The base year cancels out. So deflating to 1984 vs 1982 gives the same answer.
# This is definitely not the issue.

# MAYBE: the issue is that the paper uses a different earnings concept.
# DWWEEKL is "usual weekly earnings" which might be hourly rate * usual hours
# Perhaps the paper adjusts for hours worked?
print("\n\nUHRSWORK1 for sample:")
print(sample['UHRSWORK1'].describe())
