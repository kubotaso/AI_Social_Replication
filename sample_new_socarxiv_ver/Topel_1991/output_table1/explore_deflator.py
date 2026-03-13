import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# Best sample: DWREAS 1-6, full-time, valid wages, valid DWLASTWRK
m = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
    (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & \
    (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & \
    (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
    (df['DWWEEKC']>0) & (df['DWWEEKC']<9000) & \
    (df['DWLASTWRK']<99)
sample = df[m].copy()
sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']

# Our current values are about 0.03-0.07 too negative
# The paper says "GNP price deflator for consumption expenditure"
# This is the "PCE deflator" or "implicit price deflator for PCE"

# Different sources give slightly different values
# Let me try several deflator series

# Version 1: Original values
d1 = {1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
      1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8}

# Version 2: BEA NIPA Table 1.1.9 (Implicit Price Deflators for GDP)
# GDP deflator values, not PCE
d2 = {1978: 72.22, 1979: 78.55, 1980: 85.73, 1981: 94.02,
      1982: 100.00, 1983: 103.88, 1984: 107.66, 1985: 110.88, 1986: 113.33}

# Version 3: NIPA Table 2.3.4 (PCE Price Index)
# These are chain-weighted index numbers
d3 = {1978: 63.3, 1979: 68.6, 1980: 74.9, 1981: 81.8,
      1982: 86.7, 1983: 90.0, 1984: 93.6, 1985: 96.6, 1986: 98.6}

# Version 4: GNP (not GDP) implicit price deflator
# Pre-1991, GNP deflator was more commonly used
d4 = {1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 93.7,
      1982: 100.0, 1983: 103.2, 1984: 107.3, 1985: 110.2, 1986: 113.0}

# Version 5: No deflation (nominal)
d5 = {y: 100.0 for y in range(1978, 1987)}

# Version 6: Try deflating current wage to prior year
# Instead of deflating both to same base

# Version 7: Maybe the paper deflates to a common year
# but does it differently. Let me compute without deflation first
# to see the raw nominal change

for name, defl in [("V1 Original", d1), ("V2 GDP deflator", d2),
                    ("V3 PCE chain index", d3), ("V4 GNP deflator", d4),
                    ("V5 Nominal", d5)]:
    ws = sample.copy()
    ws['def_cur'] = ws['YEAR'].map(defl)
    ws['def_pri'] = ws['disp_year'].map(defl)
    ws = ws.dropna(subset=['def_cur', 'def_pri'])
    ws['log_wc'] = np.log(ws['DWWEEKC']/ws['def_cur']) - np.log(ws['DWWEEKL']/ws['def_pri'])

    total_mean = ws['log_wc'].mean()
    total_se = ws['log_wc'].std() / np.sqrt(len(ws))
    print(f"\n{name}: Total mean = {total_mean:.3f} ({total_se:.3f})")
    print(f"  Paper: -.135 (.009)")

    # By bin
    ws['tenure_bin'] = pd.cut(ws['DWYEARS'], bins=[-0.1, 5, 10, 20, 100], labels=['0-5', '6-10', '11-20', '21+'])
    for b in ['0-5', '6-10', '11-20', '21+']:
        data = ws[ws['tenure_bin']==b]
        m_val = data['log_wc'].mean()
        print(f"  {b}: {m_val:.3f}")

# The nominal version gives -.069, way too high (not negative enough)
# The deflated versions give about -.170, too negative
# The paper says -.135, which is between nominal and our deflated

# Maybe the deflation is done differently:
# Perhaps the paper uses calendar year deflators but adjusts for the fact
# that both earnings are in dollars of different years

# Let me think: if worker was displaced in 1982 (DWLASTWRK=2 in 1984 survey)
# Prior earnings (DWWEEKL) are in 1982 dollars
# Current earnings (DWWEEKC) are in 1984 dollars
# Real change = log(DWWEEKC/deflator[1984]) - log(DWWEEKL/deflator[1982])
# = log(DWWEEKC) - log(DWWEEKL) - log(deflator[1984]) + log(deflator[1982])
# = nominal change - (log(deflator[1984]) - log(deflator[1982]))
# = nominal change - inflation adjustment

# The inflation adjustment makes things MORE negative (prices went up)
# Nominal is -.069, deflated is -.171
# The paper says -.135, halfway between

# What if the paper uses deflation year differently?
# DWLASTWRK: years ago last worked at that job
# Maybe displacement happened BEFORE they left that job
# Or maybe the earnings refer to different points in time

# What if DWWEEKL earnings are from the year BEFORE displacement?
# Or what if they're already in survey-year dollars?

# Try: don't deflate DWWEEKL (assume it's in survey-year dollars)
print("\n\n=== Alternative: DWWEEKL already adjusted? ===")
ws2 = sample.copy()
ws2['log_wc_no_adj'] = np.log(ws2['DWWEEKC']) - np.log(ws2['DWWEEKL'])
total_nom = ws2['log_wc_no_adj'].mean()
print(f"Nominal: {total_nom:.3f}")

# Maybe the paper deflates both to the SURVEY year
# That would be: real_prior = DWWEEKL * deflator[YEAR] / deflator[disp_year]
# So: log_wc = log(DWWEEKC) - log(DWWEEKL * deflator[YEAR] / deflator[disp_year])
# = log(DWWEEKC/DWWEEKL) - log(deflator[YEAR]/deflator[disp_year])
# This is the SAME as what we computed. So direction is right.

# Maybe the issue is not the deflation but the sample
# Let me try: include DWYEARS==0 (0 years tenure) in the sample
m2 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
     (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & \
     (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & \
     (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
     (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
s2 = df[m2].copy()
print(f"\n\n=== Without DWLASTWRK filter ===")
print(f"N = {len(s2)}")

# For those without valid DWLASTWRK, what do we do about deflation?
# DWLASTWRK == 99: NIU, can't determine displacement year
# There are only 7 such cases
s2['disp_year'] = s2['YEAR'] - s2['DWLASTWRK']
s2_valid = s2[s2['DWLASTWRK'] < 99].copy()
s2_valid['def_cur'] = s2_valid['YEAR'].map(d1)
s2_valid['def_pri'] = s2_valid['disp_year'].map(d1)
s2_valid = s2_valid.dropna(subset=['def_cur', 'def_pri'])
s2_valid['log_wc'] = np.log(s2_valid['DWWEEKC']/s2_valid['def_cur']) - np.log(s2_valid['DWWEEKL']/s2_valid['def_pri'])
print(f"Valid deflation: N = {len(s2_valid)}, mean log_wc = {s2_valid['log_wc'].mean():.3f}")

# What if some workers were displaced SAME year as survey (DWLASTWRK=0)?
# Then disp_year = YEAR, and deflation cancels out: just nominal change
# Let me handle DWLASTWRK=0 differently
s3 = sample.copy()
# For DWLASTWRK=0, displacement year = survey year, so no deflation needed
s3.loc[s3['DWLASTWRK']==0, 'disp_year'] = s3.loc[s3['DWLASTWRK']==0, 'YEAR']
print(f"\nDWLASTWRK==0 count: {(s3['DWLASTWRK']==0).sum()}")

# Actually, maybe the deflator values I'm using are wrong
# The paper was written around 1988-1990, before BEA revised NIPA extensively
# Vintage data from that era may have slightly different values

# Let me try adjusting the deflator to see what gets the right answer
# Target: total mean log_wc = -0.135
# Our nominal = -0.069
# Difference = -0.066
# So the deflation adjustment should be -0.066 on average

# Compute the average inflation adjustment we're currently applying
adj = -(np.log(sample['YEAR'].map(d1)) - np.log(sample['disp_year'].map(d1)))
sample_adj = sample.dropna(subset=['disp_year']).copy()
sample_adj['adj'] = -(np.log(sample_adj['YEAR'].map(d1)) - np.log(sample_adj['disp_year'].map(d1)))
print(f"\nAverage inflation adjustment: {sample_adj['adj'].mean():.4f}")
print(f"We need adjustment: {-0.135 - (-0.069):.4f} = -0.066")
print(f"Our adjustment: {sample_adj['adj'].mean():.4f}")
print(f"Ratio: {(-0.135 - (-0.069)) / sample_adj['adj'].mean():.3f}")

# The ratio tells us we're over-deflating by about 55%
# Maybe we should scale the deflation down, or use a different deflator

# What if we use half the deflation?
ws4 = sample.copy()
ws4['def_cur'] = ws4['YEAR'].map(d1)
ws4['def_pri'] = ws4['disp_year'].map(d1)
ws4 = ws4.dropna(subset=['def_cur', 'def_pri'])
ws4['log_wc_half'] = np.log(ws4['DWWEEKC']/ws4['DWWEEKL']) + 0.5 * (np.log(ws4['def_pri']) - np.log(ws4['def_cur']))
print(f"\nHalf deflation: {ws4['log_wc_half'].mean():.3f}")

# Or CPI-U instead of PCE deflator?
# CPI-U values (base 1982-84 = 100)
cpi_u = {1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
         1982: 96.5, 1983: 99.6, 1984: 103.9, 1985: 107.6, 1986: 109.6}
ws5 = sample.copy()
ws5['def_cur'] = ws5['YEAR'].map(cpi_u)
ws5['def_pri'] = ws5['disp_year'].map(cpi_u)
ws5 = ws5.dropna(subset=['def_cur', 'def_pri'])
ws5['log_wc_cpi'] = np.log(ws5['DWWEEKC']/ws5['def_cur']) - np.log(ws5['DWWEEKL']/ws5['def_pri'])
print(f"CPI-U deflation: {ws5['log_wc_cpi'].mean():.3f}")

# What about GDP implicit price deflator from FRED?
# Also try with different base years
for base_year in [1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986]:
    ws_test = sample.copy()
    d_test = {y: d1[y]/d1[base_year] for y in d1}
    ws_test['def_cur'] = ws_test['YEAR'].map(d_test)
    ws_test['def_pri'] = ws_test['disp_year'].map(d_test)
    ws_test = ws_test.dropna(subset=['def_cur', 'def_pri'])
    ws_test['log_wc'] = np.log(ws_test['DWWEEKC']/ws_test['def_cur']) - np.log(ws_test['DWWEEKL']/ws_test['def_pri'])
    # Note: base year doesn't matter for log differences!
    # log(a/d_base) - log(b/d_base) = log(a/b) + log(d_base/d_base) = log(a/b)
    # Wait no: log(DWWEEKC/d[YEAR]) - log(DWWEEKL/d[disp]) = log(DWWEEKC/DWWEEKL) - (log(d[YEAR]) - log(d[disp]))
    # The base cancels out.
    break  # Just verify this once

# So the base year doesn't matter. The issue must be the deflator VALUES.
# Let me compute what deflator ratio would give the right answer

# We need: log(DWWEEKC/DWWEEKL) + log(d_prior/d_current) = -0.135
# We have: log(DWWEEKC/DWWEEKL) = -0.069
# So: log(d_prior/d_current) = -0.135 - (-0.069) = -0.066
# Currently: average log(d_prior/d_current) = adj.mean() ≈ -0.102
# We need -0.066 instead of -0.102
# That means we're over-deflating by factor of 0.102/0.066 = 1.55

# What if the paper doesn't deflate DWWEEKL at all?
# If DWWEEKL is reported in "constant dollars" or already adjusted?
# That would give: log(DWWEEKC/d_current) - log(DWWEEKL)
# = log(DWWEEKC/DWWEEKL) - log(d_current)
# This doesn't make sense dimensionally

# What if only DWWEEKC needs deflation (to base year)?
# And DWWEEKL is already in base year dollars?
ws6 = sample.copy()
ws6['def_cur'] = ws6['YEAR'].map(d1)
ws6['log_wc_cur_only'] = np.log(ws6['DWWEEKC']/ws6['def_cur']*100) - np.log(ws6['DWWEEKL'])
print(f"\nDeflate current only: {ws6['log_wc_cur_only'].mean():.3f}")

# What if only DWWEEKL needs adjustment?
ws7 = sample.copy()
ws7['def_pri'] = ws7['disp_year'].map(d1)
ws7['log_wc_pri_only'] = np.log(ws7['DWWEEKC']) - np.log(ws7['DWWEEKL']*d1[1984]/ws7['def_pri'])
print(f"Inflate prior only (to 1984): {ws7.dropna(subset=['def_pri'])['log_wc_pri_only'].mean():.3f}")
