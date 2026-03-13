import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# Best so far: deflate current by YEAR-1 deflator
# This gives Total=-0.141 vs paper's -0.135
# Rationale: survey in January, "current" earnings are from previous calendar year

# But this makes 0-5 and Total slightly too negative, and 21+ too negative.
# Let me try weighted versions of the "deflate current by YEAR-1" approach

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

bins_list = ['0-5', '6-10', '11-20', '21+']

# Approach 1: Deflate current by YEAR-1 (unweighted)
ws = sample.copy()
ws['def_cur'] = (ws['YEAR'] - 1).map(deflator)
ws['def_pri'] = ws['disp_year'].map(deflator)
ws = ws.dropna(subset=['def_cur', 'def_pri'])
ws['lwc'] = np.log(ws['DWWEEKC'] / ws['def_cur']) - np.log(ws['DWWEEKL'] / ws['def_pri'])
print("YEAR-1 unweighted:")
for b in bins_list + ['Total']:
    d = ws if b == 'Total' else ws[ws['tenure_bin'] == b]
    m = d['lwc'].mean()
    se = d['lwc'].std() / np.sqrt(len(d))
    print(f"  {b}: {m:.3f} ({se:.3f})")

# Approach 2: Deflate current by YEAR-1 (weighted)
print("\nYEAR-1 weighted:")
for b in bins_list + ['Total']:
    d = ws if b == 'Total' else ws[ws['tenure_bin'] == b]
    w = d['DWSUPPWT']
    m = np.average(d['lwc'], weights=w)
    v = np.average((d['lwc'] - m)**2, weights=w)
    n_eff = w.sum()**2 / (w**2).sum()
    se = np.sqrt(v / n_eff)
    print(f"  {b}: {m:.3f} ({se:.3f})")

# Approach 3: What if we interpolate between YEAR and YEAR-1?
# The survey is January, so current earnings might be a mix
# Try: deflator for current = average of YEAR and YEAR-1
print("\nAvg(YEAR, YEAR-1) unweighted:")
ws2 = sample.copy()
ws2['def_cur'] = 0.5 * (ws2['YEAR'].map(deflator) + (ws2['YEAR'] - 1).map(deflator))
ws2['def_pri'] = ws2['disp_year'].map(deflator)
ws2 = ws2.dropna(subset=['def_cur', 'def_pri'])
ws2['lwc'] = np.log(ws2['DWWEEKC'] / ws2['def_cur']) - np.log(ws2['DWWEEKL'] / ws2['def_pri'])
for b in bins_list + ['Total']:
    d = ws2 if b == 'Total' else ws2[ws2['tenure_bin'] == b]
    m = d['lwc'].mean()
    se = d['lwc'].std() / np.sqrt(len(d))
    print(f"  {b}: {m:.3f} ({se:.3f})")

# Approach 4: Maybe the answer is weighted + standard deflation
# (weights can shift the mean)
print("\nStandard deflation, weighted:")
ws3 = sample.copy()
ws3['def_cur'] = ws3['YEAR'].map(deflator)
ws3['def_pri'] = ws3['disp_year'].map(deflator)
ws3 = ws3.dropna(subset=['def_cur', 'def_pri'])
ws3['lwc'] = np.log(ws3['DWWEEKC'] / ws3['def_cur']) - np.log(ws3['DWWEEKL'] / ws3['def_pri'])
for b in bins_list + ['Total']:
    d = ws3 if b == 'Total' else ws3[ws3['tenure_bin'] == b]
    w = d['DWSUPPWT']
    m = np.average(d['lwc'], weights=w)
    v = np.average((d['lwc'] - m)**2, weights=w)
    n_eff = w.sum()**2 / (w**2).sum()
    se = np.sqrt(v / n_eff)
    print(f"  {b}: {m:.3f} ({se:.3f})")

# Approach 5: YEAR-1 + disp_year+1 approach
# DWWEEKL from one year later, DWWEEKC from one year earlier
print("\nYEAR-1 current, disp_yr+1 prior:")
ws4 = sample.copy()
ws4['def_cur'] = (ws4['YEAR'] - 1).map(deflator)
ws4['wage_yr'] = ws4['disp_year'] + 1
ws4['def_pri'] = ws4['wage_yr'].map(deflator)
ws4 = ws4.dropna(subset=['def_cur', 'def_pri'])
ws4['lwc'] = np.log(ws4['DWWEEKC'] / ws4['def_cur']) - np.log(ws4['DWWEEKL'] / ws4['def_pri'])
for b in bins_list + ['Total']:
    d = ws4 if b == 'Total' else ws4[ws4['tenure_bin'] == b]
    m = d['lwc'].mean()
    se = d['lwc'].std() / np.sqrt(len(d))
    print(f"  {b}: {m:.3f} ({se:.3f})")

# Compare all approaches to paper
print("\n\nPaper values:")
print("  0-5: -.095 (.010)")
print("  6-10: -.223 (.021)")
print("  11-20: -.282 (.026)")
print("  21+: -.439 (.071)")
print("  Total: -.135 (.009)")

# Approach 6: What if we need to match SEs more carefully?
# The paper SEs are smaller than ours for 21+ (.071 vs our ~.071)
# This suggests similar sample sizes, which is consistent

# Let me check: what N do we get if we DON'T require DWFULLTIME==2?
mask_noft = (df['SEX'] == 1) & (df['AGE'] >= 20) & (df['AGE'] <= 60) & \
            df['DWREAS'].isin([1, 2, 3, 4, 5, 6]) & \
            df['EMPSTAT'].isin([10, 12]) & \
            (df['DWYEARS'] < 99) & \
            (df['DWWEEKL'] > 0) & (df['DWWEEKL'] < 9000) & \
            (df['DWWEEKC'] > 0) & (df['DWWEEKC'] < 9000) & \
            (df['DWLASTWRK'] < 99)
s_noft = df[mask_noft].copy()
s_noft['disp_year'] = s_noft['YEAR'] - s_noft['DWLASTWRK']
s_noft['tenure_bin'] = pd.cut(s_noft['DWYEARS'], bins=[-0.1, 5, 10, 20, 100],
                              labels=['0-5', '6-10', '11-20', '21+'])
s_noft['def_cur'] = (s_noft['YEAR'] - 1).map(deflator)
s_noft['def_pri'] = s_noft['disp_year'].map(deflator)
s_noft = s_noft.dropna(subset=['def_cur', 'def_pri'])
s_noft['lwc'] = np.log(s_noft['DWWEEKC'] / s_noft['def_cur']) - np.log(s_noft['DWWEEKL'] / s_noft['def_pri'])
s_noft['plant_closing'] = (s_noft['DWREAS'] == 1).astype(int)

print(f"\n\nNo FT filter, YEAR-1 deflation:")
print(f"N = {len(s_noft)}")
for b in bins_list + ['Total']:
    d = s_noft if b == 'Total' else s_noft[s_noft['tenure_bin'] == b]
    lwc_m = d['lwc'].mean()
    lwc_se = d['lwc'].std() / np.sqrt(len(d))
    pc_m = d['plant_closing'].mean()
    print(f"  {b}: lwc={lwc_m:.3f} ({lwc_se:.3f}) pc={pc_m:.3f}")
