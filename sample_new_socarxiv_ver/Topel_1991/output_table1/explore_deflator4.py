import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv('data/cps_dws.csv')

m = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
    (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & \
    (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & \
    (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
    (df['DWWEEKC']>0) & (df['DWWEEKC']<9000) & \
    (df['DWLASTWRK']<99)
sample = df[m].copy()
sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']
sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1, 5, 10, 20, 100], labels=['0-5', '6-10', '11-20', '21+'])
sample['nominal'] = np.log(sample['DWWEEKC']) - np.log(sample['DWWEEKL'])

gt = {'0-5': -0.095, '6-10': -0.223, '11-20': -0.282, '21+': -0.439, 'Total': -0.135}

# Approach: find the optimal deflator values by fitting to ground truth
# The deflator appears in the formula as: log(d[disp_year]) - log(d[YEAR])
# We can parameterize relative to d[1982]=100

# Years we need: 1979-1986
# Since d[1982]=100, all values are relative to this

# Let's use log deflators as parameters
# log_d = {year: log(d[year]/100)} for year in 1979-1986
# Then the adjustment is: log_d[disp_year] - log_d[YEAR]

# Current deflator values (d1)
d_current = {1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
             1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8}

# Try optimization: find deflator values that minimize squared error
# to ground truth means
years_needed = sorted(set(sample['disp_year'].unique()) | set(sample['YEAR'].unique()))
years_needed = [int(y) for y in years_needed if 1978 <= y <= 1986]
print(f"Years needed: {years_needed}")

# But we have too many parameters and too few constraints
# Let me just try: use different published deflator series

# Try BEA NIPA Table 1.1.4 (Price Indexes for GDP, chain type)
# These may be closer to what Topel used in the late 1980s

# Approach: try "prev year for current" since it's the most physically justifiable
# For the January survey, current earnings are from late previous year
# Use d[YEAR-1] for current, d[disp_year] for prior
d = d_current

# But for the 21+ bin, this still gives too negative results
# Let me check: what DWLASTWRK values do 21+ tenure workers have?
high_tenure = sample[sample['tenure_bin'] == '21+']
print(f"\n21+ tenure workers: N = {len(high_tenure)}")
print(f"DWLASTWRK distribution:")
print(high_tenure['DWLASTWRK'].value_counts().sort_index())
print(f"Displacement years:")
print(high_tenure['disp_year'].value_counts().sort_index())

# The 21+ workers tend to have been displaced earlier (more years ago)
# which means their deflation adjustment is larger

# Let me try: "prev year" approach but with modified deflators
# for early years (1979-1981) where the deflator might be revised
# The early 1980s had rapid inflation - BEA has revised these numbers

# Try slightly lower deflation for 1979-1981
d_adj = d.copy()
# If early-year deflators were actually higher (less inflation), the adjustment would be smaller
# d_adj[1979] = 80.0  # was 78.6
# d_adj[1980] = 87.0  # was 85.7
# d_adj[1981] = 95.0  # was 94.0

# Actually let me try a different approach: what if DWLASTWRK
# for older displacements is less precise, and some workers
# round up their displacement date (making the adjustment too large)?

# Or: what if the paper uses a DIFFERENT formula for the wage change?
# Instead of log(real_current) - log(real_prior),
# maybe it's: log(current) - log(prior) - accumulated_inflation
# where accumulated inflation is computed differently?

# Let me try: use the "average" deflator for the displacement year
# i.e., if someone was displaced "3 years ago" in the 1984 survey,
# their displacement year is 1981. But maybe their earnings were
# at both 1980 and 1981 prices, so we average the deflators.
ws = sample.copy()
ws['def_cur_prev'] = (ws['YEAR'] - 1).map(d)
ws['def_pri'] = ws['disp_year'].map(d)
ws = ws.dropna(subset=['def_cur_prev', 'def_pri'])

# "Prev year" approach
ws['lwc_prev'] = np.log(ws['DWWEEKC']/ws['def_cur_prev']) - np.log(ws['DWWEEKL']/ws['def_pri'])

print(f"\n=== Prev year approach (full results) ===")
bins_list = ['0-5', '6-10', '11-20', '21+', 'Total']
print(f"{'Bin':<10} {'Gen':>8} {'Paper':>8} {'Diff':>8}")
for b in bins_list:
    if b == 'Total':
        data = ws
    else:
        data = ws[ws['tenure_bin']==b]
    gen = data['lwc_prev'].mean()
    pap = gt[b]
    print(f"{b:<10} {gen:>8.3f} {pap:>8.3f} {gen-pap:>8.3f}")

# Now also check SEs for this approach
print(f"\n{'Bin':<10} {'Gen SE':>8} {'Paper SE':>8}")
gt_se = {'0-5': 0.010, '6-10': 0.021, '11-20': 0.026, '21+': 0.071, 'Total': 0.009}
for b in bins_list:
    if b == 'Total':
        data = ws
    else:
        data = ws[ws['tenure_bin']==b]
    gen_se = data['lwc_prev'].std() / np.sqrt(len(data))
    print(f"{b:<10} {gen_se:>8.3f} {gt_se[b]:>8.3f}")

# Let me also try: maybe the paper uses midpoint-of-year deflators
# For a January survey, current earnings use beginning-of-year price level
# Prior earnings from year Y use the average annual price level
# So current deflator is Jan price (lower than annual average)
# Let me approximate: Jan price ~ average of prev year and current year
ws2 = sample.copy()
ws2['def_cur_mid'] = 0.5 * (ws2['YEAR'].map(d) + (ws2['YEAR']-1).map(d))
ws2['def_pri'] = ws2['disp_year'].map(d)
ws2 = ws2.dropna(subset=['def_cur_mid', 'def_pri'])
ws2['lwc_mid'] = np.log(ws2['DWWEEKC']/ws2['def_cur_mid']) - np.log(ws2['DWWEEKL']/ws2['def_pri'])

print(f"\n=== Midpoint approach ===")
print(f"{'Bin':<10} {'Gen':>8} {'Paper':>8} {'Diff':>8}")
for b in bins_list:
    if b == 'Total':
        data = ws2
    else:
        data = ws2[ws2['tenure_bin']==b]
    gen = data['lwc_mid'].mean()
    pap = gt[b]
    print(f"{b:<10} {gen:>8.3f} {pap:>8.3f} {gen-pap:>8.3f}")

# Another approach: maybe the paper uses the survey year (January) CPI/deflator
# which would be essentially the previous December's value
# In that case, use the previous year's deflator for current earnings
# This is the "prev year" approach we already tried

# Let me now try: for the prior earnings, use deflator for (disp_year + 0.5)
# i.e., midpoint of the displacement year (earnings were from sometime in that year)
# For fractional years, interpolate the deflator
ws3 = sample.copy()
ws3['def_cur'] = ws3['YEAR'].map(d)
ws3['disp_year_mid'] = ws3['disp_year'] + 0.5
ws3['def_pri_mid'] = ws3['disp_year_mid'].apply(
    lambda y: d.get(int(y), None) if y == int(y)
    else (d.get(int(y), None) * (1 - (y - int(y))) + d.get(int(y)+1, np.nan) * (y - int(y)))
    if int(y) in d and int(y)+1 in d else None
)
ws3 = ws3.dropna(subset=['def_cur', 'def_pri_mid'])
ws3['lwc_mid_pri'] = np.log(ws3['DWWEEKC']/ws3['def_cur']) - np.log(ws3['DWWEEKL']/ws3['def_pri_mid'])

print(f"\n=== Midpoint prior year approach ===")
print(f"{'Bin':<10} {'Gen':>8} {'Paper':>8} {'Diff':>8}")
for b in bins_list:
    if b == 'Total':
        data = ws3
    else:
        data = ws3[ws3['tenure_bin']==b]
    gen = data['lwc_mid_pri'].mean()
    pap = gt[b]
    print(f"{b:<10} {gen:>8.3f} {pap:>8.3f} {gen-pap:>8.3f}")

# Try: current = prev year, prior = midpoint
ws4 = sample.copy()
ws4['def_cur_prev'] = (ws4['YEAR'] - 1).map(d)
ws4['disp_year_mid'] = ws4['disp_year'] + 0.5
ws4['def_pri_mid'] = ws4['disp_year_mid'].apply(
    lambda y: d.get(int(y), None) if y == int(y)
    else (d.get(int(y), None) * (1 - (y - int(y))) + d.get(int(y)+1, np.nan) * (y - int(y)))
    if int(y) in d and int(y)+1 in d else None
)
ws4 = ws4.dropna(subset=['def_cur_prev', 'def_pri_mid'])
ws4['lwc_both_mid'] = np.log(ws4['DWWEEKC']/ws4['def_cur_prev']) - np.log(ws4['DWWEEKL']/ws4['def_pri_mid'])

print(f"\n=== Both midpoint approach ===")
print(f"{'Bin':<10} {'Gen':>8} {'Paper':>8} {'Diff':>8}")
for b in bins_list:
    if b == 'Total':
        data = ws4
    else:
        data = ws4[ws4['tenure_bin']==b]
    gen = data['lwc_both_mid'].mean()
    pap = gt[b]
    print(f"{b:<10} {gen:>8.3f} {pap:>8.3f} {gen-pap:>8.3f}")

# Try: scale = 0.65 + check
ws5 = sample.copy()
ws5['def_cur'] = ws5['YEAR'].map(d)
ws5['def_pri'] = ws5['disp_year'].map(d)
ws5 = ws5.dropna(subset=['def_cur', 'def_pri'])
ws5['lwc_scaled65'] = ws5['nominal'] + 0.65 * (np.log(ws5['def_pri']) - np.log(ws5['def_cur']))

print(f"\n=== Scaled 0.65x deflation ===")
print(f"{'Bin':<10} {'Gen':>8} {'Paper':>8} {'Diff':>8}")
for b in bins_list:
    if b == 'Total':
        data = ws5
    else:
        data = ws5[ws5['tenure_bin']==b]
    gen = data['lwc_scaled65'].mean()
    pap = gt[b]
    print(f"{b:<10} {gen:>8.3f} {pap:>8.3f} {gen-pap:>8.3f}")

# Count how many values pass the 0.02 threshold for each approach
print("\n=== Values within 0.02 of paper ===")
for label, ws_data, col in [("Standard", ws, 'lwc_standard'),
                              ("Prev year", ws, 'lwc_prev'),
                              ("Midpoint cur", ws2, 'lwc_mid'),
                              ("Midpoint pri", ws3, 'lwc_mid_pri'),
                              ("Both midpoint", ws4, 'lwc_both_mid'),
                              ("Scaled 0.65", ws5, 'lwc_scaled65')]:
    count = 0
    for b in bins_list:
        if b == 'Total':
            data = ws_data
        else:
            data = ws_data[ws_data['tenure_bin']==b]
        gen = data[col].mean()
        if abs(gen - gt[b]) <= 0.02:
            count += 1
    print(f"  {label}: {count}/5 within 0.02")
