import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# Finding from explore_deflator.py:
# DWREAS 1-6, FT, valid wages (DWWEEKL>0 & DWWEEKC>0), no DWLASTWRK filter
# gives N=4378 (paper: 4367), very close!
# And "inflate prior only to survey year" gives mean lwc = -0.140

# Let me investigate this more carefully

# Sample definition
mask = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
       df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & \
       (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & \
       (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
       (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
sample = df[mask].copy()
print(f"Sample N = {len(sample)}")
print(f"Plant closing = {(sample['DWREAS']==1).mean():.3f} (paper: .390)")

# Need N closer to 4367. Let me try without fulltime filter
mask_noft = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
            df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & \
            (df['DWYEARS']<99) & \
            (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
            (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
s_noft = df[mask_noft]
print(f"\nWithout FT filter: N={len(s_noft)}, pc={(s_noft['DWREAS']==1).mean():.3f}")

# DWREAS 1-3 with valid wages
mask_13 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
          df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & \
          (df['DWYEARS']<99) & \
          (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
          (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
s_13 = df[mask_13]
print(f"DWREAS 1-3 valid wages: N={len(s_13)}, pc={(s_13['DWREAS']==1).mean():.3f}")

# DWREAS 1-3, FT, valid wages
mask_13ft = mask_13 & (df['DWFULLTIME']==2)
s_13ft = df[mask_13ft]
print(f"DWREAS 1-3 + FT + valid wages: N={len(s_13ft)}, pc={(s_13ft['DWREAS']==1).mean():.3f}")

# Hmm, N=4378 for DWREAS 1-6 + FT + valid wages is close to 4367
# But plant_closing = 0.386, close to paper's .390

# Maybe the paper's sample IS defined by having valid wage data
# This makes sense: it's a table about WAGE CHANGES
# The total N=4367 is the wage sample size

# Now let me figure out the deflation
# "Inflate prior only (to survey year)" approach:
# log_wc = log(DWWEEKC) - log(DWWEEKL * deflator[YEAR] / deflator[disp_year])
# = log(DWWEEKC) - log(DWWEEKL) - log(deflator[YEAR]/deflator[disp_year])
# This is the SAME as the standard approach! Both give the same answer.

# Actually wait, let me re-read the explore_deflator.py code:
# ws7['log_wc_pri_only'] = np.log(ws7['DWWEEKC']) - np.log(ws7['DWWEEKL']*d1[1984]/ws7['def_pri'])
# This deflates to 1984 specifically, not to the survey year
# For 1986 survey workers, this would use 1984 deflator for current,
# which is WRONG (they earn in 1986 dollars)
# BUT, it gave -0.140 which is close

# Actually no, it's "inflating PRIOR to 1984" and comparing to NOMINAL current
# For a 1984 survey worker: current is in 1984 dollars, prior inflated to 1984 = correct
# For a 1986 survey worker: current is in 1986 dollars, prior inflated to 1984 = WRONG
# This is a mistake. Let me think more carefully.

# What if the paper deflates everything to a common year like 1982?
# That's what we're doing. So why are our values more negative?

# Maybe the answer is simpler: different sample gives different values
# Let me try DWREAS 1-6 + valid wages, compute PROPER deflation
deflator = {1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0,
            1982:100.0, 1983:103.9, 1984:107.7, 1985:110.9, 1986:113.8}

sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']
sample['plant_closing'] = (sample['DWREAS']==1).astype(int)
sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1,5,10,20,100], labels=['0-5','6-10','11-20','21+'])

# For valid deflation
ws = sample[sample['DWLASTWRK']<99].copy()
ws['def_cur'] = ws['YEAR'].map(deflator)
ws['def_pri'] = ws['disp_year'].map(deflator)
ws = ws.dropna(subset=['def_cur','def_pri'])
ws['real_lwc'] = np.log(ws['DWWEEKC']/ws['def_cur']) - np.log(ws['DWWEEKL']/ws['def_pri'])

print(f"\n=== DWREAS 1-6 + FT + valid wages, proper deflation ===")
print(f"N = {len(ws)}")

bins_list = ['0-5','6-10','11-20','21+','Total']
print(f"\n{'':30s} {'0-5':>10} {'6-10':>10} {'11-20':>10} {'21+':>10} {'Total':>10}")
for row in ['lwc', 'pc', 'wu']:
    means = []
    ses = []
    for b in bins_list:
        if b == 'Total':
            if row == 'lwc': d = ws
            elif row == 'pc': d = sample
            else: d = sample[sample['DWWKSUN']<999]
        else:
            if row == 'lwc': d = ws[ws['tenure_bin']==b]
            elif row == 'pc': d = sample[sample['tenure_bin']==b]
            else: d = sample[(sample['DWWKSUN']<999) & (sample['tenure_bin']==b)]

        n = len(d)
        if row == 'lwc':
            m = d['real_lwc'].mean()
            se = d['real_lwc'].std() / np.sqrt(n)
        elif row == 'pc':
            m = d['plant_closing'].mean()
            se = np.sqrt(m*(1-m)/n)
        else:
            m = d['DWWKSUN'].mean()
            se = d['DWWKSUN'].std() / np.sqrt(n)
        means.append(f"{m:.3f}")
        ses.append(f"({se:.3f})")

    labels = {'lwc': 'Avg chg log wkly wage', 'pc': 'Pct plant closing', 'wu': 'Weeks unemp'}
    print(f"{labels[row]:30s} " + " ".join(f"{v:>10}" for v in means))
    print(f"{'':30s} " + " ".join(f"{v:>10}" for v in ses))

# Try: use full DWREAS 1-6 sample for pc and wu, but wage-valid sample for lwc
# The N=4367 might be the wage sample
# While pc and wu are computed on larger sample with different N
# But the paper only reports one N (4367)...
# Actually, re-reading the paper: "Table 1 presents summary data..."
# "The data... are from the January 1984 and January 1986 Current Population
# Survey displaced workers surveys... The sample consists of 4,367 men,
# aged 20-60, who were displaced from a job in the past 5 years,
# for economic reasons... and were employed at the time of the survey."
# Then it shows the table with three rows.

# So N=4367 is the overall sample, and rows 2 and 3 use same sample,
# while row 1 is a subset (only those with valid wage data)

# But we're getting N=4378 with valid wage filter. If we remove the wage filter,
# DWREAS 1-6 + FT gives N=6367. That's too many.

# What if "full-time" is not the right filter? Let me check DWFULLTIME values
print(f"\n\nDWFULLTIME values for DWREAS 1-6 base:")
m_base = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99)
s_base = df[m_base]
print(s_base['DWFULLTIME'].value_counts().sort_index())

# What about checking if N=4367 means the paper uses a weight-based effective N?
# Or if IPUMS has some records that the original data didn't have

# Let me try: maybe the paper uses EMPSTAT 10 only (at work, not "has job")
mask_e10 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
           df['DWREAS'].isin([1,2,3,4,5,6]) & (df['EMPSTAT']==10) & \
           (df['DWYEARS']<99) & (df['DWFULLTIME']==2)
s_e10 = df[mask_e10]
print(f"\nDWREAS 1-6 + FT + EMPSTAT=10: N={len(s_e10)}, pc={(s_e10['DWREAS']==1).mean():.3f}")

# + valid wages
mask_e10_vw = mask_e10 & (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
s_e10_vw = df[mask_e10_vw]
print(f"  + valid wages: N={len(s_e10_vw)}, pc={(s_e10_vw['DWREAS']==1).mean():.3f}")

# No FT filter
mask_16_noft = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
               df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & \
               (df['DWYEARS']<99) & \
               (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
               (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
s_noft2 = df[mask_16_noft]
print(f"\nDWREAS 1-6 + valid wages (no FT): N={len(s_noft2)}, pc={(s_noft2['DWREAS']==1).mean():.3f}")

# EMPSTAT 10 only, no FT, valid wages
mask_e10_noft = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
                df['DWREAS'].isin([1,2,3,4,5,6]) & (df['EMPSTAT']==10) & \
                (df['DWYEARS']<99) & \
                (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
                (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
s_e10_noft = df[mask_e10_noft]
print(f"DWREAS 1-6 + EMPSTAT=10 + valid wages: N={len(s_e10_noft)}, pc={(s_e10_noft['DWREAS']==1).mean():.3f}")

# KEY QUESTION: What if N=4367 comes from using the original DWS supplement
# where only displaced workers are included, and IPUMS adds all CPS respondents?
# In IPUMS, DWREAS=99 means "NIU" = not in universe = not displaced
# So DWREAS 1-6 gives all displaced workers
# Original CPS tape would have had a similar set

# The issue is that 6738 (all displaced, employed, male 20-60) is too many
# while 4378 (with valid wages) is just right
# The paper probably defines the sample as having valid wage data

# Let me just go with DWREAS 1-6, FT, valid wages (N=4378)
# and see if we can match the values

# Actually, wait - maybe not FT. Let me check N without FT but with valid wages
# DWREAS 1-6 + valid wages (no FT): N=4611 - still 5% off from 4367
# DWREAS 1-6 + FT + valid wages: N=4378 - only 0.25% off from 4367!

# So DWREAS 1-6 + FT + valid wages is the right definition
# The 11 records difference (4378 vs 4367) is likely IPUMS coding differences

# HOWEVER, the plant_closing rate is 0.386 vs paper's 0.390
# And the log wage change is too negative

# Perhaps the pc and wu rows use the FULL sample (N=6367 for DWREAS 1-6 + FT)
# and only row 1 uses the wage-valid subsample (N=4378)
# But the paper says N=4367...

# Let me just compute all three approaches and compare

print(f"\n\n=== APPROACH A: DWREAS 1-6 + FT, wage rows use wage-valid ===")
# Full sample for pc and wu
mask_full = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
            df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & \
            (df['DWYEARS']<99) & (df['DWFULLTIME']==2)
full = df[mask_full].copy()
full['tenure_bin'] = pd.cut(full['DWYEARS'], bins=[-0.1,5,10,20,100], labels=['0-5','6-10','11-20','21+'])
full['plant_closing'] = (full['DWREAS']==1).astype(int)

# Wage-valid for lwc
wv = (full['DWWEEKL']>0) & (full['DWWEEKL']<9000) & (full['DWWEEKC']>0) & (full['DWWEEKC']<9000) & (full['DWLASTWRK']<99)
ws = full[wv].copy()
ws['disp_year'] = ws['YEAR'] - ws['DWLASTWRK']
ws['def_cur'] = ws['YEAR'].map(deflator)
ws['def_pri'] = ws['disp_year'].map(deflator)
ws = ws.dropna(subset=['def_cur','def_pri'])
ws['real_lwc'] = np.log(ws['DWWEEKC']/ws['def_cur']) - np.log(ws['DWWEEKL']/ws['def_pri'])

us = full[full['DWWKSUN']<999]

print(f"Full: N={len(full)}, Wage: N={len(ws)}, Unemp: N={len(us)}")
for b in bins_list:
    if b == 'Total':
        ws_b = ws; ps_b = full; us_b = us
    else:
        ws_b = ws[ws['tenure_bin']==b]; ps_b = full[full['tenure_bin']==b]; us_b = us[us['tenure_bin']==b]
    lwc_m = ws_b['real_lwc'].mean()
    pc_m = ps_b['plant_closing'].mean()
    wu_m = us_b['DWWKSUN'].mean()
    print(f"  {b:>5}: lwc={lwc_m:.3f} pc={pc_m:.3f} wu={wu_m:.2f} N_pc={len(ps_b)} N_wc={len(ws_b)} N_wu={len(us_b)}")
