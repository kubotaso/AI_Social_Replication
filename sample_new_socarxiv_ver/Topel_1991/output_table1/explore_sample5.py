import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# Let me try various approaches to get closer to the paper values
# The key values to match: total mean log wc = -.135, total plant_closing = .390
# Current best (base, no extra filters, N=5691) gives -.162 and .460

# APPROACH 1: No deflation (nominal log wage change)
# Maybe the paper reports nominal changes?
m_base = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & (df['DWYEARS']<99)
sample = df[m_base].copy()
vw = (sample['DWWEEKL']>0) & (sample['DWWEEKL']<9000) & \
     (sample['DWWEEKC']>0) & (sample['DWWEEKC']<9000)
ws = sample[vw].copy()
ws['log_wc_nominal'] = np.log(ws['DWWEEKC']) - np.log(ws['DWWEEKL'])

print("=== APPROACH 1: Nominal log wage change ===")
print(f"Total mean (nominal): {ws['log_wc_nominal'].mean():.3f} (paper: -.135)")

# APPROACH 2: Different deflator values
# Maybe I should use PCE deflator index from different source
# Try BEA actual values
deflator_v2 = {
    1978: 71.55, 1979: 77.60, 1980: 85.70, 1981: 93.95,
    1982: 100.00, 1983: 103.24, 1984: 107.28, 1985: 110.86, 1986: 113.35
}

ws2 = ws.copy()
ws2['disp_year'] = ws2['YEAR'] - ws2['DWLASTWRK']
ws2 = ws2[ws2['DWLASTWRK'] < 99].copy()
ws2['def_cur'] = ws2['YEAR'].map(deflator_v2)
ws2['def_pri'] = ws2['disp_year'].map(deflator_v2)
ws2 = ws2.dropna(subset=['def_cur', 'def_pri'])
ws2['log_wc_real'] = np.log(ws2['DWWEEKC']/ws2['def_cur']) - np.log(ws2['DWWEEKL']/ws2['def_pri'])
print(f"\n=== APPROACH 2: Different deflator ===")
print(f"Total mean (real v2): {ws2['log_wc_real'].mean():.3f} (paper: -.135)")

# APPROACH 3: Maybe DWWEEKL is already the pre-displacement weekly wage
# and we don't know the actual displacement year
# Paper says earnings are from "the most recent full-time civilian job"
# For nominal approach, just do log(DWWEEKC) - log(DWWEEKL)

# Actually wait -- the paper says the data is deflated. But let me also check
# what DWWEEKL actually represents. IPUMS says:
# DWWEEKL: "Usual weekly earnings at lost job"
# DWWEEKC: "Usual weekly earnings at current job"
# Both are in nominal dollars from the time of the respective job

# The paper says "Nominal data deflated by GNP price deflator"
# So Table 1 shows REAL wage changes

# APPROACH 4: Maybe DWREAS definition differs in the paper
# Paper says "layoffs or plant closings" - in the original CPS microdata:
# Plant closed or moved = plant closing
# Insufficient work = layoff
# Position or shift abolished = layoff
# All three are "economic reasons"
# But maybe the original CPS codes are different

# APPROACH 5: Weighted statistics
print(f"\n=== APPROACH 5: Weighted statistics ===")
# Using all valid wage obs, undeflated
w = ws['DWSUPPWT']
wm = np.average(ws['log_wc_nominal'], weights=w)
print(f"Weighted nominal log wc: {wm:.3f} (paper: -.135)")

# Weighted plant closing
wp = np.average(sample['DWREAS']==1, weights=sample['DWSUPPWT'])
print(f"Weighted plant closing: {wp:.3f} (paper: .390)")

# Weighted weeks unemp
us = sample[sample['DWWKSUN']<999].copy()
ww = np.average(us['DWWKSUN'], weights=us['DWSUPPWT'])
print(f"Weighted weeks unemp: {ww:.2f} (paper: 20.41)")

# Weighted real log wc (using ws2 which has deflated values)
wm2 = np.average(ws2['log_wc_real'], weights=ws2['DWSUPPWT'])
print(f"Weighted real log wc: {wm2:.3f} (paper: -.135)")

# APPROACH 6: Maybe the issue is how DWREAS maps to plant closing
# In the ORIGINAL CPS supplement (not IPUMS), coding might differ
# IPUMS DWREAS:
# 1 = Plant or company closed down or moved
# 2 = Insufficient work
# 3 = Position or shift abolished
# Maybe in the original, "plant closing" = only "Plant closed down" (not "moved"?)
# Or maybe the code 1 also includes other things

# Let me just focus on getting the N right and see what plant_closing % looks like

# APPROACH 7: The paper might use DWREAS 1 and 2 for "layoffs and plant closings"
# and code 3 might be something else
# With DWREAS 1,2 only:
m_12 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
       (df['DWREAS'].isin([1,2])) & (df['EMPSTAT'].isin([10,12])) & (df['DWYEARS']<99)
s12 = df[m_12]
print(f"\n=== APPROACH 7: DWREAS 1,2 only ===")
print(f"N = {len(s12)}")
pc12 = (s12['DWREAS']==1).mean()
print(f"Plant closing: {pc12:.3f} (paper: .390)")

# Hmm, .381 would be close to .390 with DWREAS 1,2 only
# Let me check weighted
wp12 = np.average(s12['DWREAS']==1, weights=s12['DWSUPPWT'])
print(f"Weighted plant closing: {wp12:.3f}")

# APPROACH 8: DWREAS 1,2 only + some other filter to get N~4367
# DWREAS 1,2 gives 5034, still too many
# Full-time: 4768
# Full-time + DWYEARS>=1: 3883

# Let me check DWREAS 1,2 + full-time
m_12_ft = m_12 & (df['DWFULLTIME']==2)
s12ft = df[m_12_ft]
print(f"\nDWREAS 1,2 + full-time: N = {len(s12ft)}")
pc12ft = (s12ft['DWREAS']==1).mean()
print(f"Plant closing: {pc12ft:.3f}")

# Try weighted
wp12ft = np.average(s12ft['DWREAS']==1, weights=s12ft['DWSUPPWT'])
print(f"Weighted plant closing: {wp12ft:.3f}")

# Check log wage change for this sample
vw12 = (s12ft['DWWEEKL']>0) & (s12ft['DWWEEKL']<9000) & \
       (s12ft['DWWEEKC']>0) & (s12ft['DWWEEKC']<9000)
ws12 = s12ft[vw12].copy()
ws12['log_wc_nom'] = np.log(ws12['DWWEEKC']) - np.log(ws12['DWWEEKL'])
print(f"Nominal log wc: {ws12['log_wc_nom'].mean():.3f}")

# With deflation
ws12['disp_year'] = ws12['YEAR'] - ws12['DWLASTWRK']
ws12 = ws12[ws12['DWLASTWRK'] < 99].copy()
deflator = {1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
            1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8}
ws12['def_cur'] = ws12['YEAR'].map(deflator)
ws12['def_pri'] = ws12['disp_year'].map(deflator)
ws12 = ws12.dropna(subset=['def_cur', 'def_pri'])
ws12['log_wc_real'] = np.log(ws12['DWWEEKC']/ws12['def_cur']) - np.log(ws12['DWWEEKL']/ws12['def_pri'])
print(f"Real log wc: {ws12['log_wc_real'].mean():.3f}")

# WEIGHTED
print(f"\nWeighted for DWREAS 1,2 + full-time:")
wm_nom = np.average(ws12['log_wc_nom'], weights=ws12['DWSUPPWT']) if len(ws12)>0 else float('nan')
wm_real = np.average(ws12['log_wc_real'], weights=ws12['DWSUPPWT']) if len(ws12)>0 else float('nan')
print(f"  Weighted nominal log wc: {wm_nom:.3f}")
print(f"  Weighted real log wc: {wm_real:.3f}")

# APPROACH 9: Base sample (all DWREAS 1-3) with NOMINAL wage change
# This approach gives N=5691 which is too many
# But what if N=4367 refers to the wage sample only?
# Valid wage subsample has N=3965 -- too few
# Actually let me count the total N more carefully
# The paper says N = 4,367 men. Maybe it's the number with
# at least one valid outcome (wage change OR plant closing OR weeks unemp)
# All have DWREAS (plant closing is always defined)
# So N must be the overall sample

# APPROACH 10: Maybe DWREAS includes codes differently
# Let me check what IPUMS DWS code 98 means
print(f"\n=== Additional checks ===")
all_disp = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
           (df['EMPSTAT'].isin([10,12])) & (df['DWYEARS']<99)
ad = df[all_disp]
print(f"All displaced (any reason), employed, valid tenure: N = {len(ad)}")
print(f"DWREAS distribution:")
print(ad['DWREAS'].value_counts().sort_index())

# Check EDUC distribution
print(f"\nEDUC range in base sample: {sample['EDUC'].min()} - {sample['EDUC'].max()}")
print(f"EDUC value counts:")
print(sample['EDUC'].value_counts().sort_index().head(20))

# APPROACH 11: Maybe DWREAS includes 4,5,6 and "plant closing" is recoded
# 4=Slack work or business conditions
# 5=Seasonal job completed
# 6=Other
m_all_reas = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
             (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & (df['DWYEARS']<99)
s_all = df[m_all_reas]
print(f"\nAll DWREAS 1-6: N = {len(s_all)}")
pc_all = (s_all['DWREAS']==1).mean()
print(f"Plant closing (DWREAS==1): {pc_all:.3f}")
