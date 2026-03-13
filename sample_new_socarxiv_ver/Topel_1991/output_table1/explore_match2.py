import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# DWREAS 1-6 + FT gives pc=0.389 (matches paper's .390), but N=6367
# We need N closer to 4367
# Let me explore more filters on the DWREAS 1-6 + FT base

base = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
       df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & \
       (df['DWYEARS']<99) & (df['DWFULLTIME']==2)
s = df[base].copy()
print(f"DWREAS 1-6 + FT: N={len(s)}")

# What if IPUMS codes DWREAS differently from original CPS?
# In original CPS January 1984 DWS:
# 1 = Plant closed
# 2 = Slack work
# 3 = Position abolished
# These are "economic reasons"
# But the IPUMS extract also has codes 4, 5, 6 for "other" reasons
# Maybe the paper actually uses codes 1-3 but with different data definition

# Key observation: the paper says "displaced... for economic reasons (layoffs
# or plant closings)" and in the text mentions 5-year lookback window

# Let me check: what if we restrict to DWLASTWRK 1-5 (displaced 1-5 years ago)?
# "Past 5 years" from 1984 = 1979-1983, from 1986 = 1981-1985
# DWLASTWRK=0 means "this year" which might be excluded

# Also, what if the paper data is from the raw CPS tape and IPUMS has
# different coding? The key metrics to match simultaneously:
# - N ~ 4367
# - Total plant_closing ~ 0.390
# - Total log wage change ~ -0.135
# - Total weeks unemp ~ 20.41

# Let me try: DWREAS 1-3 but with different bin boundaries for tenure
# What if DWYEARS is coded differently? E.g. DWYEARS=0 means <1 year?
# And the "0-5" bin means DWYEARS 1-5?
print(f"\nDYEARS=0 count in DWREAS 1-3 base: {(df[(df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99)]['DWYEARS']==0).sum()}")

# what does DWYEARS=0 mean? Could be <1 year or could be NIU
# Actually 0 years = less than 1 year tenure
# But "0-5 years seniority" should include 0

# Let me think differently. The paper uses ORIGINAL CPS data, not IPUMS.
# Maybe IPUMS adds some records or codes differently.

# Let me try a completely different approach:
# What if the sample is defined by having valid DWWEEKL (pre-displacement wage)?
# The paper focuses on wage CHANGES, so maybe the "sample" is just people
# with valid wage data

# DWREAS 1-3, valid DWWEEKL
m_vwl = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
        df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & \
        (df['DWYEARS']<99) & (df['DWWEEKL']>0) & (df['DWWEEKL']<9000)
svwl = df[m_vwl]
print(f"\nDWREAS 1-3 + valid DWWEEKL: N={len(svwl)}, pc={(svwl['DWREAS']==1).mean():.3f}")

# + valid DWWEEKC
m_vw = m_vwl & (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
svw = df[m_vw]
print(f"  + valid DWWEEKC: N={len(svw)}, pc={(svw['DWREAS']==1).mean():.3f}")

# + fulltime + valid DWWEEKL
m_ftvwl = m_vwl & (df['DWFULLTIME']==2)
sftvwl = df[m_ftvwl]
print(f"  + fulltime: N={len(sftvwl)}, pc={(sftvwl['DWREAS']==1).mean():.3f}")

# Let me also check: what if we use both January 1984 and January 1986
# but there are different EMPSTAT codes in different years?
print(f"\nEMPSTAT by year for displaced men:")
for y in [1984, 1986]:
    sub = df[(df['YEAR']==y) & (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & df['DWREAS'].isin([1,2,3])]
    print(f"  Year {y}: N={len(sub)}")
    print(sub['EMPSTAT'].value_counts().sort_index())

# DWLOSTJOB check - maybe this is the better variable for displacement
print(f"\nDWLOSTJOB values:")
print(df['DWLOSTJOB'].value_counts().sort_index())

# Check if DWLOSTJOB helps define the sample better
m_lost = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         (df['DWLOSTJOB']==2) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99)
slost = df[m_lost]
print(f"\nDWLOSTJOB=2 + employed: N={len(slost)}")

# Check unique DWREAS for this
print(f"DWREAS in DWLOSTJOB=2 sample:")
print(slost['DWREAS'].value_counts().sort_index())

# Actually, maybe the key is that the paper includes ALL displaced workers
# (DWREAS 1-3) but with valid earnings on the prior job (DWWEEKL > 0)
# since the table is about WAGE CHANGES
# Let me check if DWWEEKL > 0 gets us closer to 4367
# Already checked: 4571, still too many

# Let me try: maybe the answer is that the original CPS tapes
# have different sample sizes than IPUMS extracts, and we should
# just use the closest sample and accept the N difference

# Key insight: let me focus on getting VALUES right, not just N
# Let me compute ALL rows for various sample definitions and see
# which best matches the paper

deflator = {1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0,
            1982:100.0, 1983:103.9, 1984:107.7, 1985:110.9, 1986:113.8}

configs = {
    'DWREAS 1-3 base': (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99),
    'DWREAS 1-3 FT': (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99) & (df['DWFULLTIME']==2),
    'DWREAS 1-6 base': (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99),
    'DWREAS 1-6 FT': (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99) & (df['DWFULLTIME']==2),
}

paper_vals = {
    'N': 4367,
    'total_lwc': -0.135,
    'total_pc': 0.390,
    'total_wu': 20.41,
    'lwc_05': -0.095,
    'pc_05': 0.352,
    'wu_05': 18.69,
}

print("\n\n=== Comparing sample definitions ===")
for label, mask in configs.items():
    sample = df[mask].copy()
    sample['plant_closing'] = (sample['DWREAS']==1).astype(int)
    sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']
    sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1,5,10,20,100], labels=['0-5','6-10','11-20','21+'])

    # Wage sample
    vw = (sample['DWWEEKL']>0) & (sample['DWWEEKL']<9000) & \
         (sample['DWWEEKC']>0) & (sample['DWWEEKC']<9000) & (sample['DWLASTWRK']<99)
    ws = sample[vw].copy()
    ws['def_cur'] = ws['YEAR'].map(deflator)
    ws['def_pri'] = ws['disp_year'].map(deflator)
    ws = ws.dropna(subset=['def_cur','def_pri'])
    ws['real_lwc'] = np.log(ws['DWWEEKC']/ws['def_cur']) - np.log(ws['DWWEEKL']/ws['def_pri'])

    # Unemp sample
    us = sample[sample['DWWKSUN']<999]

    # 0-5 bin
    ws05 = ws[ws['tenure_bin']=='0-5']
    ps05 = sample[sample['tenure_bin']=='0-5']
    us05 = us[us['tenure_bin']=='0-5']

    total_lwc = ws['real_lwc'].mean()
    total_pc = sample['plant_closing'].mean()
    total_wu = us['DWWKSUN'].mean()
    lwc_05 = ws05['real_lwc'].mean()
    pc_05 = ps05['plant_closing'].mean()
    wu_05 = us05['DWWKSUN'].mean()

    print(f"\n{label}: N={len(sample)}")
    print(f"  Total: lwc={total_lwc:.3f}({paper_vals['total_lwc']:.3f}), pc={total_pc:.3f}({paper_vals['total_pc']:.3f}), wu={total_wu:.2f}({paper_vals['total_wu']:.2f})")
    print(f"  0-5:   lwc={lwc_05:.3f}({paper_vals['lwc_05']:.3f}), pc={pc_05:.3f}({paper_vals['pc_05']:.3f}), wu={wu_05:.2f}({paper_vals['wu_05']:.2f})")

    # Compute error metric
    err_pc = abs(total_pc - paper_vals['total_pc'])
    err_lwc = abs(total_lwc - paper_vals['total_lwc'])
    err_wu = abs(total_wu - paper_vals['total_wu'])
    err_n = abs(len(sample) - paper_vals['N']) / paper_vals['N']
    print(f"  Errors: N_pct={err_n:.3f}, pc={err_pc:.3f}, lwc={err_lwc:.3f}, wu={err_wu:.2f}")

# Try nominal wage change (no deflation)
print("\n\n=== Try NOMINAL wage change for DWREAS 1-3 base ===")
mask_base = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
            df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99)
sample = df[mask_base].copy()
sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1,5,10,20,100], labels=['0-5','6-10','11-20','21+'])

vw = (sample['DWWEEKL']>0) & (sample['DWWEEKL']<9000) & \
     (sample['DWWEEKC']>0) & (sample['DWWEEKC']<9000)
ws = sample[vw].copy()
ws['nom_lwc'] = np.log(ws['DWWEEKC']) - np.log(ws['DWWEEKL'])

bins_list = ['0-5','6-10','11-20','21+','Total']
print(f"{'':30s} {'0-5':>8} {'6-10':>8} {'11-20':>8} {'21+':>8} {'Total':>8}")
means = []
for b in bins_list:
    if b == 'Total':
        d = ws
    else:
        d = ws[ws['tenure_bin']==b]
    means.append(f"{d['nom_lwc'].mean():.3f}")
print(f"{'Nominal log wc':30s} " + " ".join(f"{v:>8}" for v in means))
print(f"{'Paper':30s}    -.095    -.223    -.282    -.439    -.135")
