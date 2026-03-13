import pandas as pd

df = pd.read_csv('data/cps_dws.csv')

# Base: male, 20-60, economic displacement, employed, valid tenure
m = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
    (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & \
    (df['DWYEARS']<99)
s = df[m]

# The paper says "displaced from a job in the past 5 years"
# Maybe the 3-year tenure minimum? Paper discusses 3+ years seniority later
# But Table 1 has 0-5 year bin so must include low tenure

# DWYEARS >= 1 + full-time gives 4414, close to 4367
# Try DWYEARS >= 1 + DWREAS 1,2 (not 3)
m_y1_r12 = m & (df['DWYEARS']>=1) & (df['DWREAS'].isin([1,2]))
print(f"DWYEARS>=1 + DWREAS 1,2: N = {len(df[m_y1_r12])}")

# DWYEARS >= 1 + full-time + DWREAS 1,2
m_y1_r12_ft = m_y1_r12 & (df['DWFULLTIME']==2)
print(f"DWYEARS>=1 + DWREAS 1,2 + full-time: N = {len(df[m_y1_r12_ft])}")

# Full-time + DWYEARS >=1 (already know: 4414)
print(f"\nFull-time + DWYEARS>=1: N = 4414")

# Maybe need to exclude DWLASTWRK == 99
m_ft_y1 = m & (df['DWFULLTIME']==2) & (df['DWYEARS']>=1)
sub = df[m_ft_y1]
print(f"ft+y1: N = {len(sub)}")
print(f"ft+y1+DWLASTWRK<99: N = {len(sub[sub['DWLASTWRK']<99])}")

# Try excluding DWLASTWRK == 0 (displaced this year - might not have valid prior wage)
m_ft_y1_lw = m & (df['DWFULLTIME']==2) & (df['DWYEARS']>=1) & (df['DWLASTWRK']>=1) & (df['DWLASTWRK']<=5)
print(f"ft+y1+DWLASTWRK 1-5: N = {len(df[m_ft_y1_lw])}")

# Actually maybe paper uses DWYEARS >= 1 and includes DWREAS 1,2,3
# Let's try: full-time, DWYEARS >= 1, DWREAS 1,2,3
# Already know: 4414

# What about DWYEARS >= 1, no full-time filter, but remove DWLASTWRK==99
m_y1_lw = m & (df['DWYEARS']>=1) & (df['DWLASTWRK']<99)
print(f"\nDYEARS>=1 + DWLASTWRK<99: N = {len(df[m_y1_lw])}")

# Try different tenure bins: paper says 0-5, which would include 0
# but if minimum is 1, first bin is really 1-5
# Let's check: full-time + DWYEARS >= 1 gives 4414
# The difference from 4367 is 47, about 1%

# Maybe try restricting to private wage/salary workers (CLASSWKR)
# In IPUMS: 13=Self-employed, 14=Self-employed inc., 21=Private, 25=Federal, 27=State, 28=Local, 29=Unpaid
# But the low counts for 25,27,28 suggest CLASSWKR is for current job
# The paper might use DWCLASS for the lost job's class of worker
print(f"\nDWCLASS values in full sample:")
print(s['DWCLASS'].value_counts().sort_index())

# Try full-time + DWYEARS >= 1 + DWCLASS private wage workers
m_ft_y1_priv = m & (df['DWFULLTIME']==2) & (df['DWYEARS']>=1) & (df['DWCLASS'].isin([1, 2]))
print(f"\nft+y1+DWCLASS private: N = {len(df[m_ft_y1_priv])}")

# Maybe exclude workers where DWYEARS rounds to 0?
# Actually let me try DWYEARS > 0 vs >= 1 -- DWYEARS is integer in this data
print(f"\nDYEARS > 0 unique vals:", sorted(s[s['DWYEARS']>0]['DWYEARS'].unique())[:10])
print(f"DWYEARS distribution 0-5:")
for y in range(6):
    n = len(s[s['DWYEARS']==y])
    n_ft = len(s[(s['DWYEARS']==y)&(s['DWFULLTIME']==2)])
    print(f"  DWYEARS={y}: N={n}, ft={n_ft}")

# Let me try: the paper says "displaced from a job... in past 5 years"
# and "currently employed". Maybe DWREAS includes 4-6 in some interpretation?
# Or maybe age restriction is different.

# Try age ranges
for lo in [20, 21]:
    for hi in [59, 60]:
        m_age = (df['SEX']==1) & (df['AGE']>=lo) & (df['AGE']<=hi) & \
                (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & \
                (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & (df['DWYEARS']>=1)
        print(f"Age {lo}-{hi}, ft, y>=1: N = {len(df[m_age])}")

# Maybe the paper doesn't restrict to full-time on lost job
# but does restrict DWYEARS >= 1
# N = 4581 for DWYEARS >= 1 without full-time
# Still too high by 214

# What about DWLASTWRK restriction more carefully?
# Paper says "displaced in past 5 years" -- DWS asks about displacement in past 5 years
# So everyone should satisfy this, but DWLASTWRK values may affect it
# DWLASTWRK: years since last worked at displaced job
# 0 = this year through 5 = 5 years ago

# Try removing observations with missing DWLASTWRK
m_y1_lwv = m & (df['DWYEARS']>=1) & (df['DWLASTWRK']<=5)
print(f"\nDYEARS>=1 + DWLASTWRK<=5: N = {len(df[m_y1_lwv])}")

# ft + DWYEARS>=1 + DWLASTWRK<=5
m_all = m & (df['DWFULLTIME']==2) & (df['DWYEARS']>=1) & (df['DWLASTWRK']<=5)
print(f"ft + DWYEARS>=1 + DWLASTWRK<=5: N = {len(df[m_all])}")

# The 1984 DWS asked about displacements from January 1979 to January 1984
# The 1986 DWS asked about displacements from January 1981 to January 1986
# So different lookback windows
# In 1984: DWLASTWRK can be 0-5 (displaced 1979-1984)
# In 1986: DWLASTWRK can be 0-5 (displaced 1981-1986)
# This means both surveys cover 5-year windows

# Maybe some workers displaced more than 5 years ago are excluded
# DWLASTWRK==99 means NIU
m_ft_y1_lwv = m & (df['DWFULLTIME']==2) & (df['DWYEARS']>=1) & (df['DWLASTWRK']<99)
print(f"ft + DWYEARS>=1 + DWLASTWRK<99: N = {len(df[m_ft_y1_lwv])}")

# Let me try excluding DWYEARS==0 without full-time filter but also checking if
# excluding observations with DWWEEKL == 9999.99 or DWWEEKC == 9999.99 matters
# Actually, N in the paper could be the wage sample N
# Let me compute: ft + DWYEARS >= 1 + valid wages
m_ft_y1_vw = m & (df['DWFULLTIME']==2) & (df['DWYEARS']>=1) & \
             (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
             (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
print(f"\nft + DWYEARS>=1 + valid wages: N = {len(df[m_ft_y1_vw])}")
