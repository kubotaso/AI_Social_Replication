import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# Base: male, 20-60, employed, valid tenure
base = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
       df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99)

# For each DWREAS combo + additional filters, find N close to 4367
combos = [
    ('DWREAS 1-3', [1,2,3]),
    ('DWREAS 1-2', [1,2]),
    ('DWREAS 1', [1]),
]

for label, reas_codes in combos:
    m = base & df['DWREAS'].isin(reas_codes)
    s = df[m]
    n = len(s)
    pc = (s['DWREAS']==1).mean()

    # Various additional filters
    print(f"\n=== {label} (N={n}, pc={pc:.3f}) ===")

    # + fulltime
    ft = s[s['DWFULLTIME']==2]
    print(f"  + fulltime: N={len(ft)}, pc={(ft['DWREAS']==1).mean():.3f}")

    # + DWYEARS>=1
    y1 = s[s['DWYEARS']>=1]
    print(f"  + DWYEARS>=1: N={len(y1)}, pc={(y1['DWREAS']==1).mean():.3f}")

    # + fulltime + DWYEARS>=1
    fty1 = ft[ft['DWYEARS']>=1]
    print(f"  + fulltime + DWYEARS>=1: N={len(fty1)}, pc={(fty1['DWREAS']==1).mean():.3f}")

    # + private sector
    priv = s[s['DWCLASS'].isin([1,2])]
    print(f"  + private: N={len(priv)}, pc={(priv['DWREAS']==1).mean():.3f}")

    # + private + fulltime
    pft = priv[priv['DWFULLTIME']==2]
    print(f"  + private + fulltime: N={len(pft)}, pc={(pft['DWREAS']==1).mean():.3f}")

    # + private + DWYEARS>=1
    py1 = priv[priv['DWYEARS']>=1]
    print(f"  + private + DWYEARS>=1: N={len(py1)}, pc={(py1['DWREAS']==1).mean():.3f}")

    # + private + fulltime + DWYEARS>=1
    pfy1 = pft[pft['DWYEARS']>=1]
    print(f"  + private + fulltime + DWYEARS>=1: N={len(pfy1)}, pc={(pfy1['DWREAS']==1).mean():.3f}")

# Also check: what if we need valid wage data for ALL
print("\n\n=== Checking if N=4367 is the wage-valid sample ===")
m13 = base & df['DWREAS'].isin([1,2,3])
s13 = df[m13].copy()
vw = (s13['DWWEEKL']>0) & (s13['DWWEEKL']<9000) & (s13['DWWEEKC']>0) & (s13['DWWEEKC']<9000)
wg = s13[vw]
print(f"DWREAS 1-3, valid wages: N={len(wg)}")

# what about valid wages + fulltime
wg_ft = wg[wg['DWFULLTIME']==2]
print(f"  + fulltime: N={len(wg_ft)}")

# Valid wages + DWYEARS>=1
wg_y1 = wg[wg['DWYEARS']>=1]
print(f"  + DWYEARS>=1: N={len(wg_y1)}")

# Valid wages + fulltime + DWYEARS>=1
wg_ft_y1 = wg_ft[wg_ft['DWYEARS']>=1]
print(f"  + fulltime + DWYEARS>=1: N={len(wg_ft_y1)}")

# maybe DWREAS 1-3, fulltime, private sector?
m13_ft_priv = m13 & (df['DWFULLTIME']==2) & (df['DWCLASS'].isin([1,2]))
s_ftp = df[m13_ft_priv]
print(f"\nDWREAS 1-3 + fulltime + private: N={len(s_ftp)}, pc={(s_ftp['DWREAS']==1).mean():.3f}")

# DWREAS 1-3, private, no fulltime
m13_priv = m13 & (df['DWCLASS'].isin([1,2]))
s_priv = df[m13_priv]
print(f"DWREAS 1-3 + private: N={len(s_priv)}, pc={(s_priv['DWREAS']==1).mean():.3f}")

# Maybe the paper uses DWREAS 1-3 but with DWLASTWRK <= 4 (within last 5 years)
# In 1984 survey: displaced 1979-1983
# In 1986 survey: displaced 1981-1985
# DWLASTWRK values: 0=current year, 1=last year, etc.
# For 1984 survey: DWLASTWRK 0-5 means displaced 1978-1983
# "past 5 years" - could be DWLASTWRK 1-5 (excluding current year)
m13_lw15 = m13 & (df['DWLASTWRK']>=1) & (df['DWLASTWRK']<=5)
s_lw15 = df[m13_lw15]
print(f"\nDWREAS 1-3 + DWLASTWRK 1-5: N={len(s_lw15)}, pc={(s_lw15['DWREAS']==1).mean():.3f}")

# DWLASTWRK 0-4 (up to 4 years ago)
m13_lw04 = m13 & (df['DWLASTWRK']>=0) & (df['DWLASTWRK']<=4)
s_lw04 = df[m13_lw04]
print(f"DWREAS 1-3 + DWLASTWRK 0-4: N={len(s_lw04)}, pc={(s_lw04['DWREAS']==1).mean():.3f}")

# DWLASTWRK 0-3
m13_lw03 = m13 & (df['DWLASTWRK']>=0) & (df['DWLASTWRK']<=3)
s_lw03 = df[m13_lw03]
print(f"DWREAS 1-3 + DWLASTWRK 0-3: N={len(s_lw03)}, pc={(s_lw03['DWREAS']==1).mean():.3f}")

# Let me also try EMPSTAT 10 only (at work, not just "has job")
m13_e10 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
          df['DWREAS'].isin([1,2,3]) & (df['EMPSTAT']==10) & (df['DWYEARS']<99)
s_e10 = df[m13_e10]
print(f"\nDWREAS 1-3 + EMPSTAT=10 only: N={len(s_e10)}, pc={(s_e10['DWREAS']==1).mean():.3f}")

# fulltime + EMPSTAT 10
s_e10_ft = s_e10[s_e10['DWFULLTIME']==2]
print(f"  + fulltime: N={len(s_e10_ft)}, pc={(s_e10_ft['DWREAS']==1).mean():.3f}")

# private + EMPSTAT 10
s_e10_priv = s_e10[s_e10['DWCLASS'].isin([1,2])]
print(f"  + private: N={len(s_e10_priv)}, pc={(s_e10_priv['DWREAS']==1).mean():.3f}")

# private + fulltime + EMPSTAT 10
s_e10_pft = s_e10_ft[s_e10_ft['DWCLASS'].isin([1,2])]
print(f"  + private + fulltime: N={len(s_e10_pft)}, pc={(s_e10_pft['DWREAS']==1).mean():.3f}")

# Try: DWREAS 1-3 + fulltime only (no private)
m13_ft = m13 & (df['DWFULLTIME']==2)
s_ft = df[m13_ft]
print(f"\nDWREAS 1-3 + fulltime: N={len(s_ft)}, pc={(s_ft['DWREAS']==1).mean():.3f}")

# Check DWSTAT variable (might capture reemployment status better)
print(f"\nDWSTAT values:")
print(df['DWSTAT'].value_counts().sort_index())

# Try using DWSTAT to identify employed displaced workers
m_dwstat = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
           df['DWREAS'].isin([1,2,3]) & (df['DWYEARS']<99)
# DWSTAT has employment outcome codes
print(f"\nDWSTAT for DWREAS 1-3 filtered sample:")
print(df[m_dwstat]['DWSTAT'].value_counts().sort_index())
