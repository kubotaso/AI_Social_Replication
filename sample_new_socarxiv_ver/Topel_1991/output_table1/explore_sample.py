import pandas as pd

df = pd.read_csv('data/cps_dws.csv')

m = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
    (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & \
    (df['DWYEARS']<99)
s = df[m]

print("=== BASE SAMPLE ===")
print(f"N = {len(s)}")

# Try full-time on lost job
ft2 = s[s['DWFULLTIME'] == 2]
print(f"\nDWFULLTIME==2 (full-time on lost job): N = {len(ft2)}")

# Try DWREAS 1,2 only (exclude 3=position abolished)
m12 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
      (df['DWREAS'].isin([1,2])) & (df['EMPSTAT'].isin([10,12])) & \
      (df['DWYEARS']<99)
print(f"\nDWREAS 1,2 only: N = {len(df[m12])}")

# Full-time + DWREAS 1,2
m12ft = m12 & (df['DWFULLTIME']==2)
print(f"DWREAS 1,2 + full-time: N = {len(df[m12ft])}")

# Full-time + DWREAS 1,2,3
m123ft = m & (df['DWFULLTIME']==2)
print(f"DWREAS 1,2,3 + full-time: N = {len(df[m123ft])}")

# Try valid wages (maybe paper N is wage-valid only)
vw = (s['DWWEEKL']>0) & (s['DWWEEKL']<9000) & \
     (s['DWWEEKC']>0) & (s['DWWEEKC']<9000)
print(f"\nValid wages in base sample: N = {vw.sum()}")

# Full-time + valid wages
ft2vw = ft2.loc[vw.reindex(ft2.index, fill_value=False)]
print(f"Full-time + valid wages: N = {len(ft2vw)}")

# Try DWREAS only 1 (plant closing)
m1 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
     (df['DWREAS']==1) & (df['EMPSTAT'].isin([10,12])) & (df['DWYEARS']<99)
print(f"\nDWREAS 1 only (plant closing): N = {len(df[m1])}")

# What if N=4367 refers to a broader employment definition?
# EMPSTAT: 1 = Armed forces, 10 = At work, 12 = Has job
m_emp1 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([1, 10, 12])) & \
         (df['DWYEARS']<99)
print(f"\nWith EMPSTAT 1,10,12: N = {len(df[m_emp1])}")

# Try without DWYEARS filter (N is not about wage sample)
m_nody = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12]))
print(f"\nWithout DWYEARS filter: N = {len(df[m_nody])}")

# Full-time without DWYEARS filter
m_nody_ft = m_nody & (df['DWFULLTIME']==2)
print(f"Full-time, no DWYEARS filter: N = {len(df[m_nody_ft])}")

# Try with DWLASTWRK restriction (displaced in past 5 years)
# Paper says "displaced from a job in the past 5 years"
m_last5 = m & (s['DWLASTWRK'] <= 5)
print(f"\nDWLASTWRK<=5: N = {len(s[s['DWLASTWRK']<=5])}")

# Full-time + last 5 years
m_ft_last5 = m & (df['DWFULLTIME']==2) & (df['DWLASTWRK']<=5)
# need to apply all
sub = df[(df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & \
         (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & (df['DWLASTWRK']<=5)]
print(f"Full-time + DWLASTWRK<=5: N = {len(sub)}")

# Maybe N=4367 includes DWREAS 4,5,6 for some definitions?
# 4=slack work, 5=material shortage, 6=other
# No, those are non-economic

# Perhaps DWYEARS >= 1 (at least 1 year tenure)?
m_y1 = m & (df['DWYEARS'] >= 1)
print(f"\nDYEARS>=1: N = {len(df[m_y1])}")

# Full-time + DWYEARS >= 1
m_y1ft = m_y1 & (df['DWFULLTIME']==2)
print(f"Full-time + DWYEARS>=1: N = {len(df[m_y1ft])}")

# Check: maybe include DWREAS 4 (slack work)?
m_14 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
       (df['DWREAS'].isin([1,2,3,4])) & (df['EMPSTAT'].isin([10,12])) & (df['DWYEARS']<99)
print(f"\nDWREAS 1-4: N = {len(df[m_14])}")

# Full-time + DWREAS 1-4
m_14ft = m_14 & (df['DWFULLTIME']==2)
print(f"Full-time + DWREAS 1-4: N = {len(df[m_14ft])}")

# Also check: base with full-time and valid DWLASTWRK
sub2 = df[(df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
          (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & \
          (df['DWYEARS']<99) & (df['DWFULLTIME']==2)]
print(f"\nFull-time only: N = {len(sub2)}")
print(f"Full-time + DWLASTWRK<99: N = {len(sub2[sub2['DWLASTWRK']<99])}")

# YEAR distribution
print(f"\nBy YEAR:")
print(s['YEAR'].value_counts().sort_index())

# Try age 21-60
m21 = (df['SEX']==1) & (df['AGE']>=21) & (df['AGE']<=60) & \
      (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & (df['DWYEARS']<99)
print(f"\nAge 21-60: N = {len(df[m21])}")

# CLASSWKR - restrict to wage/salary workers?
print(f"\nCLASSWKR in sample:")
print(s['CLASSWKR'].value_counts().sort_index())
wage_workers = s[s['CLASSWKR'].isin([22, 25, 27, 28])]  # Private and government wage/salary workers
print(f"Wage/salary workers only: N = {len(wage_workers)}")
