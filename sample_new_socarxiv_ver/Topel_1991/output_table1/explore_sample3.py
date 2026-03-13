import pandas as pd

df = pd.read_csv('data/cps_dws.csv')

# This combination gives N=4368, very close to paper's 4367:
# Male, age 20-60, DWREAS 1-3, EMPSTAT 10/12, DWYEARS>=1, DWFULLTIME==2, DWCLASS in [1,2]
# DWCLASS: 1 = Private (incorporated), 2 = Private (not incorporated)
# This excludes government workers and self-employed from the lost job

# Let me check all DWCLASS values:
print("DWCLASS codes:")
print("1 = Private, incorporated")
print("2 = Private, not incorporated")
print("3 = Federal government")
print("4 = State government")
print("5 = Local government")
print("99 = NIU")

m = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
    (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & \
    (df['DWYEARS']>=1) & (df['DWYEARS']<99) & \
    (df['DWFULLTIME']==2) & (df['DWCLASS'].isin([1,2]))
s = df[m]
print(f"\nN = {len(s)}")

# Let's check if DWCLASS == 99 might need to be included
m2 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
     (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & \
     (df['DWYEARS']>=1) & (df['DWYEARS']<99) & \
     (df['DWFULLTIME']==2)
s2 = df[m2]
print(f"\nWithout DWCLASS filter: N = {len(s2)}")
print(f"DWCLASS distribution:")
print(s2['DWCLASS'].value_counts().sort_index())

# Try with DWCLASS != 99 (exclude NIU only)
m3 = m2 & (df['DWCLASS'] != 99)
print(f"\nDWCLASS != 99: N = {len(df[m3])}")

# Actually wait - try DWCLASS in [1, 2] but also check if we get 4367 with a slight tweak
# Maybe DWYEARS > 0 includes 0.5 etc? No, it seems integer.
# Try without full-time but with DWCLASS private
m4 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
     (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & \
     (df['DWYEARS']>=1) & (df['DWYEARS']<99) & (df['DWCLASS'].isin([1,2]))
print(f"\nNo ft filter + DWCLASS private + DWYEARS>=1: N = {len(df[m4])}")

# Try DWCLASS 2 only (private not incorporated)
m5 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
     (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & \
     (df['DWYEARS']>=1) & (df['DWYEARS']<99) & \
     (df['DWFULLTIME']==2) & (df['DWCLASS']==2)
print(f"\nft + DWCLASS 2 only: N = {len(df[m5])}")

# So N=4368 is just 1 off from 4367. Could be a minor coding difference.
# Let's proceed with this sample and compute the table

# Now compute the table with this sample
import numpy as np

sample = s.copy()
sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[0.5, 5, 10, 20, 100], labels=['1-5', '6-10', '11-20', '21+'])
sample['plant_closing'] = (sample['DWREAS'] == 1).astype(int)
sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']

deflator = {1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
            1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8}

# Valid wages
vw = (sample['DWWEEKL']>0) & (sample['DWWEEKL']<9000) & \
     (sample['DWWEEKC']>0) & (sample['DWWEEKC']<9000) & \
     (sample['DWLASTWRK']<99)
ws = sample[vw].copy()
ws['deflator_current'] = ws['YEAR'].map(deflator)
ws['deflator_prior'] = ws['disp_year'].map(deflator)
ws = ws.dropna(subset=['deflator_current', 'deflator_prior'])
ws['log_wc'] = np.log(ws['DWWEEKC']/ws['deflator_current']*100) - np.log(ws['DWWEEKL']/ws['deflator_prior']*100)

# Valid unemployment
us = sample[sample['DWWKSUN']<999].copy()

print(f"\nSample sizes:")
print(f"Total: {len(sample)}")
print(f"Valid wages: {len(ws)}")
print(f"Valid unemp: {len(us)}")

bins = ['1-5', '6-10', '11-20', '21+']
print(f"\nBy bin:")
for b in bins:
    wb = ws[ws['tenure_bin']==b]
    sb = sample[sample['tenure_bin']==b]
    ub = us[us['tenure_bin']==b]
    print(f"  {b}: total={len(sb)}, wages={len(wb)}, unemp={len(ub)}")

print(f"\nRow 1: Average change in log weekly wage")
for b in bins + ['Total']:
    if b == 'Total':
        data = ws
    else:
        data = ws[ws['tenure_bin']==b]
    m = data['log_wc'].mean()
    se = data['log_wc'].std() / np.sqrt(len(data))
    print(f"  {b}: {m:.3f} ({se:.3f})")

print(f"\nRow 2: Percentage displaced by plant closing")
for b in bins + ['Total']:
    if b == 'Total':
        data = sample
    else:
        data = sample[sample['tenure_bin']==b]
    m = data['plant_closing'].mean()
    se = np.sqrt(m*(1-m)/len(data))
    print(f"  {b}: {m:.3f} ({se:.3f})")

print(f"\nRow 3: Weeks unemployed since displacement")
for b in bins + ['Total']:
    if b == 'Total':
        data = us
    else:
        data = us[us['tenure_bin']==b]
    m = data['DWWKSUN'].mean()
    se = data['DWWKSUN'].std() / np.sqrt(len(data))
    print(f"  {b}: {m:.2f} ({se:.3f})")
