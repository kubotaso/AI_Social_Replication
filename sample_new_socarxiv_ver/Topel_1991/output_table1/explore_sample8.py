import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# Best match: DWREAS 1-6, full-time, valid wages, valid DWLASTWRK
# N = 4371, plant_closing = 0.389

m = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
    (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & \
    (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & \
    (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
    (df['DWWEEKC']>0) & (df['DWWEEKC']<9000) & \
    (df['DWLASTWRK']<99)

sample = df[m].copy()
print(f"Sample N = {len(sample)} (paper: 4367)")

# Tenure bins
sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1, 5, 10, 20, 100], labels=['0-5', '6-10', '11-20', '21+'])
sample['plant_closing'] = (sample['DWREAS'] == 1).astype(int)
sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']

# Deflation
deflator = {1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
            1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8}

sample['def_cur'] = sample['YEAR'].map(deflator)
sample['def_pri'] = sample['disp_year'].map(deflator)
valid_defl = sample['def_pri'].notna()
ws = sample[valid_defl].copy()
ws['log_wc'] = np.log(ws['DWWEEKC']/ws['def_cur']) - np.log(ws['DWWEEKL']/ws['def_pri'])

print(f"Valid deflated wages: N = {len(ws)}")

# Valid weeks unemployed
us = sample[sample['DWWKSUN'] < 999].copy()
print(f"Valid weeks unemp: N = {len(us)}")

# Compute stats - UNWEIGHTED
bins = ['0-5', '6-10', '11-20', '21+', 'Total']

print(f"\n=== UNWEIGHTED RESULTS ===")
print(f"\n{'Variable':<35} {'0-5':>10} {'6-10':>10} {'11-20':>10} {'21+':>10} {'Total':>10}")
print("-" * 85)

# Row 1: Log wage change
vals = []
ses = []
for b in bins:
    if b == 'Total':
        data = ws
    else:
        data = ws[ws['tenure_bin']==b]
    m_val = data['log_wc'].mean()
    se_val = data['log_wc'].std() / np.sqrt(len(data))
    vals.append(f"{m_val:.3f}")
    ses.append(f"({se_val:.3f})")
print(f"{'Avg change in log weekly wage':<35} " + " ".join(f"{v:>10}" for v in vals))
print(f"{'Paper values':<35}     -0.095     -0.223     -0.282     -0.439     -0.135")
print(f"{'':<35} " + " ".join(f"{v:>10}" for v in ses))
print(f"{'Paper SEs':<35}     (0.010)    (0.021)    (0.026)    (0.071)    (0.009)")

# Row 2: Plant closing
vals = []
ses = []
for b in bins:
    if b == 'Total':
        data = sample
    else:
        data = sample[sample['tenure_bin']==b]
    m_val = data['plant_closing'].mean()
    se_val = np.sqrt(m_val * (1-m_val) / len(data))
    vals.append(f"{m_val:.3f}")
    ses.append(f"({se_val:.3f})")
print(f"\n{'Pct displaced by plant closing':<35} " + " ".join(f"{v:>10}" for v in vals))
print(f"{'Paper values':<35}      0.352      0.463      0.528      0.750      0.390")
print(f"{'':<35} " + " ".join(f"{v:>10}" for v in ses))
print(f"{'Paper SEs':<35}     (0.008)    (0.021)    (0.026)    (0.043)    (0.007)")

# Row 3: Weeks unemployed
vals = []
ses = []
for b in bins:
    if b == 'Total':
        data = us
    else:
        data = us[us['tenure_bin']==b]
    m_val = data['DWWKSUN'].mean()
    se_val = data['DWWKSUN'].std() / np.sqrt(len(data))
    vals.append(f"{m_val:.2f}")
    ses.append(f"({se_val:.3f})")
print(f"\n{'Weeks unemployed':<35} " + " ".join(f"{v:>10}" for v in vals))
print(f"{'Paper values':<35}      18.69      24.54      26.66      31.79      20.41")
print(f"{'':<35} " + " ".join(f"{v:>10}" for v in ses))
print(f"{'Paper SEs':<35}     (0.413)    (1.202)    (1.536)    (3.288)    (0.385)")

# Sample sizes by bin
print(f"\nSample sizes:")
for b in bins:
    if b == 'Total':
        print(f"  {b}: {len(sample)}")
    else:
        print(f"  {b}: {len(sample[sample['tenure_bin']==b])}")

# Now try WEIGHTED
print(f"\n\n=== WEIGHTED RESULTS ===")
print(f"\n{'Variable':<35} {'0-5':>10} {'6-10':>10} {'11-20':>10} {'21+':>10} {'Total':>10}")
print("-" * 85)

# Row 1 weighted
vals = []
ses = []
for b in bins:
    if b == 'Total':
        data = ws
    else:
        data = ws[ws['tenure_bin']==b]
    w = data['DWSUPPWT']
    m_val = np.average(data['log_wc'], weights=w)
    var = np.average((data['log_wc'] - m_val)**2, weights=w)
    n_eff = w.sum()**2 / (w**2).sum()
    se_val = np.sqrt(var / n_eff)
    vals.append(f"{m_val:.3f}")
    ses.append(f"({se_val:.3f})")
print(f"{'Avg change in log weekly wage':<35} " + " ".join(f"{v:>10}" for v in vals))
print(f"{'Paper values':<35}     -0.095     -0.223     -0.282     -0.439     -0.135")
print(f"{'':<35} " + " ".join(f"{v:>10}" for v in ses))
print(f"{'Paper SEs':<35}     (0.010)    (0.021)    (0.026)    (0.071)    (0.009)")

# Row 2 weighted
vals = []
ses = []
for b in bins:
    if b == 'Total':
        data = sample
    else:
        data = sample[sample['tenure_bin']==b]
    w = data['DWSUPPWT']
    m_val = np.average(data['plant_closing'], weights=w)
    n_eff = w.sum()**2 / (w**2).sum()
    se_val = np.sqrt(m_val * (1-m_val) / n_eff)
    vals.append(f"{m_val:.3f}")
    ses.append(f"({se_val:.3f})")
print(f"\n{'Pct displaced by plant closing':<35} " + " ".join(f"{v:>10}" for v in vals))
print(f"{'Paper values':<35}      0.352      0.463      0.528      0.750      0.390")
print(f"{'':<35} " + " ".join(f"{v:>10}" for v in ses))
print(f"{'Paper SEs':<35}     (0.008)    (0.021)    (0.026)    (0.043)    (0.007)")

# Row 3 weighted
vals = []
ses = []
for b in bins:
    if b == 'Total':
        data = us
    else:
        data = us[us['tenure_bin']==b]
    w = data['DWSUPPWT']
    m_val = np.average(data['DWWKSUN'], weights=w)
    var = np.average((data['DWWKSUN'] - m_val)**2, weights=w)
    n_eff = w.sum()**2 / (w**2).sum()
    se_val = np.sqrt(var / n_eff)
    vals.append(f"{m_val:.2f}")
    ses.append(f"({se_val:.3f})")
print(f"\n{'Weeks unemployed':<35} " + " ".join(f"{v:>10}" for v in vals))
print(f"{'Paper values':<35}      18.69      24.54      26.66      31.79      20.41")
print(f"{'':<35} " + " ".join(f"{v:>10}" for v in ses))
print(f"{'Paper SEs':<35}     (0.413)    (1.202)    (1.536)    (3.288)    (0.385)")

# Also try: maybe don't require valid wages for the sample
# The paper may define the sample broadly and then compute
# wages only for those with valid data

# Try: DWREAS 1-6, full-time, valid DWLASTWRK (no wage requirement)
m2 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
     (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & \
     (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & (df['DWLASTWRK']<99)
s2 = df[m2]
print(f"\n\n=== Alternative: ft + DWLASTWRK<99 (no wage req) ===")
print(f"N = {len(s2)} (paper: 4367)")

# Check plant closing by bin
s2_copy = s2.copy()
s2_copy['tenure_bin'] = pd.cut(s2_copy['DWYEARS'], bins=[-0.1, 5, 10, 20, 100], labels=['0-5', '6-10', '11-20', '21+'])
s2_copy['plant_closing'] = (s2_copy['DWREAS'] == 1).astype(int)
print(f"Plant closing: {s2_copy['plant_closing'].mean():.3f}")
for b in ['0-5', '6-10', '11-20', '21+']:
    sub = s2_copy[s2_copy['tenure_bin']==b]
    print(f"  {b}: {sub['plant_closing'].mean():.3f} (N={len(sub)})")
