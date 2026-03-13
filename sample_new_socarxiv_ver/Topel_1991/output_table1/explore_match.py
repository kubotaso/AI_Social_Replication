import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# The filter that gives N=4368 (almost exactly 4367):
# DWREAS 1-3 + private + fulltime + DWYEARS>=1
base = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
       df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & \
       (df['DWYEARS']<99) & (df['DWYEARS']>=1) & \
       (df['DWFULLTIME']==2) & (df['DWCLASS'].isin([1,2]))

sample = df[base].copy()
print(f"Sample N = {len(sample)}")
print(f"Plant closing pct = {(sample['DWREAS']==1).mean():.3f} (paper: .390)")

# That gives 0.476, not .390. Too high.
# Let me also check the "fulltime + DWYEARS>=1" combination (N=4414)
base2 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
        df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & \
        (df['DWYEARS']<99) & (df['DWYEARS']>=1) & (df['DWFULLTIME']==2)
s2 = df[base2].copy()
print(f"\nDWREAS 1-3 + FT + DWYEARS>=1: N={len(s2)}, pc={(s2['DWREAS']==1).mean():.3f}")

# The problem: plant closing % doesn't match .390 for any of these combos
# .390 matches the base DWREAS 1-6 sample (N=6738)
# Or maybe .390 matches DWREAS 1-3 with NO additional filters:
# DWREAS 1-3 base: N=5691, pc=0.460 -- too high

# Let me try: maybe the paper includes ALL displacement reasons (1-6)
# and defines "economic reasons" more broadly
# With DWREAS 1-6: N=6738, pc = (DWREAS==1)/total
# But N is too large

# Maybe DWREAS 1-6 + fulltime + DWYEARS>=1?
base3 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
        df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & \
        (df['DWYEARS']<99) & (df['DWYEARS']>=1) & (df['DWFULLTIME']==2)
s3 = df[base3].copy()
print(f"\nDWREAS 1-6 + FT + DWYEARS>=1: N={len(s3)}, pc={(s3['DWREAS']==1).mean():.3f}")

base3b = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & \
         (df['DWYEARS']<99) & (df['DWYEARS']>=1) & (df['DWFULLTIME']==2) & \
         (df['DWCLASS'].isin([1,2]))
s3b = df[base3b].copy()
print(f"DWREAS 1-6 + FT + DWYEARS>=1 + private: N={len(s3b)}, pc={(s3b['DWREAS']==1).mean():.3f}")

# What about DWREAS 1-6 + fulltime only
base3c = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & \
         (df['DWYEARS']<99) & (df['DWFULLTIME']==2)
s3c = df[base3c].copy()
print(f"DWREAS 1-6 + FT: N={len(s3c)}, pc={(s3c['DWREAS']==1).mean():.3f}")

# DWREAS 1-6 + private
base3d = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & \
         (df['DWYEARS']<99) & (df['DWCLASS'].isin([1,2]))
s3d = df[base3d].copy()
print(f"DWREAS 1-6 + private: N={len(s3d)}, pc={(s3d['DWREAS']==1).mean():.3f}")

# Hmm, let me look at weighted pc for various combos
print("\n=== WEIGHTED plant closing percentages ===")
for label, mask_expr in [
    ('DWREAS 1-3, base', (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99)),
    ('DWREAS 1-6, base', (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99)),
    ('DWREAS 1-3 + FT', (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99) & (df['DWFULLTIME']==2)),
    ('DWREAS 1-3 + FT + DWYEARS>=1', (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & (df['DWYEARS']>=1)),
]:
    sub = df[mask_expr]
    wpc = np.average(sub['DWREAS']==1, weights=sub['DWSUPPWT'])
    uwpc = (sub['DWREAS']==1).mean()
    print(f"  {label}: N={len(sub)}, unwt_pc={uwpc:.3f}, wt_pc={wpc:.3f}")

# Let me now compute all 3 rows for the closest N match and see
# which gets closest to the paper overall
print("\n\n=== Full computation for DWREAS 1-3, base (N=5691) ===")
m = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
    df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99)
sample = df[m].copy()
sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1,5,10,20,100], labels=['0-5','6-10','11-20','21+'])
sample['plant_closing'] = (sample['DWREAS']==1).astype(int)
sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']

deflator = {1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0,
            1982:100.0, 1983:103.9, 1984:107.7, 1985:110.9, 1986:113.8}

# Wage sample
vw = (sample['DWWEEKL']>0) & (sample['DWWEEKL']<9000) & \
     (sample['DWWEEKC']>0) & (sample['DWWEEKC']<9000) & (sample['DWLASTWRK']<99)
ws = sample[vw].copy()
ws['def_cur'] = ws['YEAR'].map(deflator)
ws['def_pri'] = ws['disp_year'].map(deflator)
ws = ws.dropna(subset=['def_cur','def_pri'])
ws['real_lwc'] = np.log(ws['DWWEEKC']/ws['def_cur']) - np.log(ws['DWWEEKL']/ws['def_pri'])

# Unemp sample
us = sample[sample['DWWKSUN']<999].copy()

bins_list = ['0-5','6-10','11-20','21+','Total']

# Unweighted
print(f"{'':30s} {'0-5':>8} {'6-10':>8} {'11-20':>8} {'21+':>8} {'Total':>8}")
for row, label in [('lwc','Avg chg log wkly wage'), ('pc','Pct plant closing'), ('wu','Weeks unemp')]:
    means = []
    ses = []
    for b in bins_list:
        if b == 'Total':
            if row == 'lwc': d = ws
            elif row == 'pc': d = sample
            else: d = us
        else:
            if row == 'lwc': d = ws[ws['tenure_bin']==b]
            elif row == 'pc': d = sample[sample['tenure_bin']==b]
            else: d = us[us['tenure_bin']==b]

        if row == 'lwc':
            m = d['real_lwc'].mean()
            se = d['real_lwc'].std() / np.sqrt(len(d))
        elif row == 'pc':
            m = d['plant_closing'].mean()
            se = np.sqrt(m*(1-m)/len(d))
        else:
            m = d['DWWKSUN'].mean()
            se = d['DWWKSUN'].std() / np.sqrt(len(d))
        means.append(f"{m:.3f}")
        ses.append(f"({se:.3f})")
    print(f"{label:30s} " + " ".join(f"{v:>8}" for v in means))
    print(f"{'':30s} " + " ".join(f"{v:>8}" for v in ses))

print("\nPaper values:")
print(f"{'Avg chg log wkly wage':30s}   -.095    -.223    -.282    -.439    -.135")
print(f"{'':30s}   (.010)   (.021)   (.026)   (.071)   (.009)")
print(f"{'Pct plant closing':30s}    .352     .463     .528     .750     .390")
print(f"{'':30s}   (.008)   (.021)   (.026)   (.043)   (.007)")
print(f"{'Weeks unemp':30s}   18.69    24.54    26.66    31.79    20.41")
print(f"{'':30s}   (.413)  (1.202)  (1.536)  (3.288)   (.385)")

print(f"\nN (wage): {len(ws)}, N (sample): {len(sample)}, N (unemp): {len(us)}")
