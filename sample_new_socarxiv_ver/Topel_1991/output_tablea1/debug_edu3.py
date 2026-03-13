#!/usr/bin/env python3
import pandas as pd, numpy as np

df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

# Current mapping
EDUC_OLD = {0:0, 1:3, 2:7, 3:10, 4:12, 5:12, 6:14, 7:16, 8:17}
# With code 5 = 14 (some college)
EDUC_NEW = {0:0, 1:3, 2:7, 3:10, 4:12, 5:14, 6:16, 7:17, 8:17}

# Apply both
df['edu_old'] = df['education_clean'].copy()
df['edu_new'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'edu_old'] = df.loc[cat_mask, 'education_clean'].map(EDUC_OLD)
df.loc[cat_mask, 'edu_new'] = df.loc[cat_mask, 'education_clean'].map(EDUC_NEW)

df['exp_old'] = (df['age'] - df['edu_old'] - 6).clip(lower=0)
df['exp_new'] = (df['age'] - df['edu_new'] - 6).clip(lower=0)

# Apply restrictions
r = df.copy()
r = r[(r['age'] >= 18) & (r['age'] <= 60)]
r = r[r['govt_worker'] != 1]
r = r[r['self_employed'] != 1]
r = r[r['agriculture'] != 1]
r['hw'] = r['wages'] / r['hours']
bad = ~((r['hw'] > 0) & np.isfinite(r['hw']))
r.loc[bad, 'hw'] = r.loc[bad, 'hourly_wage']
r = r[r['hw'] > 0]
r = r[r['hw'] < 200]
r = r[r['edu_old'].notna()]
r = r[np.isfinite(np.log(r['hw']))]

# With old education
print("OLD mapping (code 5 = 12):")
print(f"  Education mean: {r['edu_old'].mean():.3f} (target 12.645)")
print(f"  Experience mean: {r['exp_old'].mean():.3f} (target 20.021)")
print(f"  Education SD: {r['edu_old'].std(ddof=0):.3f} (target 2.809)")
print(f"  Experience SD: {r['exp_old'].std(ddof=0):.3f} (target 11.045)")

print("\nNEW mapping (code 5 = 14):")
print(f"  Education mean: {r['edu_new'].mean():.3f} (target 12.645)")
print(f"  Experience mean: {r['exp_new'].mean():.3f} (target 20.021)")
print(f"  Education SD: {r['edu_new'].std(ddof=0):.3f} (target 2.809)")
print(f"  Experience SD: {r['exp_new'].std(ddof=0):.3f} (target 11.045)")

# Check how many have code 5
for yr in [1971, 1972, 1973, 1974]:
    sub = r[r['year'] == yr]
    n5 = (sub['education_clean'] == 5).sum()
    print(f"  Year {yr}: code 5 count = {n5} / {len(sub)}")
