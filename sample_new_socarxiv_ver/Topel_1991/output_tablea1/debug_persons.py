#!/usr/bin/env python3
"""Debug: Analyze person count discrepancy."""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, 'data', 'psid_panel.csv'))

# Education recode
EDUC = {0:0, 1:3, 2:7, 3:10, 4:12, 5:12, 6:14, 7:16, 8:17}
df['edu_yrs'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'edu_yrs'] = df.loc[cat_mask, 'education_clean'].map(EDUC)

# Experience
df['exp'] = df['age'] - df['edu_yrs'] - 6
df.loc[df['exp'] < 0, 'exp'] = 0

# Hourly wage
df['hw'] = df['wages'] / df['hours']
bad = ~((df['hw'] > 0) & np.isfinite(df['hw']))
df.loc[bad, 'hw'] = df.loc[bad, 'hourly_wage']

# Restrictions
r = df.copy()
r = r[(r['age'] >= 18) & (r['age'] <= 60)]
r = r[r['govt_worker'] != 1]
r = r[r['self_employed'] != 1]
r = r[r['agriculture'] != 1]
r = r[r['hw'] > 0]
r = r[r['hw'] < 200]
r = r[r['edu_yrs'].notna()]
r = r[r['tenure_topel'] >= 1]

print(f'After restrictions: {len(r)} obs, {r["person_id"].nunique()} persons')

# Person appearance after restrictions
a = r.groupby('person_id').size()
print('\nPost-restriction appearances:')
for i in range(1, 14):
    n = (a == i).sum()
    print(f'  {i} years: {n} persons')
print(f'  Mean: {a.mean():.1f}')

# Person number analysis
pn = r['person_id'] % 1000
print(f'\npn distribution:')
print(f'  pn == 1 (head): {(pn == 1).sum()} obs, {r[pn==1]["person_id"].nunique()} persons')
print(f'  pn < 20: {(pn < 20).sum()} obs, {r[pn<20]["person_id"].nunique()} persons')
print(f'  pn < 50: {(pn < 50).sum()} obs, {r[pn<50]["person_id"].nunique()} persons')
print(f'  pn < 170: {(pn < 170).sum()} obs, {r[pn<170]["person_id"].nunique()} persons')

# Try head of household only (pn == 1)
r_head = r[pn == 1]
a_head = r_head.groupby('person_id').size()
print(f'\nHead only (pn==1): {len(r_head)} obs, {r_head["person_id"].nunique()} persons, mean appearances: {a_head.mean():.1f}')

# Year distribution for heads
yc = r_head['year'].value_counts().sort_index()
print('\nYear distribution (heads only):')
for yr in range(1971, 1984):
    n = yc.get(yr, 0)
    pct = n / len(r_head) if len(r_head) > 0 else 0
    print(f'  {yr}: {n} ({pct:.3f})')

# Try requiring at least 2 consecutive observations
print('\n--- Trying different min-appearance thresholds ---')
for min_app in [2, 3, 4, 5, 6, 7, 8]:
    keep = a[a >= min_app].index
    sub = r[r['person_id'].isin(keep)]
    print(f'  >= {min_app} appearances: {len(sub)} obs, {sub["person_id"].nunique()} persons')
