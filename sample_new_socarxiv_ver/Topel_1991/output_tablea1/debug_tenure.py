#!/usr/bin/env python3
"""Debug: Compare tenure reconstruction approaches."""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, 'data', 'psid_panel.csv'))

EDUC = {0:0, 1:3, 2:7, 3:10, 4:12, 5:12, 6:14, 7:16, 8:17}
TENURE_CAT = {0: 0.5, 1: 0.25, 2: 0.75, 3: 1.5, 4: 3.5, 5: 7.0, 6: 14.5, 7: 25.0, 9: np.nan}

# Education
df['edu_yrs'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'edu_yrs'] = df.loc[cat_mask, 'education_clean'].map(EDUC)

# Experience
df['exp'] = (df['age'] - df['edu_yrs'] - 6).clip(lower=0)

# Hourly wage
df['hw'] = df['wages'] / df['hours']
bad = ~((df['hw'] > 0) & np.isfinite(df['hw']))
df.loc[bad, 'hw'] = df.loc[bad, 'hourly_wage']

# Apply basic restrictions
r = df.copy()
r = r[(r['age'] >= 18) & (r['age'] <= 60)]
r = r[r['govt_worker'] != 1]
r = r[r['self_employed'] != 1]
r = r[r['agriculture'] != 1]
r = r[r['hw'] > 0]
r = r[r['hw'] < 200]
r = r[r['edu_yrs'].notna()]
r = r[np.isfinite(np.log(r['hw']))]

# Compare tenure approaches
# 1. tenure_topel (from build script)
tt = r['tenure_topel']
print(f"tenure_topel: mean={tt.mean():.3f}, sd={tt.std(ddof=0):.3f}, min={tt.min()}, max={tt.max()}")
print(f"  >= 1: {(tt >= 1).sum()} obs")

# 2. Raw tenure variable
t_raw = r['tenure']
print(f"\nraw tenure: mean={t_raw.dropna().mean():.3f}, sd={t_raw.dropna().std(ddof=0):.3f}")

# 3. tenure_mos
tm = r['tenure_mos']
print(f"tenure_mos: mean={tm.dropna().mean():.3f}, sd={tm.dropna().std(ddof=0):.3f}")
for yr in [1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983]:
    sub = r[r['year'] == yr]['tenure_mos']
    n_valid = sub.notna().sum()
    if n_valid > 0:
        print(f"  {yr}: {n_valid} valid, mean={sub.dropna().mean():.1f}")

# 4. Check tenure_topel statistics after >=1 filter
r1 = r[r['tenure_topel'] >= 1]
print(f"\nAfter tenure_topel >= 1: {len(r1)} obs, {r1['person_id'].nunique()} persons")
print(f"  tenure_topel mean={r1['tenure_topel'].mean():.3f}, sd={r1['tenure_topel'].std(ddof=0):.3f}")
print(f"  experience mean={r1['exp'].mean():.3f}, sd={r1['exp'].std(ddof=0):.3f}")

# 5. What about using tenure_topel directly (without reconstruction)?
# This is the simplest approach
print("\n--- Using tenure_topel directly ---")
for min_t in [1, 1.5, 2]:
    sub = r[r['tenure_topel'] >= min_t]
    print(f"  tenure_topel >= {min_t}: N={len(sub)}, persons={sub['person_id'].nunique()}, "
          f"tenure_mean={sub['tenure_topel'].mean():.3f}, tenure_sd={sub['tenure_topel'].std(ddof=0):.3f}")

# 6. Analyze job_id patterns
print("\n--- Job analysis ---")
jobs = r.groupby('job_id').agg(
    n_obs=('year', 'size'),
    min_year=('year', 'min'),
    max_year=('year', 'max'),
    mean_tenure=('tenure_topel', 'mean')
).reset_index()
print(f"Total jobs: {len(jobs)}")
print(f"Jobs with 1+ obs at tenure >= 1: {(jobs['mean_tenure'] >= 1).sum()}")
print(f"Mean tenure_topel across all jobs: {jobs['mean_tenure'].mean():.3f}")

# 7. What if we DON'T use reconstructed tenure but just tenure_topel?
# And compare to our reconstructed version
print("\n--- Comparing approaches for tenure >= 1 restricted sample ---")
r2 = r[r['tenure_topel'] >= 1].copy()
print(f"Using tenure_topel: N={len(r2)}, mean={r2['tenure_topel'].mean():.3f}, sd={r2['tenure_topel'].std(ddof=0):.3f}")

# Now try a different anchor strategy: use the EARLIEST anchor point
# (paper says "starting tenure gauged from period of MAX reported tenure")
from collections import defaultdict
anchor_earliest = {}
anchor_latest = {}
anchor_max = {}

for jid in r['job_id'].unique():
    job = r[r['job_id'] == jid].sort_values('year')
    obs = []
    for _, row in job.iterrows():
        yr = int(row['year'])
        if pd.notna(row.get('tenure_mos', np.nan)):
            mos = row['tenure_mos']
            if 0 < mos < 900 and yr != 1977:
                obs.append((yr, mos / 12.0))
        if pd.notna(row.get('tenure', np.nan)):
            t = row['tenure']
            if yr in [1971, 1972]:
                val = TENURE_CAT.get(int(t), np.nan)
                if pd.notna(val):
                    obs.append((yr, val))
            elif yr == 1976:
                if 0 < t < 900:
                    obs.append((yr, t / 12.0))
    if obs:
        earliest = min(obs, key=lambda x: x[0])
        latest = max(obs, key=lambda x: x[0])
        maxval = max(obs, key=lambda x: x[1])
        anchor_earliest[jid] = (earliest[1], earliest[0])
        anchor_latest[jid] = (latest[1], latest[0])
        anchor_max[jid] = (maxval[1], maxval[0])

# Compute tenure using each strategy
for name, anchors in [('earliest', anchor_earliest), ('latest', anchor_latest), ('max_value', anchor_max)]:
    r2 = r.copy()
    r2['ten'] = np.nan
    for jid, (at, ay) in anchors.items():
        mask = r2['job_id'] == jid
        r2.loc[mask, 'ten'] = at + (r2.loc[mask, 'year'] - ay)
    # Fill missing with tenure_topel
    r2.loc[r2['ten'].isna(), 'ten'] = r2.loc[r2['ten'].isna(), 'tenure_topel']
    r2.loc[r2['ten'] < 0, 'ten'] = 0
    sub = r2[r2['ten'] >= 1]
    print(f"  {name}: N={len(sub)}, mean={sub['ten'].mean():.3f}, sd={sub['ten'].std(ddof=0):.3f}")
