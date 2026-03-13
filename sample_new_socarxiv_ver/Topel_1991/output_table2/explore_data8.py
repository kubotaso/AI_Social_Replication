#!/usr/bin/env python3
"""Check raw tenure distributions."""
import pandas as pd
df = pd.read_csv('data/psid_panel_full.csv')

for y in [1970, 1971, 1972]:
    sub = df[df['year']==y]
    t = sub['tenure'].dropna()
    print(f'Year {y}:')
    print(t.value_counts().sort_index())
    print()

# Check tenure_mos in 1976
sub76 = df[df['year']==1976]
tm = sub76['tenure_mos'].dropna()
tm_valid = tm[tm < 900]
print(f'Year 1976 tenure_mos valid:')
print(tm_valid.describe())
print()

# Cross-tabulate: for persons observed in both 1972 and 1976
# with same job, compare the raw tenure (bracket) in 1972 with tenure_mos in 1976
df_sorted = df.sort_values(['person_id', 'job_id', 'year'])

# Find persons with same job in 1972 and 1976
obs_72 = df_sorted[(df_sorted['year']==1972)][['person_id','job_id','tenure','tenure_topel']].copy()
obs_76 = df_sorted[(df_sorted['year']==1976)][['person_id','job_id','tenure_mos','tenure_topel']].copy()

merged = obs_72.merge(obs_76, on=['person_id','job_id'], suffixes=('_72','_76'))
merged = merged[(merged['tenure_mos'] < 900) & (merged['tenure_mos'].notna())]
merged = merged[(merged['tenure'] < 900) & (merged['tenure'].notna())]

print(f'Matched person-jobs (1972 to 1976): {len(merged)}')
if len(merged) > 0:
    merged['tenure_mos_yrs'] = merged['tenure_mos'] / 12.0
    print(f'Raw tenure 1972 vs tenure_mos/12 1976:')
    for t72_val in sorted(merged['tenure'].unique()):
        sub = merged[merged['tenure']==t72_val]
        print(f'  raw_tenure_72={t72_val:.0f}: n={len(sub)}, mean_tenure_mos_yrs_76={sub["tenure_mos_yrs"].mean():.1f}, min={sub["tenure_mos_yrs"].min():.1f}, max={sub["tenure_mos_yrs"].max():.1f}')
