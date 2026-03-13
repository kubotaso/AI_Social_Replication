#!/usr/bin/env python3
"""Better tenure reconstruction using all available data."""
import pandas as pd, numpy as np

df = pd.read_csv('data/psid_panel.csv')
EDUC_CAT = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17}
df['education_years'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT)
df = df[(df['age'] >= 18) & (df['age'] <= 60)]
df = df[df['govt_worker'] != 1]
df = df[df['self_employed'] != 1]
df = df[df['agriculture'] != 1]
df = df[df['hourly_wage'] > 0]
df = df[df['education_years'].notna()]
df = df[df['hourly_wage'] < 200]

# Tenure category to years mapping (for 1971-1972)
TENURE_CAT = {
    0: 0.5,  # less than 1 year -> 0.5
    1: 0.25, # 1 to 5 months -> 0.25
    2: 0.75, # 6 to 11 months -> 0.75
    3: 1.5,  # 1 to less than 2 years -> 1.5
    4: 3.5,  # 2 to less than 5 years -> 3.5
    5: 7.0,  # 5 to less than 10 years -> 7.0
    6: 14.5, # 10 to less than 20 years -> 14.5
    7: 25.0, # 20 or more years -> 25.0
    9: np.nan # NA/DK
}

# Better reconstruction strategy:
# 1. For each job, collect ALL tenure observations from all sources
# 2. Find the "best anchor" - the observation with the most reliable tenure
# 3. Extrapolate forward/backward from that anchor

anchor_data = {}

for jid in df['job_id'].unique():
    job = df[df['job_id'] == jid].sort_values('year')
    tenure_obs = []

    for _, row in job.iterrows():
        yr = int(row['year'])

        # Source 1: tenure_mos (months on current job) - available 1976, 1980-1983
        if pd.notna(row.get('tenure_mos', np.nan)):
            mos = row['tenure_mos']
            if 0 < mos < 900 and yr != 1977:  # 1977 tenure_mos is all 0
                tenure_obs.append((yr, mos / 12.0, 'mos'))

        # Source 2: tenure column
        if pd.notna(row.get('tenure', np.nan)):
            t = row['tenure']
            if yr in [1971, 1972]:
                # Categorical codes
                val = TENURE_CAT.get(int(t), np.nan)
                if not np.isnan(val):
                    tenure_obs.append((yr, val, 'cat'))
            elif yr == 1976:
                # In 1976, 'tenure' contains months on job
                if 0 < t < 900:
                    tenure_obs.append((yr, t / 12.0, 'raw76'))

    if tenure_obs:
        # Prefer the most precise observation: tenure_mos from 1980+ > 1976 raw > categorical
        # Among same type, prefer highest value (= most time on job = better anchor)
        # Actually for Topel's reconstruction, use "maximum reported tenure" as anchor
        best = max(tenure_obs, key=lambda x: x[1])
        anchor_data[jid] = {'anchor_tenure': best[1], 'anchor_year': best[0], 'source': best[2]}
    else:
        anchor_data[jid] = {'anchor_tenure': np.nan, 'anchor_year': np.nan, 'source': 'none'}

# Apply anchors
df['tenure_recon'] = np.nan
for jid in df['job_id'].unique():
    job_mask = df['job_id'] == jid
    info = anchor_data[jid]
    if np.isnan(info['anchor_tenure']):
        # No anchor - use tenure_topel
        df.loc[job_mask, 'tenure_recon'] = df.loc[job_mask, 'tenure_topel']
    else:
        # Extrapolate from anchor
        df.loc[job_mask, 'tenure_recon'] = info['anchor_tenure'] + (df.loc[job_mask, 'year'] - info['anchor_year'])
        df.loc[job_mask & (df['tenure_recon'] < 0), 'tenure_recon'] = 0

print(f"Before tenure>=1 filter:")
print(f"  N={len(df)}")
print(f"  tenure_recon: mean={df['tenure_recon'].mean():.3f}, sd={df['tenure_recon'].std(ddof=0):.3f}")

# Apply tenure >= 1 filter
df = df[df['tenure_recon'] >= 1].copy()
print(f"\nAfter tenure>=1 filter:")
print(f"  N={len(df)}")
print(f"  tenure_recon: mean={df['tenure_recon'].mean():.3f}, sd={df['tenure_recon'].std(ddof=0):.3f}")
print(f"  Target: mean=9.978, sd=8.944")

# Check year distribution
print(f"\nYear counts:")
for yr, cnt in df['year'].value_counts().sort_index().items():
    print(f"  {yr}: {cnt} ({cnt/len(df):.3f})")

# Now check with more detailed anchor strategy
# Some jobs that span many years might have tenure_topel = 1,2,...,13
# but actual tenure at start could be much higher
# Let's see: jobs where tenure_topel starts at 1 in 1971 have been
# with employer since at least 1970. If raw tenure in 1971 says code 5 (5-10 yrs),
# then starting tenure could be 7, not 1.

# Check how many jobs have NO anchor at all
n_no_anchor = sum(1 for v in anchor_data.values() if np.isnan(v['anchor_tenure']))
n_total_jobs = len(anchor_data)
print(f"\nJobs with no tenure anchor: {n_no_anchor} / {n_total_jobs}")

# Check anchor source distribution
source_counts = {}
for v in anchor_data.values():
    s = v['source']
    source_counts[s] = source_counts.get(s, 0) + 1
print(f"Anchor sources: {source_counts}")
