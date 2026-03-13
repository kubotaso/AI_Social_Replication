#!/usr/bin/env python3
"""Check tenure distribution details."""
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

# Check tenure_mos distribution for 1981-1983 (most reliable)
for yr in [1980, 1981, 1982, 1983]:
    sub = df[df['year'] == yr]
    tm = sub['tenure_mos'].dropna()
    tm_valid = tm[(tm > 0) & (tm < 900)]
    print(f"{yr}: tenure_mos N={len(tm_valid)}, mean={tm_valid.mean()/12:.1f} yrs, median={tm_valid.median()/12:.1f}")

# Check the distribution of reconstructed tenure by year
TENURE_CAT = {0:0.5,1:0.25,2:0.75,3:1.5,4:3.5,5:7.0,6:14.5,7:25.0,9:np.nan}

anchor_data = {}
for jid in df['job_id'].unique():
    job = df[df['job_id'] == jid].sort_values('year')
    tenure_obs = []
    for _, row in job.iterrows():
        yr = int(row['year'])
        if pd.notna(row.get('tenure_mos', np.nan)):
            mos = row['tenure_mos']
            if 0 < mos < 900 and yr != 1977:
                tenure_obs.append((yr, mos / 12.0))
        if pd.notna(row.get('tenure', np.nan)):
            t = row['tenure']
            if yr in [1971, 1972]:
                val = TENURE_CAT.get(int(t), np.nan)
                if not np.isnan(val):
                    tenure_obs.append((yr, val))
            elif yr == 1976:
                if 0 < t < 900:
                    tenure_obs.append((yr, t / 12.0))
    anchor_data[jid] = tenure_obs

# Apply anchors
df['tenure_recon'] = np.nan
for jid in df['job_id'].unique():
    job_mask = df['job_id'] == jid
    obs = anchor_data[jid]
    if len(obs) == 0:
        df.loc[job_mask, 'tenure_recon'] = df.loc[job_mask, 'tenure_topel']
    else:
        best = max(obs, key=lambda x: x[1])
        df.loc[job_mask, 'tenure_recon'] = best[1] + (df.loc[job_mask, 'year'] - best[0])
        df.loc[job_mask & (df['tenure_recon'] < 0), 'tenure_recon'] = 0

df_filt = df[df['tenure_recon'] >= 1]

print(f"\nReconstructed tenure by year:")
for yr in sorted(df_filt['year'].unique()):
    sub = df_filt[df_filt['year'] == yr]
    print(f"  {yr}: mean={sub['tenure_recon'].mean():.2f}, sd={sub['tenure_recon'].std():.2f}, N={len(sub)}")

print(f"\nOverall: mean={df_filt['tenure_recon'].mean():.3f}, sd={df_filt['tenure_recon'].std(ddof=0):.3f}")
print(f"Target: mean=9.978, sd=8.944")

# The paper's sample includes 1968-1983. Our sample is 1971-1983 only.
# The missing early years (1968-1970) would have lower average tenure
# (people are 3 years younger, 3 years less time to accumulate tenure).
# So actually, the missing years should LOWER the mean, not raise it.
# But the paper reports 9.978 INCLUDING those years. Our 9.712 is lower
# which is surprising since we're missing the low-tenure early years.

# This suggests our tenure reconstruction is systematically too low.
# Let's check: for jobs that span 1981-1983 (reliable tenure_mos),
# how does the anchor compare to what we'd get from tenure_topel?
print("\nComparison of anchor vs tenure_topel for jobs with 1981+ data:")
for jid in sorted(list(anchor_data.keys()))[:50]:
    obs = anchor_data[jid]
    if len(obs) == 0:
        continue
    job = df[df['job_id'] == jid].sort_values('year')
    if len(job) == 0:
        continue
    best = max(obs, key=lambda x: x[1])
    first_yr = job['year'].min()
    first_topel = job['tenure_topel'].iloc[0]
    implied_start = best[0] - best[1]  # year started job
    print(f"  job {jid}: anchor=({best[0]}, {best[1]:.1f}yr), first_panel_yr={first_yr}, "
          f"tenure_topel_start={first_topel}, implied_job_start={implied_start:.0f}")
    if len(list(filter(lambda x: True, obs))) > 3:
        break

# What if we use a weighted average of tenure_mos and tenure_topel extrapolation?
# For each job, if we have multiple tenure observations, check consistency
print("\nJobs with multiple tenure observations:")
n_inconsistent = 0
for jid in df['job_id'].unique():
    obs = anchor_data[jid]
    if len(obs) >= 2:
        # Check if observations are consistent (tenure increases by ~1/year)
        obs_sorted = sorted(obs, key=lambda x: x[0])
        for i in range(1, len(obs_sorted)):
            yr_diff = obs_sorted[i][0] - obs_sorted[i-1][0]
            ten_diff = obs_sorted[i][1] - obs_sorted[i-1][1]
            expected_diff = yr_diff  # should increase by 1/year
            if abs(ten_diff - expected_diff) > 2:
                n_inconsistent += 1
                if n_inconsistent <= 5:
                    print(f"  job {jid}: obs={obs_sorted}, "
                          f"yr_diff={yr_diff}, ten_diff={ten_diff:.1f}")
                break

print(f"Total inconsistent jobs: {n_inconsistent}")
