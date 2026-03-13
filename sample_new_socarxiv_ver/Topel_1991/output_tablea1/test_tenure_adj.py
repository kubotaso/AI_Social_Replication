#!/usr/bin/env python3
"""Test different tenure reconstruction strategies."""
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

TENURE_CAT = {0:0.5,1:0.25,2:0.75,3:1.5,4:3.5,5:7.0,6:14.5,7:25.0,9:np.nan}

# Current approach: anchor from tenure_mos/raw_tenure, fallback to tenure_topel
# The issue: jobs without anchors use tenure_topel which starts at 1 for first panel year
# But those jobs could have been going on for years before the panel

# Strategy 1: For unanchored jobs, add estimated prior tenure
# based on the first year the person appears in the panel
# If first year is 1971 and they were 40 years old with 12 years education,
# they had 22 years experience, and the job could have been going on for a while

# Strategy 2: Use tenure_mos more aggressively for years where it's available
# 1976: tenure column has months (but wait, we already use that)
# 1980-1983: tenure_mos has months

# Strategy 3: For the 1976 "tenure" column which seems to have both
# months and raw codes, be more careful

# Let me check what job_ids have no anchor and how many obs they contribute
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

n_no_anchor = sum(1 for v in anchor_data.values() if len(v) == 0)
n_with_anchor = sum(1 for v in anchor_data.values() if len(v) > 0)
print(f"Jobs with anchor: {n_with_anchor}, without: {n_no_anchor}")

# Count obs in unanchored jobs
unanchored_jobs = [jid for jid, obs in anchor_data.items() if len(obs) == 0]
n_unanchored_obs = len(df[df['job_id'].isin(unanchored_jobs)])
print(f"Obs in unanchored jobs: {n_unanchored_obs} / {len(df)}")

# What's the tenure_topel distribution for unanchored jobs?
un_df = df[df['job_id'].isin(unanchored_jobs)]
print(f"Unanchored job tenure_topel: mean={un_df['tenure_topel'].mean():.3f}, sd={un_df['tenure_topel'].std():.3f}")

# How about the duration of unanchored jobs?
un_dur = un_df.groupby('job_id').agg({'year': ['min', 'max', 'count']})
un_dur.columns = ['first_year', 'last_year', 'n_years']
un_dur['span'] = un_dur['last_year'] - un_dur['first_year'] + 1
print(f"Unanchored job spans: mean={un_dur['span'].mean():.1f}, max={un_dur['span'].max()}")
print(f"  first_year distribution:")
print(un_dur['first_year'].value_counts().sort_index())

# Now test: what if for unanchored jobs starting at 1971 (first panel year),
# we assume the person had been in the job for at least some time before?
# Average tenure at panel start would be roughly half the average total tenure
# For Topel's data, mean tenure=10 years, so prior tenure could be ~5 years

# Let's try different offsets
for offset in [0, 1, 2, 3, 4, 5]:
    df_t = df.copy()
    for jid in df_t['job_id'].unique():
        job_mask = df_t['job_id'] == jid
        obs = anchor_data[jid]
        if len(obs) == 0:
            # Add offset for jobs starting at first panel year
            first_yr = df_t.loc[job_mask, 'year'].min()
            if first_yr == 1971:
                df_t.loc[job_mask, 'tenure_adj'] = df_t.loc[job_mask, 'tenure_topel'] + offset
            else:
                df_t.loc[job_mask, 'tenure_adj'] = df_t.loc[job_mask, 'tenure_topel']
        else:
            best = max(obs, key=lambda x: x[1])
            df_t.loc[job_mask, 'tenure_adj'] = best[1] + (df_t.loc[job_mask, 'year'] - best[0])
            df_t.loc[job_mask & (df_t['tenure_adj'] < 0), 'tenure_adj'] = 0

    df_filt = df_t[df_t['tenure_adj'] >= 1]
    print(f"  Offset {offset}: tenure mean={df_filt['tenure_adj'].mean():.3f}, "
          f"sd={df_filt['tenure_adj'].std(ddof=0):.3f}, N={len(df_filt)}")
