#!/usr/bin/env python3
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

# Check tenure_topel and raw tenure
print("tenure_topel stats:")
print(f"  mean={df['tenure_topel'].mean():.3f}, median={df['tenure_topel'].median():.1f}")
print(f"  max={df['tenure_topel'].max()}, min={df['tenure_topel'].min()}")
print(f"  tenure_topel >= 1: {(df['tenure_topel'] >= 1).sum()}")

print("\nraw tenure stats:")
print(f"  mean={df['tenure'].mean():.3f}, median={df['tenure'].median():.1f}")
print(f"  max={df['tenure'].max()}, min={df['tenure'].min()}")

print("\ntenure_mos stats:")
mask_tm = df['tenure_mos'].notna()
if mask_tm.any():
    tm = df.loc[mask_tm, 'tenure_mos']
    tm_years = tm / 12.0
    print(f"  mean (months)={tm.mean():.1f}, mean (years)={tm_years.mean():.3f}")
    print(f"  N with tenure_mos: {mask_tm.sum()}")

# Check the 'tenure' column by year - what does it contain?
print("\nTenure by year:")
for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    t = sub['tenure'].dropna()
    print(f"  {yr}: N={len(t)}, mean={t.mean():.2f}, min={t.min():.1f}, max={t.max():.1f}, unique_vals={len(t.unique())}")

# Check tenure_mos by year
print("\nTenure_mos by year:")
for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    t = sub['tenure_mos'].dropna()
    if len(t) > 0:
        print(f"  {yr}: N={len(t)}, mean={t.mean():.1f} mos ({t.mean()/12:.1f} yrs), min={t.min()}, max={t.max()}")
    else:
        print(f"  {yr}: no data")

# Try to reconstruct proper tenure
# The paper says tenure is "reconstructed" - for jobs starting during panel,
# tenure starts at 0; for jobs in progress at start, starting tenure gauged
# from period of maximum reported tenure.

# tenure_topel appears to be years within panel starting from 0
# We need to add the initial tenure for jobs already in progress

# Let's check the actual tenure data structure
print("\nJob tenure analysis:")
for jid in df['job_id'].unique()[:20]:
    job = df[df['job_id'] == jid].sort_values('year')
    years = job['year'].values
    topel = job['tenure_topel'].values
    raw = job['tenure'].values
    mos = job['tenure_mos'].values
    print(f"  job {jid}: years={list(years)}, topel={list(topel)}, raw_ten={list(raw[:5])}, mos={list(mos[:5])}")

# The key insight: tenure_topel starts from 0 for first year in panel,
# then increments by 1 each year. But it doesn't account for prior
# tenure if the job was already in progress when person entered the panel.
# We need to add a "starting tenure" offset based on raw tenure data.

# Let's try to construct proper tenure:
# For each job, find the maximum reported tenure (in months from tenure_mos)
# and use that to anchor the tenure level.

TENURE_CAT = {0: 0.04, 1: 0.25, 2: 0.75, 3: 1.5, 4: 3.5, 5: 7.0, 6: 14.5, 7: 25.0, 9: np.nan}

df_t = df.copy()
anchor_data = []

for jid in df_t['job_id'].unique():
    job = df_t[df_t['job_id'] == jid].sort_values('year')
    tenure_obs = []

    # Tenure months from later years (1976+)
    for _, row in job.iterrows():
        if pd.notna(row.get('tenure_mos', np.nan)):
            mos = row['tenure_mos']
            yr = int(row['year'])
            if 0 < mos < 900:
                tenure_obs.append((yr, mos / 12.0))

    # Categorical tenure from 1971-1972
    for _, row in job[job['year'].isin([1971, 1972])].iterrows():
        if pd.notna(row.get('tenure', np.nan)):
            code = int(row['tenure'])
            val = TENURE_CAT.get(code, np.nan)
            if not np.isnan(val):
                tenure_obs.append((int(row['year']), val))

    if tenure_obs:
        # Use the observation with maximum tenure as anchor
        best = max(tenure_obs, key=lambda x: x[1])
        anchor_data.append({'job_id': jid, 'anchor_tenure': best[1], 'anchor_year': best[0]})
    else:
        anchor_data.append({'job_id': jid, 'anchor_tenure': np.nan, 'anchor_year': np.nan})

anchor_df = pd.DataFrame(anchor_data)
df_t = df_t.merge(anchor_df, on='job_id', how='left')
df_t['tenure_recon'] = df_t['anchor_tenure'] + (df_t['year'] - df_t['anchor_year'])
df_t['tenure_recon'] = df_t['tenure_recon'].clip(lower=0)

# Where no anchor, fall back to tenure_topel
no_anchor = df_t['tenure_recon'].isna()
df_t.loc[no_anchor, 'tenure_recon'] = df_t.loc[no_anchor, 'tenure_topel']

print(f"\nReconstructed tenure: mean={df_t['tenure_recon'].mean():.3f}, sd={df_t['tenure_recon'].std(ddof=0):.3f}")
print(f"Target: mean=9.978, sd=8.944")

# Filter tenure >= 1
df_t = df_t[df_t['tenure_recon'] >= 1]
print(f"After tenure >= 1: N={len(df_t)}")
print(f"  tenure_recon: mean={df_t['tenure_recon'].mean():.3f}, sd={df_t['tenure_recon'].std(ddof=0):.3f}")
