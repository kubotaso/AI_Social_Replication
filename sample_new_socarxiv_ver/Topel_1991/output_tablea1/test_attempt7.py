#!/usr/bin/env python3
"""Test improvements for attempt 7."""
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
df = df[df['education_years'].notna()]

GNP = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,1973:44.4,
       1974:48.9,1975:53.6,1976:56.9,1977:60.6,1978:65.2,1979:72.6,
       1980:82.4,1981:90.9,1982:100.0,1983:103.9}
CPS = {1968:1.000,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,
       1981:1.109,1982:1.103,1983:1.089}

df['gnp_int'] = df['year'].map(GNP)  # interview year
df['cps'] = df['year'].map(CPS)

# Test: wages/hours vs hourly_wage with interview-year GNP
df['hw_alt'] = df['wages'] / df['hours']
mask1 = (df['hourly_wage'] > 0) & (df['hourly_wage'] < 200) & df['gnp_int'].notna()
mask2 = (df['hw_alt'] > 0) & (df['hw_alt'] < 200) & df['gnp_int'].notna()

df.loc[mask1, 'rw_hw'] = np.log(df.loc[mask1, 'hourly_wage']) - np.log(df.loc[mask1, 'gnp_int']/33.4) - np.log(df.loc[mask1, 'cps'])
df.loc[mask2, 'rw_alt'] = np.log(df.loc[mask2, 'hw_alt']) - np.log(df.loc[mask2, 'gnp_int']/33.4) - np.log(df.loc[mask2, 'cps'])

# Labor_inc / hours
df['hw_labor'] = df['labor_inc'] / df['hours']
mask3 = (df['hw_labor'] > 0) & (df['hw_labor'] < 200) & df['gnp_int'].notna()
df.loc[mask3, 'rw_labor'] = np.log(df.loc[mask3, 'hw_labor']) - np.log(df.loc[mask3, 'gnp_int']/33.4) - np.log(df.loc[mask3, 'cps'])

print("Real wage using interview-year GNP:")
print(f"  hourly_wage: mean={df.loc[mask1, 'rw_hw'].mean():.4f}, sd={df.loc[mask1, 'rw_hw'].std(ddof=0):.4f}, N={mask1.sum()}")
print(f"  wages/hours: mean={df.loc[mask2, 'rw_alt'].mean():.4f}, sd={df.loc[mask2, 'rw_alt'].std(ddof=0):.4f}, N={mask2.sum()}")
print(f"  labor_inc/h: mean={df.loc[mask3, 'rw_labor'].mean():.4f}, sd={df.loc[mask3, 'rw_labor'].std(ddof=0):.4f}, N={mask3.sum()}")
print(f"  Target: mean=1.131, sd=0.497")

# Test union as job-level variable
# Paper: "1 if union member in more than half of years on the job"
df_filt = df[mask1].copy()
print(f"\nUnion analysis:")
print(f"  Raw union_member mean (all obs): {df_filt['union_member'].dropna().mean():.3f}")

# Compute job-level union
union_by_job = df_filt.groupby('job_id')['union_member'].agg(['mean', 'count'])
union_by_job['union_job'] = (union_by_job['mean'] > 0.5).astype(float)
# Where all NaN, mark as NaN
union_by_job.loc[union_by_job['count'] == 0, 'union_job'] = np.nan
print(f"  Job-level union (>50% years): mean={union_by_job['union_job'].mean():.3f}")

# Apply back to observations
df_filt = df_filt.merge(union_by_job[['union_job']], on='job_id', how='left')
print(f"  Observation-level union (job-level): mean={df_filt['union_job'].dropna().mean():.3f}")
print(f"  Target: 0.344")

# Test experience with different filters
print(f"\nExperience analysis:")
exp = df_filt['age'] - df_filt['education_years'] - 6
exp = exp.clip(lower=0)
print(f"  Recomputed: mean={exp.mean():.3f}, sd={exp.std(ddof=0):.3f}")
print(f"  Target: mean=20.021, sd=11.045")

# Try trimming extreme experience values
mask_exp = (exp >= 1) & (exp <= 50)
print(f"  With exp 1-50: mean={exp[mask_exp].mean():.3f}, sd={exp[mask_exp].std(ddof=0):.3f}, N={mask_exp.sum()}")

# Test married excluding 1975
print(f"\nMarried analysis:")
# Raw married by year
for yr in [1974, 1975, 1976]:
    sub = df_filt[df_filt['year'] == yr]
    print(f"  {yr}: mean={sub['married'].mean():.3f}, N={len(sub)}")

# What if we use marital_status from the full panel?
# Check if there's a raw marital status variable
print(f"\nMarried overall (raw): {df_filt['married'].mean():.3f}")
