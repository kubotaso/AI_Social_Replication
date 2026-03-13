"""
Try various sample restrictions to get closer to paper's N.
Paper: Step 1 N=8,683, Step 2 N=10,685 (13,128 job-years on 1,540 individuals)
Our data: Step 1 N=10,749, Step 2 N=13,922 (on ~2,400 individuals)

We need to reduce from 2,400 to ~1,540 persons.
The paper says "first 16 (1968-83) waves of the PSID" and data from Altonji and Sicherman.
"""
import pandas as pd, numpy as np, statsmodels.api as sm

df = pd.read_csv('data/psid_panel.csv')
df = df[~df['region'].isin([5, 6])]

EDUC = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ey'] = df['education_clean'].copy()
m = ~df['year'].isin([1975, 1976])
df.loc[m, 'ey'] = df.loc[m, 'education_clean'].map(EDUC)
df = df.dropna(subset=['ey'])
df['exp'] = (df['age'] - df['ey'] - 6).clip(lower=0)

CPS = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,
       1982:1.103,1983:1.089}
df['cps'] = df['year'].map(CPS)
df['lrw'] = df['log_hourly_wage'] - np.log(df['cps'])
df['ie'] = (df['exp'] - df['tenure_topel']).clip(lower=0)

for c in ['married','union_member','disabled','region_ne','region_nc','region_south']:
    df[c] = df[c].fillna(0)

print(f"Full data: {len(df)} obs, {df['person_id'].nunique()} persons, {df['job_id'].nunique()} jobs")

# Try different restrictions
# 1. Drop if hourly wage is extreme
df_t1 = df[(df['hourly_wage'] >= 1) & (df['hourly_wage'] <= 100)]
print(f"After wage 1-100: {len(df_t1)} obs, {df_t1['person_id'].nunique()} persons")

# 2. Require at least 2 observations in a job (need for first diffs)
job_counts = df.groupby(['person_id','job_id']).size()
good_jobs = job_counts[job_counts >= 2].index
df_t2 = df[df.set_index(['person_id','job_id']).index.isin(good_jobs)]
print(f"After require 2+ obs per job: {len(df_t2)} obs, {df_t2['person_id'].nunique()} persons")

# 3. Require experience > 0
df_t3 = df[df['exp'] > 0]
print(f"After exp > 0: {len(df_t3)} obs, {df_t3['person_id'].nunique()} persons")

# 4. Try dropping obs where education years > 17 or < 6
df_t4 = df[(df['ey'] >= 6) & (df['ey'] <= 17)]
print(f"After edu 6-17: {len(df_t4)} obs, {df_t4['person_id'].nunique()} persons")

# 5. Try keeping only persons observed in multiple years
person_year_counts = df.groupby('person_id')['year'].nunique()
multi_year = person_year_counts[person_year_counts >= 3].index
df_t5 = df[df['person_id'].isin(multi_year)]
print(f"After 3+ years: {len(df_t5)} obs, {df_t5['person_id'].nunique()} persons")

# 6. Try wage within 3 SD of mean
wage_mean = df['log_hourly_wage'].mean()
wage_std = df['log_hourly_wage'].std()
df_t6 = df[df['log_hourly_wage'].between(wage_mean - 3*wage_std, wage_mean + 3*wage_std)]
print(f"After wage 3SD trim: {len(df_t6)} obs, {df_t6['person_id'].nunique()} persons")

# 7. The paper likely has a different set of individuals from the 1968-83 waves
# that we cannot exactly replicate. But let's try: keep only persons who appear
# in at least 5 waves (more established panel members)
person_year_counts5 = df.groupby('person_id')['year'].nunique()
multi_year5 = person_year_counts5[person_year_counts5 >= 5].index
df_t7 = df[df['person_id'].isin(multi_year5)]
print(f"After 5+ years: {len(df_t7)} obs, {df_t7['person_id'].nunique()} persons")

# 8. What about hours restriction? Paper says positive earnings
# but maybe also needs reasonable hours (e.g., >= 500 hours/year for full-time)
df_t8 = df[df['hours'] >= 500]
print(f"After hours >= 500: {len(df_t8)} obs, {df_t8['person_id'].nunique()} persons")

# Combine: wage 1-100, exp > 0, hours >= 500
df_combo = df[(df['hourly_wage'] >= 1) & (df['hourly_wage'] <= 100) & (df['exp'] > 0) & (df['hours'] >= 500)]
print(f"\nCombo (wage 1-100 + exp>0 + hours>=500): {len(df_combo)} obs, {df_combo['person_id'].nunique()} persons")

# Combo with 3+ years requirement
df_combo2 = df_combo[df_combo['person_id'].isin(person_year_counts[person_year_counts>=3].index)]
print(f"Combo + 3+ years: {len(df_combo2)} obs, {df_combo2['person_id'].nunique()} persons")
