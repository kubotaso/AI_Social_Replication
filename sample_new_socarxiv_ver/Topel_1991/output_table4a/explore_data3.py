import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')

# Check what person-level filters might reduce the sample
# 1. Persons per number of observations
obs_per_person = df.groupby('person_id').size()
print("Obs per person:")
print(obs_per_person.describe())
print(f"  Persons with only 1 obs: {(obs_per_person == 1).sum()}")
print(f"  Persons with only 2 obs: {(obs_per_person == 2).sum()}")
print(f"  Persons with 1-2 obs: {(obs_per_person <= 2).sum()}")
print(f"  Persons with 3+ obs: {(obs_per_person >= 3).sum()}")

# 2. Check education_clean NaN
edu_na = df.groupby('person_id')['education_clean'].apply(lambda x: x.isna().any())
print(f"\nPersons with any education NaN: {edu_na.sum()}")

# 3. Check if some persons have no valid job spells
jobs_per_person = df.groupby('person_id')['job_id'].nunique()
print(f"\nJobs per person: min={jobs_per_person.min()}, max={jobs_per_person.max()}")

# 4. Check if some persons never have consecutive within-job observations
df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
df['prev_year'] = df.groupby('job_id')['year'].shift(1)
df['is_within'] = (df['prev_year'].notna()) & (df['year'] - df['prev_year'] == 1)
within_persons = df[df['is_within']]['person_id'].nunique()
print(f"\nPersons with any within-job obs: {within_persons}")
print(f"Persons without within-job obs: {df['person_id'].nunique() - within_persons}")

# 5. Year coverage
years_per_person = df.groupby('person_id')['year'].nunique()
print(f"\nYears per person: mean={years_per_person.mean():.1f}")
print(f"  Coverage < 3 years: {(years_per_person < 3).sum()}")
print(f"  Coverage < 5 years: {(years_per_person < 5).sum()}")

# 6. Minimum tenure in any job
max_tenure = df.groupby('person_id')['tenure_topel'].max()
print(f"\nMax tenure by person: mean={max_tenure.mean():.1f}")
print(f"  Max tenure < 2: {(max_tenure < 2).sum()}")
print(f"  Max tenure < 3: {(max_tenure < 3).sum()}")

# 7. Try requiring persons to appear in at least N years
for min_years in [3, 4, 5, 7, 10]:
    n_persons = (years_per_person >= min_years).sum()
    n_obs = df[df['person_id'].isin(years_per_person[years_per_person >= min_years].index)].shape[0]
    print(f"  Persons with {min_years}+ years: {n_persons} ({n_obs} obs)")

# 8. Check for possible household heads vs non-heads
print(f"\nid_1968 range: {df['id_1968'].min()} to {df['id_1968'].max()}")
print(f"Unique id_1968: {df['id_1968'].nunique()}")

# 9. Look at family structure - PSID tracks household heads
# The paper says "white males" - maybe it should be household heads only?
print(f"\nFamily IDs per person:")
fam_per = df.groupby('person_id')['fam_id'].nunique()
print(f"  Mean: {fam_per.mean():.1f}, Max: {fam_per.max()}")
