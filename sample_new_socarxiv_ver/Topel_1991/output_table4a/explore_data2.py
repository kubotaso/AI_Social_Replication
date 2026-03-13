import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')

# Basic stats
print(f"Total obs: {len(df)}")
print(f"Persons: {df['person_id'].nunique()}")

# Check hours
print(f"\nHours stats:")
print(f"  Min: {df['hours'].min()}")
print(f"  Max: {df['hours'].max()}")
print(f"  Mean: {df['hours'].mean():.0f}")
print(f"  hours >= 250: {(df['hours'] >= 250).sum()}")
print(f"  hours >= 500: {(df['hours'] >= 500).sum()}")

# Check hourly wage
print(f"\nHourly wage stats:")
print(f"  Min: {df['hourly_wage'].min():.2f}")
print(f"  Max: {df['hourly_wage'].max():.2f}")
print(f"  < 1.0: {(df['hourly_wage'] < 1.0).sum()}")
print(f"  < 2.0: {(df['hourly_wage'] < 2.0).sum()}")

# Check labor_inc
print(f"\nLabor income stats:")
print(f"  Min: {df['labor_inc'].min()}")
print(f"  Max: {df['labor_inc'].max()}")

# Check for NaN
for col in ['education_clean', 'age', 'hours', 'hourly_wage', 'log_hourly_wage', 'tenure_topel']:
    n_na = df[col].isna().sum()
    if n_na > 0:
        print(f"  NaN in {col}: {n_na}")

# Check self_employed
print(f"\nSelf employed: {df['self_employed'].value_counts().to_dict()}")

# Check agriculture
print(f"Agriculture: {df['agriculture'].value_counts().to_dict()}")

# Check disabled
print(f"Disabled: {df['disabled'].value_counts().to_dict()}")

# Try computing first-differences with various filters
EDUC_CAT_TO_YEARS = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
df['education_years'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)
df['experience'] = df['age'] - df['education_years'] - 6
df['experience'] = df['experience'].clip(lower=0)

df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
df['prev_year'] = df.groupby('job_id')['year'].shift(1)
wj = df[(df['prev_year'].notna()) & (df['year'] - df['prev_year'] == 1)].copy()
print(f"\nWithin-job obs: {len(wj)}")

# Try different N-reducing filters
print(f"\nFilter exploration:")
print(f"  hours >= 250: {len(wj[wj['hours'] >= 250])}")
print(f"  hours >= 500: {len(wj[wj['hours'] >= 500])}")
print(f"  hourly_wage >= 1: {len(wj[wj['hourly_wage'] >= 1.0])}")
print(f"  hourly_wage >= 2: {len(wj[wj['hourly_wage'] >= 2.0])}")
print(f"  hourly_wage <= 100: {len(wj[wj['hourly_wage'] <= 100.0])}")
print(f"  self_employed == 0: {len(wj[wj['self_employed'] == 0])}")
print(f"  agriculture == 0: {len(wj[wj['agriculture'] == 0])}")

# Try combo filters
mask = (wj['hours'] >= 250) & (wj['hourly_wage'] >= 1.0) & (wj['hourly_wage'] <= 100.0)
print(f"  hours>=250 & 1<=wage<=100: {mask.sum()}")

# Check the d_log_wage distribution
wj['d_log_wage'] = wj['log_hourly_wage'] - wj.groupby('job_id')['log_hourly_wage'].shift(1)
print(f"\n  d_log_wage quantiles:")
print(f"    1%: {wj['d_log_wage'].quantile(0.01):.3f}")
print(f"    5%: {wj['d_log_wage'].quantile(0.05):.3f}")
print(f"    50%: {wj['d_log_wage'].quantile(0.50):.3f}")
print(f"    95%: {wj['d_log_wage'].quantile(0.95):.3f}")
print(f"    99%: {wj['d_log_wage'].quantile(0.99):.3f}")

# Try top/bottom coding
for cutoff in [1.0, 1.5, 2.0]:
    n = len(wj[wj['d_log_wage'].between(-cutoff, cutoff)])
    print(f"  |d_log_wage| < {cutoff}: {n}")
