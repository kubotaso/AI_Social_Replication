#!/usr/bin/env python3
"""Explore full panel and education coding."""
import pandas as pd

df_full = pd.read_csv('data/psid_panel_full.csv')
print('Full panel columns:', sorted(df_full.columns.tolist()))
print()
for col in ['govt_worker', 'self_employed', 'govt', 'public', 'sector']:
    if col in df_full.columns:
        print(f'{col}: unique={sorted(df_full[col].dropna().unique())[:20]}, non_null={df_full[col].notna().sum()}')

print()
# Education distribution in categorical years
df = pd.read_csv('data/psid_panel.csv')
sub_cat = df[~df['year'].isin([1975, 1976])]
print('Education distribution in categorical years:')
print(sub_cat['education_clean'].value_counts().sort_index())

print()
# What is education=9? It could be "college with no degree" or similar
# PSID codes: 0=cant read, 1=0-5, 2=6-8, 3=9-11, 4=12, 5=12+noac, 6=some coll, 7=BA, 8=adv
# Education=9 is NOT in the standard PSID coding
# Possible: it's a special code or data artifact
sub9 = df[(~df['year'].isin([1975, 1976])) & (df['education_clean'] == 9)]
print(f'Educ=9 in categorical years: {len(sub9)} obs')
if len(sub9) > 0:
    print(f'  Years: {sorted(sub9["year"].unique())}')
    print(f'  Mean age: {sub9["age"].mean():.1f}')

# Check same_emp more carefully
print()
print('same_emp value counts:')
print(df['same_emp'].value_counts())

# Check hourly wage range
print()
print('Hourly wage percentiles:')
print(df['hourly_wage'].describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]))

# Mean log wage by year
print()
print('Mean log_hourly_wage by year:')
for y in sorted(df['year'].unique()):
    sub = df[df['year']==y]
    print(f'  {y}: mean_log_wage={sub["log_hourly_wage"].mean():.3f}, mean_wage={sub["hourly_wage"].mean():.2f}')
