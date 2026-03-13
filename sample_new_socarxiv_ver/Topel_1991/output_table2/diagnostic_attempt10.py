#!/usr/bin/env python3
"""
Diagnostic: Compare full panel vs regular panel, check CPS wage index impact,
and explore experience distribution differences.
"""
import pandas as pd
import numpy as np

# Compare datasets
df_full = pd.read_csv('data/psid_panel_full.csv')
df_reg = pd.read_csv('data/psid_panel.csv')

print(f"Full panel: {len(df_full)} obs, {df_full['person_id'].nunique()} persons")
print(f"Years in full: {sorted(df_full['year'].unique())}")
print(f"\nRegular panel: {len(df_reg)} obs, {df_reg['person_id'].nunique()} persons")
print(f"Years in regular: {sorted(df_reg['year'].unique())}")

print("\nFull panel year distribution:")
for yr in sorted(df_full['year'].unique()):
    n = (df_full['year'] == yr).sum()
    print(f"  {yr}: {n} obs")

# Check what columns are available
print(f"\nFull panel columns: {list(df_full.columns)}")
print(f"\nRegular panel columns: {list(df_reg.columns)}")

# Check if there are overlapping persons
persons_full = set(df_full['person_id'].unique())
persons_reg = set(df_reg['person_id'].unique())
print(f"\nPersons in full only: {len(persons_full - persons_reg)}")
print(f"Persons in reg only: {len(persons_reg - persons_full)}")
print(f"Persons in both: {len(persons_full & persons_reg)}")

# CPS Real Wage Index from Table A1
CPS_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

# How much does CPS index change year-to-year?
print("\nCPS wage index year-to-year changes:")
years = sorted(CPS_INDEX.keys())
for i in range(1, len(years)):
    yr = years[i]
    prev_yr = years[i-1]
    d = np.log(CPS_INDEX[yr]) - np.log(CPS_INDEX[prev_yr])
    print(f"  {prev_yr}->{yr}: d_log = {d:.4f}")

print("\nWith year dummies, the CPS index is absorbed (it only varies by year).")
print("So CPS deflation should NOT affect first-differenced regression with year dummies.")

# Key question: experience distribution
# Paper has mean experience = 20.021
# Our data has mean around 18
# With the paper covering 1968-1983, workers entering in 1968 at age 20
# with 12 years education would have exp = 20-12-6 = 2 in 1968
# By 1983, they'd have exp = 35-12-6 = 17
# So the 1968-1970 data adds a LOT of low-experience observations

# Let's check our experience distribution
print("\nExperience distribution in regular panel:")
print(df_reg['age'].describe())

# Check if using the full panel gives us more data
print("\nChecking full panel for useful extra data...")
# The full panel might have 1968-1970 data points
extra = df_full[~df_full['year'].isin(df_reg['year'].unique())]
print(f"Extra years data: {len(extra)} obs from years {sorted(extra['year'].unique())}")
