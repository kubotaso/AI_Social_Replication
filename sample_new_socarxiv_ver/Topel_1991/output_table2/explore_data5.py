#!/usr/bin/env python3
"""Explore additional data details for Table 2."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')

# Table A1 says: Experience mean = 20.021, Tenure mean = 9.978, Education mean = 12.645
# Our data: Experience mean ≈ 18.9, Tenure mean ≈ 4.2 (starting at 1), Education ≈ 12.5
# Tenure is HUGELY different: paper has mean 9.978 but we have 4.2
# This suggests the paper's tenure variable captures TOTAL job tenure including
# years before the panel started. Our tenure_topel only counts from the first
# panel observation.

# Check the 'tenure' and 'tenure_mos' columns
print("=== TENURE VARIABLES ===")
print(f"tenure_topel: mean={df['tenure_topel'].mean():.1f}, min={df['tenure_topel'].min()}, max={df['tenure_topel'].max()}")
if 'tenure' in df.columns:
    print(f"tenure: mean={df['tenure'].dropna().mean():.1f}, min={df['tenure'].dropna().min()}, max={df['tenure'].dropna().max()}, non_null={df['tenure'].notna().sum()}")
if 'tenure_mos' in df.columns:
    print(f"tenure_mos: mean={df['tenure_mos'].dropna().mean():.1f}, min={df['tenure_mos'].dropna().min()}, max={df['tenure_mos'].dropna().max()}, non_null={df['tenure_mos'].notna().sum()}")
if 'lag_tenure' in df.columns:
    print(f"lag_tenure: mean={df['lag_tenure'].dropna().mean():.1f}, min={df['lag_tenure'].dropna().min()}, max={df['lag_tenure'].dropna().max()}, non_null={df['lag_tenure'].notna().sum()}")

# Check the reported tenure in months - convert to years
if 'tenure_mos' in df.columns:
    df['tenure_reported_yrs'] = df['tenure_mos'] / 12.0
    print(f"\nReported tenure (years from months): mean={df['tenure_reported_yrs'].dropna().mean():.1f}")
    print(f"  By year:")
    for y in sorted(df['year'].unique()):
        sub = df[df['year']==y]
        t = sub['tenure_reported_yrs'].dropna()
        if len(t) > 0:
            print(f"  {y}: n={len(t)}, mean={t.mean():.1f}, max={t.max():.1f}")

# Check wage variable more carefully
print("\n=== WAGE VARIABLES ===")
print(f"hourly_wage: mean={df['hourly_wage'].mean():.2f}, computed as wages/hours?")
if 'wages' in df.columns and 'hours' in df.columns:
    df['hw_computed'] = df['wages'] / df['hours']
    print(f"wages/hours: mean={df['hw_computed'].dropna().mean():.2f}")
    # Compare
    diff = (df['hourly_wage'] - df['hw_computed']).abs()
    print(f"Difference hourly_wage vs wages/hours: mean={diff.mean():.4f}, max={diff.max():.4f}")
if 'labor_inc' in df.columns:
    df['hw_labor'] = df['labor_inc'] / df['hours']
    print(f"labor_inc/hours: mean={df['hw_labor'].dropna().mean():.2f}")
    diff2 = (df['hourly_wage'] - df['hw_labor']).abs()
    print(f"Difference hourly_wage vs labor_inc/hours: mean={diff2.mean():.4f}")

# What if we use labor_inc/hours as the wage?
print("\n=== COMPARISON: hourly_wage vs labor_inc/hours ===")
mask = (df['labor_inc'] > 0) & (df['hours'] > 0)
sub = df[mask].copy()
sub['lw'] = np.log(sub['labor_inc'] / sub['hours'])
print(f"N with valid labor_inc/hours: {len(sub)}")
print(f"Mean log(labor_inc/hours): {sub['lw'].mean():.3f}")
print(f"Mean log_hourly_wage: {sub['log_hourly_wage'].mean():.3f}")
print(f"Correlation: {sub['lw'].corr(sub['log_hourly_wage']):.4f}")

# What does Table A1 say about sample composition?
# Mean real wage = 1.131, which is log(average hourly earnings deflated)
# If we compute: log(hourly_wage) - log(CPS_index) - log(GNP_deflator/100)
CPS_IDX = {1971:1.115, 1972:1.113, 1973:1.151, 1974:1.167, 1975:1.188,
           1976:1.117, 1977:1.121, 1978:1.133, 1979:1.128, 1980:1.128,
           1981:1.109, 1982:1.103, 1983:1.089}
GNP_DEFL = {1971:118.92, 1972:123.16, 1973:130.27, 1974:143.08, 1975:155.56,
            1976:163.42, 1977:173.43, 1978:186.18, 1979:201.33, 1980:220.39,
            1981:241.02, 1982:255.09, 1983:264.00}

df['cps'] = df['year'].map(CPS_IDX)
df['gnp'] = df['year'].map(GNP_DEFL)
df['log_real_wage_both'] = df['log_hourly_wage'] - np.log(df['cps']) - np.log(df['gnp']/100)
print(f"\nMean log_real_wage (CPS+GNP): {df['log_real_wage_both'].mean():.3f}")
print(f"Paper Table A1 mean real wage: 1.131")

# Also try just GNP deflator
df['log_real_wage_gnp'] = df['log_hourly_wage'] - np.log(df['gnp']/100)
print(f"Mean log_real_wage (GNP only): {df['log_real_wage_gnp'].mean():.3f}")

# Check: what if the CPS index is DIVIDED (not subtracted in log)?
# i.e., real_wage = nominal / CPS_index
df['log_real_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps'])
print(f"Mean log_real_wage (CPS only): {df['log_real_wage_cps'].mean():.3f}")

# N with tenure_topel >= 2 (= tenure >= 1 in 0-based)
# The paper says "tenure < 1 year deleted"
# With 0-based tenure, this means tenure >= 1, so tenure_topel >= 2
nwj_t2 = len(df[df['tenure_topel'] >= 2])
print(f"\nObs with tenure_topel >= 2 (tenure >= 1 year): {nwj_t2}")
print(f"Obs with tenure_topel >= 1: {len(df)}")

# What about the job-year count? Paper says 13,128 job-years
print(f"\nTotal person-year obs: {len(df)}")
print(f"Paper: 13,128 job-years on 1,540 individuals with 3,228 jobs")
print(f"Our data: {len(df)} obs on {df['person_id'].nunique()} individuals with {df['job_id'].nunique()} jobs")
