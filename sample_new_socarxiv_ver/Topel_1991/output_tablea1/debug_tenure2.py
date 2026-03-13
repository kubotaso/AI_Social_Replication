#!/usr/bin/env python3
"""Debug: Explore raw tenure data more thoroughly."""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, 'data', 'psid_panel.csv'))

# Look at raw tenure by year
print("Raw tenure by year:")
for yr in range(1971, 1984):
    sub = df[df['year'] == yr]
    t = sub['tenure']
    tm = sub['tenure_mos']
    n_t = t.notna().sum()
    n_tm = tm.notna().sum()
    print(f"  {yr}: tenure N={n_t}, range=[{t.min():.0f},{t.max():.0f}], "
          f"tenure_mos N={n_tm}, range=[{tm.min():.0f},{tm.max():.0f}]")

# Check: what values does tenure take in 1971, 1972 (categorical years)?
for yr in [1971, 1972]:
    sub = df[df['year'] == yr]
    print(f"\n  {yr} tenure value counts:")
    print(sub['tenure'].value_counts().sort_index().to_string())

# Check: what values does tenure take in 1973-1975?
for yr in [1973, 1974, 1975]:
    sub = df[df['year'] == yr]
    t = sub['tenure']
    print(f"\n  {yr} tenure: unique values count={t.nunique()}, "
          f"min={t.min()}, max={t.max()}, mean={t.mean():.1f}")
    # Show histogram
    if t.nunique() < 30:
        print(f"  Value counts: {t.value_counts().sort_index().to_dict()}")

# Check: for 1973-1975, is tenure in months or years?
# If mean is ~100, it's months. If ~8, it's years
for yr in [1973, 1974, 1975, 1976, 1977, 1978, 1979]:
    sub = df[df['year'] == yr]
    t = sub['tenure']
    tm = sub['tenure_mos']
    if t.notna().sum() > 0:
        print(f"\n{yr}: tenure mean={t.mean():.1f}, tenure_mos mean={tm.dropna().mean():.1f}")

# Also check same_emp and new_job variables
print("\n\nsame_emp by year:")
for yr in range(1971, 1984):
    sub = df[df['year'] == yr]
    se = sub['same_emp']
    nj = sub['new_job']
    print(f"  {yr}: same_emp values={se.value_counts().to_dict()}, new_job values={nj.value_counts().to_dict()}")
