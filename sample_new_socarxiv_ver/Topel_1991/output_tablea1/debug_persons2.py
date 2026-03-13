#!/usr/bin/env python3
"""Debug: Analyze person count with 1968-1970 data added."""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load main panel (1971-1983)
df = pd.read_csv(os.path.join(BASE, 'data', 'psid_panel.csv'))
print(f'Main panel: {len(df)} obs, {df["person_id"].nunique()} persons, years {df["year"].min()}-{df["year"].max()}')

# Load full panel (1970-1983)
df_full = pd.read_csv(os.path.join(BASE, 'data', 'psid_panel_full.csv'))
print(f'Full panel: {len(df_full)} obs, {df_full["person_id"].nunique()} persons, years {df_full["year"].min()}-{df_full["year"].max()}')
print(f'Full panel columns: {list(df_full.columns)}')

# Get 1970 data from full panel
df_1970 = df_full[df_full['year'] == 1970].copy()
print(f'\n1970 data: {len(df_1970)} obs, {df_1970["person_id"].nunique()} persons')

# Get 1968-1969 data from full panel if available
for yr in [1968, 1969]:
    tmp = df_full[df_full['year'] == yr]
    print(f'{yr} data: {len(tmp)} obs')

# Check what years are in full panel
print(f'\nFull panel year distribution:')
for yr in range(1968, 1984):
    n = (df_full['year'] == yr).sum()
    if n > 0:
        print(f'  {yr}: {n}')

# Merge 1970 into main panel
# First check what columns overlap
main_cols = set(df.columns)
full_cols = set(df_full.columns)
common = main_cols & full_cols
print(f'\nCommon columns: {sorted(common)}')
print(f'Main only: {sorted(main_cols - full_cols)}')
print(f'Full only: {sorted(full_cols - main_cols)}')
