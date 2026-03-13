#!/usr/bin/env python3
import pandas as pd
df = pd.read_csv('data/psid_panel.csv')
for yr in [1971, 1972, 1973, 1974, 1975, 1976]:
    sub = df[df['year']==yr]['education_clean'].dropna()
    print(f'{yr}: unique={sorted(sub.unique())}, mean={sub.mean():.2f}, n={len(sub)}')
