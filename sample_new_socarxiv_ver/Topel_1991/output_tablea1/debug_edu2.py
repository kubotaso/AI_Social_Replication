#!/usr/bin/env python3
import pandas as pd
df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')
for yr in [1971, 1972, 1973, 1974, 1975, 1976]:
    sub = df[df['year'] == yr]
    ec = sub['education_clean']
    vals = sorted(ec.dropna().unique())
    print(f'{yr}: education_clean range=[{ec.min()},{ec.max()}], mean={ec.mean():.2f}, unique={vals}')
