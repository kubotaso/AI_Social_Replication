#!/usr/bin/env python3
import pandas as pd
df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

for yr in range(1971, 1984):
    sub = df[df['year'] == yr]
    ec = sub['education_clean']
    counts = ec.value_counts().sort_index()
    print(f'{yr}: N={len(sub)}, education_clean counts:')
    for code, n in counts.items():
        print(f'  code {code:.0f}: {n}')
    print()
