"""Quick check of psid_panel_full.csv"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/psid_panel_full.csv")
print("rows:", len(df))
print("columns:", list(df.columns))
print("unique persons:", df['person_id'].nunique() if 'person_id' in df.columns else 'no person_id')
if 'year' in df.columns:
    print("year range:", df['year'].min(), "-", df['year'].max())
    print("\nyear dist:")
    print(df['year'].value_counts().sort_index())

if 'person_id' in df.columns:
    df['pnum'] = df['person_id'] % 1000
    print("\npnum distribution:")
    print(df['pnum'].value_counts().sort_index().head(20))

# Compare with main panel
df2 = pd.read_csv("data/psid_panel.csv")
print("\n--- Main panel ---")
print("rows:", len(df2))
print("unique persons:", df2['person_id'].nunique())

# Check if full has more columns
full_cols = set(df.columns)
main_cols = set(df2.columns)
print("\nColumns in full but not main:", full_cols - main_cols)
print("Columns in main but not full:", main_cols - full_cols)
