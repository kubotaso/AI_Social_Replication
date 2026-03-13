import pandas as pd
df = pd.read_csv("gss1993_clean.csv")
# Check all columns that might relate to income or prestige
for col in sorted(df.columns):
    if any(x in col.lower() for x in ['inc', 'earn', 'wage', 'sal', 'prest', 'sei', 'occ']):
        v = pd.to_numeric(df[col], errors='coerce')
        n = v.notna().sum()
        if n > 50:
            print(f"  {col:15s}: N={n:5d}, range=[{v.min():.0f}, {v.max():.0f}], mean={v.mean():.1f}")
