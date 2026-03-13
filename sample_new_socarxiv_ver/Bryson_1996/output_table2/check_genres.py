import pandas as pd
df = pd.read_csv("gss1993_clean.csv")
genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera', 'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']
for g in genres:
    v = pd.to_numeric(df[g], errors='coerce')
    in_range = v.isin([1,2,3,4,5]).sum()
    total = v.notna().sum()
    print(f"{g:10s}: total_notna={total:5d}, in_1to5={in_range:5d}, mean={v.mean():.2f}")

# Check for alternative genre variables
print("\nPotential alternative genre columns:")
for col in sorted(df.columns):
    if any(x in col.lower() for x in ['rock', 'pop', 'swing', 'show', 'tune']):
        print(f"  {col}")
