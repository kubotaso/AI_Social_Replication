import pandas as pd
df = pd.read_csv('anes_cumulative.csv', low_memory=False)
df['VCF0004'] = pd.to_numeric(df['VCF0004'], errors='coerce')
df['VCF0301'] = pd.to_numeric(df['VCF0301'], errors='coerce')

# Check weight columns
cols = [c for c in df.columns if 'VCF0009' in c]
print('Weight columns:', cols)
print()

# Check all years
years = list(range(1952, 1997, 2))
for yr in years:
    year_df = df[df['VCF0004'] == yr]
    valid = year_df[year_df['VCF0301'].isin([1,2,3,4,5,6,7])]
    n = len(valid)
    if n == 0:
        continue
    strong = len(valid[valid['VCF0301'].isin([1,7])]) / n
    weak = len(valid[valid['VCF0301'].isin([2,6])]) / n
    leaners = len(valid[valid['VCF0301'].isin([3,5])]) / n
    pure = len(valid[valid['VCF0301'] == 4]) / n
    print(f'{yr}: N={n}, Strong={strong:.3f}, Weak={weak:.3f}, Lean={leaners:.3f}, Pure={pure:.3f}')

# Check if VCF0009z exists and sample with weights for 1966
print()
if 'VCF0009z' in df.columns:
    print("VCF0009z exists - checking weighted proportions")
    for yr in [1958, 1966, 1972]:
        year_df = df[df['VCF0004'] == yr]
        valid = year_df[year_df['VCF0301'].isin([1,2,3,4,5,6,7])].copy()
        valid['VCF0009z'] = pd.to_numeric(valid['VCF0009z'], errors='coerce')
        w = valid['VCF0009z']
        total_w = w.sum()
        if total_w > 0:
            strong_w = w[valid['VCF0301'].isin([1,7])].sum() / total_w
            weak_w = w[valid['VCF0301'].isin([2,6])].sum() / total_w
            lean_w = w[valid['VCF0301'].isin([3,5])].sum() / total_w
            pure_w = w[valid['VCF0301'] == 4].sum() / total_w
            print(f'{yr} weighted: Strong={strong_w:.3f}, Weak={weak_w:.3f}, Lean={lean_w:.3f}, Pure={pure_w:.3f}')
        else:
            print(f'{yr}: no valid weights')
else:
    # Check for other weight-like columns
    weight_cols = [c for c in df.columns if 'weight' in c.lower() or 'wgt' in c.lower() or 'VCF0009' in c.upper()]
    print("No VCF0009z. Other weight cols:", weight_cols)
