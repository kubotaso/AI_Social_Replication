import pandas as pd
import numpy as np

df = pd.read_csv('anes_cumulative.csv', low_memory=False)
df['VCF0004'] = pd.to_numeric(df['VCF0004'], errors='coerce')
df['VCF0301'] = pd.to_numeric(df['VCF0301'], errors='coerce')
df['VCF0009x'] = pd.to_numeric(df['VCF0009x'], errors='coerce')
df['VCF0009y'] = pd.to_numeric(df['VCF0009y'], errors='coerce')
df['VCF0009z'] = pd.to_numeric(df['VCF0009z'], errors='coerce')

years = list(range(1952, 1997, 2))
df = df[df['VCF0004'].isin(years)]
df = df[df['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])]

# Check weight variables availability
print("=== Weight variable info ===")
for wt in ['VCF0009x', 'VCF0009y', 'VCF0009z']:
    valid = df[wt].notna().sum()
    print(f"{wt}: {valid} valid values out of {len(df)}")
    if valid > 0:
        print(f"  range: {df[wt].min():.4f} to {df[wt].max():.4f}")
        print(f"  mean: {df[wt].mean():.4f}")
        # Check by year
        for year in [1952, 1972, 1996]:
            yd = df[df['VCF0004']==year]
            v = yd[wt].notna().sum()
            print(f"  Year {year}: {v} valid weights")

print()
print("=== Unweighted vs Weighted (VCF0009z) for key years ===")
for year in [1952, 1964, 1972, 1978, 1988, 1996]:
    yd = df[df['VCF0004']==year]
    n = len(yd)

    # Unweighted
    strong_uw = len(yd[yd['VCF0301'].isin([1,7])]) / n
    weak_uw = len(yd[yd['VCF0301'].isin([2,6])]) / n
    lean_uw = len(yd[yd['VCF0301'].isin([3,5])]) / n
    pure_uw = len(yd[yd['VCF0301']==4]) / n

    # Weighted with VCF0009z
    wt = yd['VCF0009z']
    if wt.notna().sum() > 0:
        wt_filled = wt.fillna(1.0)
        total_w = wt_filled.sum()
        strong_w = wt_filled[yd['VCF0301'].isin([1,7])].sum() / total_w
        weak_w = wt_filled[yd['VCF0301'].isin([2,6])].sum() / total_w
        lean_w = wt_filled[yd['VCF0301'].isin([3,5])].sum() / total_w
        pure_w = wt_filled[yd['VCF0301']==4].sum() / total_w
        print(f"{year} (N={n}): UW: S={strong_uw:.3f} W={weak_uw:.3f} L={lean_uw:.3f} P={pure_uw:.3f} | WT: S={strong_w:.3f} W={weak_w:.3f} L={lean_w:.3f} P={pure_w:.3f}")
    else:
        print(f"{year} (N={n}): UW: S={strong_uw:.3f} W={weak_uw:.3f} L={lean_uw:.3f} P={pure_uw:.3f} | No weights")
