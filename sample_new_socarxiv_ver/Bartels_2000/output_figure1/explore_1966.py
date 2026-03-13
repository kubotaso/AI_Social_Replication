import pandas as pd
import numpy as np

df = pd.read_csv('anes_cumulative.csv', low_memory=False)
df['VCF0004'] = pd.to_numeric(df['VCF0004'], errors='coerce')
df['VCF0301'] = pd.to_numeric(df['VCF0301'], errors='coerce')

# Check 1966 specifically
yr = df[df['VCF0004'] == 1966]
print("1966 VCF0301 value counts:")
print(yr['VCF0301'].value_counts().sort_index())
print(f"\nTotal valid (1-7): {yr['VCF0301'].isin([1,2,3,4,5,6,7]).sum()}")
print(f"Total all: {len(yr)}")

# Check if there's a VCF0303 (alternate party ID) variable
cols = [c for c in df.columns if 'VCF030' in c]
print(f"\nVCF030x columns: {cols}")

# Compare 1964 to be safe
print("\n--- 1964 ---")
yr64 = df[df['VCF0004'] == 1964]
print("1964 VCF0301 value counts:")
print(yr64['VCF0301'].value_counts().sort_index())
print(f"Total valid: {yr64['VCF0301'].isin([1,2,3,4,5,6,7]).sum()}")

# Check 1952
print("\n--- 1952 ---")
yr52 = df[df['VCF0004'] == 1952]
print("1952 VCF0301 value counts:")
print(yr52['VCF0301'].value_counts().sort_index())
n52 = yr52['VCF0301'].isin([1,2,3,4,5,6,7]).sum()
print(f"Total valid: {n52}")

# For 1952, compute proportions
valid_52 = yr52[yr52['VCF0301'].isin([1,2,3,4,5,6,7])]
print(f"\n1952 Proportions:")
print(f"Strong (1+7): {(valid_52['VCF0301'].isin([1,7]).sum())/n52:.4f}")
print(f"Weak (2+6): {(valid_52['VCF0301'].isin([2,6]).sum())/n52:.4f}")
print(f"Leaners (3+5): {(valid_52['VCF0301'].isin([3,5]).sum())/n52:.4f}")
print(f"Pure Ind (4): {(valid_52['VCF0301']==4).sum()/n52:.4f}")

# Check VCF0303 if it exists
if 'VCF0303' in df.columns:
    print("\n=== VCF0303 (7-pt party ID alternate) ===")
    yr52_303 = yr52['VCF0303'].dropna()
    print(f"1952 VCF0303 values: {yr52_303.value_counts().sort_index()}")
