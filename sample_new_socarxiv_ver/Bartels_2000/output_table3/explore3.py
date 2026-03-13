import pandas as pd
import numpy as np

df = pd.read_csv('anes_cumulative.csv', usecols=['VCF0004','VCF0301','VCF0707','VCF0902'], low_memory=False)

# For 1996, investigate in detail
print("=== 1996 Analysis ===")
sub96 = df[(df['VCF0004']==1996) & (df['VCF0707'].isin([1,2]))].copy()
print(f"Total voters: {len(sub96)}")
print(f"VCF0902 value counts:")
print(sub96['VCF0902'].value_counts().sort_index())
print(f"VCF0902 null: {sub96['VCF0902'].isna().sum()}")
print(f"VCF0301 value counts:")
print(sub96['VCF0301'].value_counts().sort_index())
print(f"VCF0301 null/missing: {sub96['VCF0301'].isna().sum()}")

# Cross-tab VCF0707 by VCF0301 for 1996
print("\nCross-tab vote by PID for 1996:")
ct = pd.crosstab(sub96['VCF0301'], sub96['VCF0707'], margins=True)
print(ct)

# For 1974, check what happens if we treat null VCF0902 as open seat
print("\n=== 1974 Analysis ===")
sub74 = df[(df['VCF0004']==1974) & (df['VCF0707'].isin([1,2]))].copy()
print(f"Total voters: {len(sub74)}")
print(f"VCF0902 null: {sub74['VCF0902'].isna().sum()}")
print(f"VCF0301 of those with null VCF0902:")
null_inc = sub74[sub74['VCF0902'].isna()]
print(null_inc['VCF0301'].value_counts().sort_index())
print(f"Vote of those with null VCF0902:")
print(null_inc['VCF0707'].value_counts().sort_index())

# For 1976, check what's missing
print("\n=== 1976 Analysis ===")
sub76 = df[(df['VCF0004']==1976) & (df['VCF0707'].isin([1,2]))].copy()
print(f"Total voters: {len(sub76)}")
print(f"VCF0902 null: {sub76['VCF0902'].isna().sum()}")
print(f"Non-null VCF0902: {sub76['VCF0902'].notna().sum()}")
# Those with null VCF0902
null_76 = sub76[sub76['VCF0902'].isna()]
print(f"VCF0301 of null VCF0902:")
print(null_76['VCF0301'].value_counts().sort_index())

# Check VCF0902 distribution for 1982
print("\n=== 1982 Analysis ===")
sub82 = df[(df['VCF0004']==1982) & (df['VCF0707'].isin([1,2]))].copy()
print(f"Total voters: {len(sub82)}")
print(f"VCF0902 value counts:")
print(sub82['VCF0902'].value_counts().sort_index())
print(f"VCF0902 null: {sub82['VCF0902'].isna().sum()}")

# Check if there's a VCF0303 (alternative PID) or other PID variable
print("\n=== Checking for alternative PID variables ===")
df_full_cols = pd.read_csv('anes_cumulative.csv', nrows=0)
pid_cols = [c for c in df_full_cols.columns if 'VCF030' in c or 'VCF031' in c]
print(f"PID-related columns: {pid_cols}")
