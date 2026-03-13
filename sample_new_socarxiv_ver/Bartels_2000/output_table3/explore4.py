import pandas as pd
import numpy as np

df = pd.read_csv('anes_cumulative.csv', usecols=['VCF0004','VCF0301','VCF0302','VCF0303','VCF0707','VCF0902'], low_memory=False)

# Check VCF0302 and VCF0303 for 1996
print("=== 1996 VCF0302 ===")
sub96 = df[(df['VCF0004']==1996) & (df['VCF0707'].isin([1,2]))].copy()
print("VCF0302 value counts:")
print(sub96['VCF0302'].value_counts().sort_index())
print()
print("VCF0303 value counts:")
print(sub96['VCF0303'].value_counts().sort_index())
print()

# Compare VCF0301 vs VCF0303 for 1996
print("Cross-tab VCF0301 vs VCF0303 for 1996:")
ct = pd.crosstab(sub96['VCF0301'], sub96['VCF0303'], margins=True)
print(ct)
print()

# Check VCF0302 coding: this is the 3-point party ID (1=Dem, 2=Ind, 3=Rep)
# VCF0303 is the initial party ID question (pre-follow-up)

# For 1982, check VCF0902 codes 40 and 49 - are they all open seat?
print("=== 1982 VCF0902 details ===")
sub82 = df[(df['VCF0004']==1982) & (df['VCF0707'].isin([1,2]))].copy()
# Check codes 40, 49, 55, 56, 59
for code in [40, 49, 55, 56, 59]:
    n = (sub82['VCF0902'] == code).sum()
    if n > 0:
        print(f"  Code {code}: {n} cases")

# Check if VCF0902 code 40 might be Dem incumbent in some vintages
# Actually, let me check if codes 31, 32 appear in any year
print("\n=== VCF0902 codes 31-39 across all years ===")
for y in [1970, 1974, 1976, 1978, 1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996]:
    sub = df[(df['VCF0004']==y) & (df['VCF0707'].isin([1,2]))]
    codes_30s = sub[sub['VCF0902'].between(30, 39)]['VCF0902'].value_counts().sort_index()
    if len(codes_30s) > 0:
        print(f"  {y}: {codes_30s.to_dict()}")

# Check VCF0902 codes 80-89
print("\n=== VCF0902 codes 80-89 across all years ===")
for y in [1970, 1974, 1976, 1978, 1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996]:
    sub = df[(df['VCF0004']==y) & (df['VCF0707'].isin([1,2]))]
    codes_80s = sub[sub['VCF0902'].between(80, 89)]['VCF0902'].value_counts().sort_index()
    if len(codes_80s) > 0:
        print(f"  {y}: {codes_80s.to_dict()}")

# Check if there's a pattern in the 1996 discrepancy
# Maybe Bartels used the 1996 pilot study or different weighting
# Let me check what VCF0707 values exist in 1996 pilot vs regular
print("\n=== 1996 detailed investigation ===")
sub96_full = df[df['VCF0004']==1996].copy()
print(f"Total 1996 respondents: {len(sub96_full)}")
print(f"VCF0707 value counts (all):")
print(sub96_full['VCF0707'].value_counts(dropna=False).sort_index())
