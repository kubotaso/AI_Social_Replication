import pandas as pd
import numpy as np

df = pd.read_csv('anes_cumulative.csv', low_memory=False)

# Check different race variables for 1952
year_df = df[df['VCF0004'] == 1952]
print("1952 race variables:")
print(f"VCF0105a: {year_df['VCF0105a'].value_counts().sort_index().to_dict()}")
print(f"VCF0105b: {year_df['VCF0105b'].value_counts().sort_index().to_dict()}")
print(f"VCF0106: {year_df['VCF0106'].value_counts().sort_index().to_dict()}")

# Check if using VCF0106 gives more white respondents
# VCF0106: 1=White, 2=Black, 3=Asian/PI/Native, 4=Hispanic, 5=Other
voters_52 = year_df[year_df['VCF0704a'].isin([1, 2])]
voters_52 = voters_52[voters_52['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])]
print(f"\n1952 voters with valid PID: {len(voters_52)}")
print(f"VCF0105a==1: {len(voters_52[voters_52['VCF0105a'] == 1])}")
if 'VCF0106' in voters_52.columns:
    print(f"VCF0106==1: {len(voters_52[voters_52['VCF0106'] == 1])}")

# Check VCF0113 coding in codebook - maybe I have it reversed?
# Let me check which region has more respondents in typical years
for year in [1952, 1972, 1988]:
    year_df = df[df['VCF0004'] == year]
    voters = year_df[year_df['VCF0704a'].isin([1, 2])]
    voters = voters[voters['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])]
    white = voters[voters['VCF0105a'] == 1]
    print(f"\n{year} white voters VCF0113:")
    print(f"  1 (South): {len(white[white['VCF0113'] == 1])}")
    print(f"  2 (Non-South): {len(white[white['VCF0113'] == 2])}")
    # South states typically have ~25% of population
    total = len(white[white['VCF0113'].isin([1,2])])
    if total > 0:
        pct_south = len(white[white['VCF0113'] == 1]) / total * 100
        print(f"  South pct: {pct_south:.1f}%")
