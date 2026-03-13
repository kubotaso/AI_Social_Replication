import pandas as pd
df = pd.read_csv('anes_cumulative.csv', usecols=['VCF0004','VCF0301','VCF0707','VCF0902'], low_memory=False)

for y in [1970, 1974, 1976, 1992]:
    sub = df[(df['VCF0004']==y) & (df['VCF0707'].isin([1,2]))]
    print(f'{y}: total voters = {len(sub)}')
    print(f'  VCF0902 value counts:')
    print(sub['VCF0902'].value_counts().sort_index().to_string())
    print(f'  VCF0902 null: {sub["VCF0902"].isna().sum()}')
    print(f'  VCF0301 valid (1-7): {sub["VCF0301"].isin([1,2,3,4,5,6,7]).sum()}')
    print(f'  VCF0301 null or 0: {(~sub["VCF0301"].isin([1,2,3,4,5,6,7])).sum()}')
    print()
