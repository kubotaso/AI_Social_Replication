import pandas as pd

# Check the WVS file to understand which version we have
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', nrows=5, low_memory=False)
print("File: WVS_Time_Series_1981-2022_csv_v5_0.csv")
print(f"Columns: {len(wvs.columns)}")

# Check S002VS unique values (all waves)
wvs_waves = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                          usecols=['S002VS', 'S003', 'S020'], low_memory=False)
print(f"Rows: {len(wvs_waves)}")
print(f"S002VS unique: {sorted(wvs_waves['S002VS'].unique())}")
print(f"Year range: {wvs_waves['S020'].min()}-{wvs_waves['S020'].max()}")
print(f"Countries in wave 1: {len(wvs_waves[wvs_waves['S002VS']==1]['S003'].unique())}")
print(f"Countries in wave 2: {len(wvs_waves[wvs_waves['S002VS']==2]['S003'].unique())}")
print(f"Countries in wave 3: {len(wvs_waves[wvs_waves['S002VS']==3]['S003'].unique())}")
print(f"Total unique countries: {len(wvs_waves['S003'].unique())}")

# The WVS/EVS Joint dataset (trend file) would have EVS countries in wave 1
# Our file name says "WVS_Time_Series" -- this is probably WVS-only, not the joint file
# The joint file would be called something like "WVS_EVS_Trend" or similar
