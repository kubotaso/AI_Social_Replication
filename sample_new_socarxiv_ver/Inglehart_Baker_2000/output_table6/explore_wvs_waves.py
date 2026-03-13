import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S002','S003','F028','X048WVS','S020'], low_memory=False)

# Check if S002 (original wave) is different from S002VS (standardized wave)
print("S002 unique values:", sorted(wvs['S002'].unique()))
print("S002VS unique values:", sorted(wvs['S002VS'].unique()))

# Check S002 vs S002VS relationship
print("\n=== S002 vs S002VS cross-tab ===")
ct = pd.crosstab(wvs['S002'], wvs['S002VS'])
print(ct)

# Check which countries are in S002==1 (maybe original wave coding)
if 1 in wvs['S002'].unique():
    w1_orig = wvs[wvs['S002'] == 1]
    print(f"\n=== S002==1 countries ===")
    for code in sorted(w1_orig['S003'].unique()):
        sub = w1_orig[w1_orig['S003'] == code]
        yr = sorted(sub['S020'].unique())
        print(f"  S003={code}: N={len(sub)}, years={yr}")

# Maybe the 1981 data is coded differently
# Check if any data from 1981-1984 exists for our missing countries
target_codes = [56, 124, 250, 276, 352, 372, 380, 528, 578, 724, 752, 826, 840]
early_years = wvs[(wvs['S020'] >= 1981) & (wvs['S020'] <= 1984)]
print(f"\n=== Data from 1981-1984 ===")
for code in sorted(early_years['S003'].unique()):
    sub = early_years[early_years['S003'] == code]
    yr = sorted(sub['S020'].unique())
    waves = sorted(sub['S002VS'].unique())
    print(f"  S003={code}: N={len(sub)}, years={yr}, S002VS={waves}")
