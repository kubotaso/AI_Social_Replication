import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S001', 'S002VS', 'S003', 'S020', 'F028'], low_memory=False)

# Check S001 values
print("S001 value counts:")
print(wvs['S001'].value_counts().sort_index())

# Check S001 by wave
print("\nS001 by S002VS:")
ct = pd.crosstab(wvs['S002VS'], wvs['S001'])
print(ct)

# Check wave 1 by study type
w1 = wvs[wvs['S002VS'] == 1]
print(f"\nWave 1 S001 counts:")
print(w1['S001'].value_counts().sort_index())

# Check if there are EVS records in wave 1
w1_evs = w1[w1['S001'] == 1]
print(f"\nWave 1 EVS records: {len(w1_evs)}")
if len(w1_evs) > 0:
    for code in sorted(w1_evs['S003'].unique()):
        sub = w1_evs[w1_evs['S003'] == code]
        yr = sorted(sub['S020'].unique())
        print(f"  S003={code}: N={len(sub)}, years={yr}")

# Check if there are EVS records in wave 2
w2 = wvs[wvs['S002VS'] == 2]
w2_evs = w2[w2['S001'] == 1]
print(f"\nWave 2 EVS records: {len(w2_evs)}")
if len(w2_evs) > 0:
    for code in sorted(w2_evs['S003'].unique()):
        sub = w2_evs[w2_evs['S003'] == code]
        yr = sorted(sub['S020'].unique())
        print(f"  S003={code}: N={len(sub)}, years={yr}")
