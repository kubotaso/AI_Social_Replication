"""
Check Hungary computation to understand the 7pt scale approach
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
evs_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

evs = pd.read_stata(evs_path, convert_categoricals=False)

print("=== HUNGARY in EVS ===")
hun_evs = evs[evs['c_abrv'] == 'HU']
print(f"Hungary rows: {len(hun_evs)}")
print(f"q336 value_counts:\n{hun_evs['q336'].value_counts().sort_index()}")

# Standard approaches
hun8 = hun_evs[hun_evs['q336'].isin([1,2,3,4,5,6,7,8])]
m8 = hun8['q336'].isin([1,2,3]).sum()
t8 = len(hun8)
print(f"\n8pt: {m8}/{t8} = {m8/t8*100:.1f}%")

hun7 = hun_evs[hun_evs['q336'].isin([1,2,3,4,5,6,7])]
m7 = hun7['q336'].isin([1,2,3]).sum()
t7 = len(hun7)
print(f"7pt: {m7}/{t7} = {m7/t7*100:.1f}%")

# Hungary in WVS Wave 2
print("\n=== HUNGARY in WVS Wave 2 ===")
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'F028', 'S017'],
                  low_memory=False)
hun_w2 = wvs[(wvs['S002VS'] == 2) & (wvs['S003'] == 348)]
print(f"Hungary WVS Wave 2: {len(hun_w2)} rows")
if len(hun_w2) > 0:
    print(f"F028:\n{hun_w2['F028'].value_counts().sort_index()}")
    hun_w2_valid = hun_w2[hun_w2['F028'].isin([1,2,3,4,5,6,7,8])]
    m = hun_w2_valid['F028'].isin([1,2,3]).sum()
    t = len(hun_w2_valid)
    print(f"WVS wave 2 Hungary: {m}/{t} = {m/t*100:.1f}%")

    # 7pt
    hun_w2_7 = hun_w2[hun_w2['F028'].isin([1,2,3,4,5,6,7])]
    m7 = hun_w2_7['F028'].isin([1,2,3]).sum()
    t7 = len(hun_w2_7)
    print(f"7pt: {m7}/{t7} = {m7/t7*100:.1f}%")

print("\n=== HUNGARY in WVS Wave 1 ===")
hun_w1 = wvs[(wvs['S002VS'] == 1) & (wvs['S003'] == 348)]
print(f"Hungary WVS Wave 1: {len(hun_w1)} rows")
if len(hun_w1) > 0:
    print(f"F028:\n{hun_w1['F028'].value_counts().sort_index()}")
    hun_w1_valid = hun_w1[hun_w1['F028'].isin([1,2,3,4,5,6,7])]
    m = hun_w1_valid['F028'].isin([1,2,3]).sum()
    t = len(hun_w1_valid)
    print(f"7pt: {m}/{t} = {m/t*100:.1f}%")

    hun_w1_8 = hun_w1[hun_w1['F028'].isin([1,2,3,4,5,6,7,8])]
    m8 = hun_w1_8['F028'].isin([1,2,3]).sum()
    t8 = len(hun_w1_8)
    print(f"8pt: {m8}/{t8} = {m8/t8*100:.1f}%")

# Check: what value 8 means in Hungary
print("\nHungary EVS q336 full distribution:")
print(hun_evs['q336'].value_counts(dropna=False).sort_index())
