import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','F028','F063','S020'], low_memory=False)

# Brazil wave 3
br3 = wvs[(wvs['S003']==76)&(wvs['S002VS']==3)]
print("Brazil W3 F028:")
print(br3['F028'].value_counts().sort_index())
print()

# Check if F063 exists and has different coding
print("Brazil W3 F063:")
print(br3['F063'].value_counts().sort_index())
print()

# Check wave 2 for comparison
br2 = wvs[(wvs['S003']==76)&(wvs['S002VS']==2)]
print("Brazil W2 F028:")
print(br2['F028'].value_counts().sort_index())
print()
print("Brazil W2 F063:")
print(br2['F063'].value_counts().sort_index())

# Also check wave 4 and 5 for Brazil to see if coding changed
for w in [4, 5]:
    brw = wvs[(wvs['S003']==76)&(wvs['S002VS']==w)]
    if len(brw) > 0:
        print(f"\nBrazil W{w} F028:")
        print(brw['F028'].value_counts().sort_index())
