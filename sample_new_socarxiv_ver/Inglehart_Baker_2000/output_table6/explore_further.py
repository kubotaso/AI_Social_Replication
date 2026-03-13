import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','F028','X048WVS','S020'], low_memory=False)

# Check WVS wave 2 for Finland
fi_w2 = wvs[(wvs['S003']==246)&(wvs['S002VS']==2)]
print(f"Finland WVS W2: N={len(fi_w2)}")
if len(fi_w2) > 0:
    print(fi_w2['F028'].value_counts().sort_index())

# Also check if Finland wave 1 gives 13%
fi_w1 = wvs[(wvs['S003']==246)&(wvs['S002VS']==1)]
print(f"\nFinland WVS W1: N={len(fi_w1)}")
if len(fi_w1) > 0:
    print(fi_w1['F028'].value_counts().sort_index())
    valid = fi_w1[fi_w1['F028'].isin([1,2,3,4,6,7,8])]
    monthly = fi_w1[fi_w1['F028'].isin([1,2,3])]
    print(f"pct: {len(monthly)/len(valid)*100:.1f}%")

# Now check East Germany in WVS wave 3 more carefully
de_w3 = wvs[(wvs['S003']==276)&(wvs['S002VS']==3)]
print(f"\nGermany WVS W3: N={len(de_w3)}")
de_w3_copy = de_w3.copy()
de_w3_copy['state'] = de_w3_copy['X048WVS'] % 1000
east_states = [12, 13, 14, 15, 16]
west_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

east_w3 = de_w3_copy[de_w3_copy['state'].isin(east_states)]
west_w3 = de_w3_copy[de_w3_copy['state'].isin(west_states)]
print(f"East N: {len(east_w3)}, West N: {len(west_w3)}")

for label, subset in [('East', east_w3), ('West', west_w3)]:
    valid = subset[subset['F028'].isin([1,2,3,4,6,7,8])]
    monthly = subset[subset['F028'].isin([1,2,3])]
    if len(valid) > 0:
        print(f"  {label}: {len(monthly)/len(valid)*100:.1f}% (N={len(valid)}) paper={'9' if label=='East' else '25'}")

# Check Norway wave 3
no_w3 = wvs[(wvs['S003']==578)&(wvs['S002VS']==3)]
print(f"\nNorway W3: N={len(no_w3)}")
valid = no_w3[no_w3['F028'].isin([1,2,3,4,6,7,8])]
monthly = no_w3[no_w3['F028'].isin([1,2,3])]
print(f"  pct: {len(monthly)/len(valid)*100:.1f}% (paper=13)")

# Check South Africa wave 3
sa_w3 = wvs[(wvs['S003']==710)&(wvs['S002VS']==3)]
print(f"\nSouth Africa W3: N={len(sa_w3)}")
valid = sa_w3[sa_w3['F028'].isin([1,2,3,4,6,7,8])]
monthly = sa_w3[sa_w3['F028'].isin([1,2,3])]
print(f"  pct: {len(monthly)/len(valid)*100:.1f}% (paper=70)")
# Try with -2
valid_n2 = sa_w3[sa_w3['F028'].isin([-2,1,2,3,4,6,7,8])]
print(f"  pct w/-2: {len(monthly)/len(valid_n2)*100:.1f}%")

# Check South Korea wave 1 more carefully - currently 27%, paper says 29
sk_w1 = wvs[(wvs['S003']==410)&(wvs['S002VS']==1)]
print(f"\nSouth Korea W1: N={len(sk_w1)}")
print(sk_w1['F028'].value_counts().sort_index())
# With -2 in denominator: 266/970 = 27%
# Without -2: 266/596 = 45%
# Paper says 29%
# Try: what if we use 8pt scale (include 5)?
valid_8 = sk_w1[sk_w1['F028'].isin([1,2,3,4,5,6,7,8])]
print(f"  8pt: {266/len(valid_8)*100:.1f}% (N={len(valid_8)})" if len(valid_8)>0 else "  8pt: N/A")
# Try with -2 and 8pt
valid_8n2 = sk_w1[sk_w1['F028'].isin([-2,1,2,3,4,5,6,7,8])]
monthly_sk = sk_w1[sk_w1['F028'].isin([1,2,3])]
print(f"  8pt+neg2: {len(monthly_sk)/len(valid_8n2)*100:.1f}% (N={len(valid_8n2)})" if len(valid_8n2)>0 else "  N/A")

# The answer might be a different treatment of -2:
# If South Korea -2 means "no religion" (different from "never attends")
# and the paper includes "no religion" in the denominator but not all -2 values mean that
# Perhaps for S.Korea: 266/(596+374) = 266/970 = 27.4% ~ 27
# Paper says 29. Close but not exact.
# What if using just values [1,2,3] out of [1,2,3,4,6,7,8] (7pt)?
# 266/596 = 44.6 -- way too high
# What if using floor division: round(266/970*100) = 27
# But round(266/920*100) = 29... what if some -2 are excluded?
# 266/(970-50) = 266/920 = 28.9 ~ 29!
# Or: what if we treat some -2 as truly missing?
print(f"\n  S.Korea W1 exact: {len(monthly_sk)}/{970} = {len(monthly_sk)/970*100:.2f}%")
# 266/970 = 27.42 => rounds to 27
# To get 29: need denominator ~917. But that's hard to justify.
