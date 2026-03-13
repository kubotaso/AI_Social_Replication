import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','F028','X048WVS','S020'], low_memory=False)

w1 = wvs[wvs['S002VS'] == 1]
print("Countries in WVS Wave 1:")
for code in sorted(w1['S003'].unique()):
    sub = w1[w1['S003'] == code]
    yr = sub['S020'].unique()
    print(f"  S003={code}: N={len(sub)}, years={sorted(yr)}")

# Countries we need 1981 data for but don't have from WVS:
# Belgium, Canada, France, Great Britain, Iceland, Ireland, N.Ireland, Italy, Netherlands, Norway, Spain, Sweden, US, West Germany
# These are all EVS-only countries for 1981
# The WVS has wave 1 data for: Argentina(32), Australia(36), Finland(246), Germany(276), Hungary(348), Japan(392), S.Korea(410), Mexico(484), S.Africa(710)

# Check Germany wave 1 for East/West split
de_w1 = w1[w1['S003'] == 276]
if len(de_w1) > 0:
    de_w1_copy = de_w1.copy()
    de_w1_copy['state'] = de_w1_copy['X048WVS'] % 1000
    east_states = [12, 13, 14, 15, 16]
    west_states = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    west_w1 = de_w1_copy[de_w1_copy['state'].isin(west_states)]
    east_w1 = de_w1_copy[de_w1_copy['state'].isin(east_states)]

    print(f"\nGermany W1: West N={len(west_w1)}, East N={len(east_w1)}")
    if len(west_w1) > 0:
        valid = west_w1[west_w1['F028'].isin([1,2,3,4,6,7,8])]
        monthly = west_w1[west_w1['F028'].isin([1,2,3])]
        if len(valid) > 0:
            print(f"  West W1: {len(monthly)/len(valid)*100:.1f}% (paper=35)")
        # Try with -2
        valid_n2 = west_w1[west_w1['F028'].isin([-2,1,2,3,4,6,7,8])]
        if len(valid_n2) > 0:
            print(f"  West W1 w/-2: {len(monthly)/len(valid_n2)*100:.1f}%")

# Check which 1981 countries we already have correct
print("\n=== WVS Wave 1 church attendance check ===")
paper_1981 = {
    32: ('Argentina', 56), 36: ('Australia', 40), 246: ('Finland', 13),
    348: ('Hungary', 16), 392: ('Japan', 12), 410: ('South Korea', 29),
    484: ('Mexico', 74), 710: ('South Africa', 61)
}

for code, (name, paper_val) in sorted(paper_1981.items()):
    sub = w1[w1['S003'] == code]
    if len(sub) == 0:
        print(f"  {name}: NO DATA")
        continue

    for vset_name, vset in [('7pt', [1,2,3,4,6,7,8]), ('7pt+neg2', [-2,1,2,3,4,6,7,8])]:
        valid = sub[sub['F028'].isin(vset)]
        monthly = sub[sub['F028'].isin([1,2,3])]
        if len(valid) > 0:
            pct = round(len(monthly)/len(valid)*100)
            diff = abs(pct - paper_val)
            match = "FULL" if diff <= 1 else ("PARTIAL" if diff <= 3 else "MISS")
            print(f"  {name:<16} {vset_name}: {pct}% (paper={paper_val}, diff={diff}) {match}")
