import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("anes_cumulative.csv", low_memory=False)

# 1952 example from paper p.39:
# Coefficients: strong=1.600, weak=0.928, leaners=0.902
# Proportions: strong=0.391, weak=0.376, leaners=0.176
# Average = 1.600*0.391 + 0.928*0.376 + 0.902*0.176 = 1.133

# Let's verify these proportions come from "the electorate"
year = 1952
ydf = df[df['VCF0004'] == year].copy()

# All respondents with valid 7-point PID
all_pid = ydf[ydf['VCF0301'].isin([1,2,3,4,5,6,7])]
n = len(all_pid)
p_strong = len(all_pid[all_pid['VCF0301'].isin([1,7])]) / n
p_weak = len(all_pid[all_pid['VCF0301'].isin([2,6])]) / n
p_lean = len(all_pid[all_pid['VCF0301'].isin([3,5])]) / n
p_ind = len(all_pid[all_pid['VCF0301'] == 4]) / n
print(f"1952 ALL respondents with valid PID (N={n}):")
print(f"  strong={p_strong:.3f} weak={p_weak:.3f} lean={p_lean:.3f} ind={p_ind:.3f} sum={p_strong+p_weak+p_lean+p_ind:.3f}")
print(f"  Paper says: strong=0.391, weak=0.376, lean=0.176")

# Just voters with valid PID
voters = ydf[(ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
n_v = len(voters)
pv_strong = len(voters[voters['VCF0301'].isin([1,7])]) / n_v
pv_weak = len(voters[voters['VCF0301'].isin([2,6])]) / n_v
pv_lean = len(voters[voters['VCF0301'].isin([3,5])]) / n_v
pv_ind = len(voters[voters['VCF0301'] == 4]) / n_v
print(f"\n1952 VOTERS with valid PID (N={n_v}):")
print(f"  strong={pv_strong:.3f} weak={pv_weak:.3f} lean={pv_lean:.3f} ind={pv_ind:.3f}")

# Partisans only (excl pure ind) from all respondents
partisans = all_pid[all_pid['VCF0301'] != 4]
n_p = len(partisans)
pp_strong = len(partisans[partisans['VCF0301'].isin([1,7])]) / n_p
pp_weak = len(partisans[partisans['VCF0301'].isin([2,6])]) / n_p
pp_lean = len(partisans[partisans['VCF0301'].isin([3,5])]) / n_p
print(f"\n1952 ALL partisans excl pure ind (N={n_p}):")
print(f"  strong={pp_strong:.3f} weak={pp_weak:.3f} lean={pp_lean:.3f}")

# Verify the computation using Table 1 coefficients
# Table 1 says: strong=1.600, weak=0.928, leaners=0.902
# Paper says average = 1.133
coefs_table1 = {'strong': 1.600, 'weak': 0.928, 'lean': 0.902}

avg_all = coefs_table1['strong']*p_strong + coefs_table1['weak']*p_weak + coefs_table1['lean']*p_lean
avg_voters = coefs_table1['strong']*pv_strong + coefs_table1['weak']*pv_weak + coefs_table1['lean']*pv_lean
avg_partisans = coefs_table1['strong']*pp_strong + coefs_table1['weak']*pp_weak + coefs_table1['lean']*pp_lean

print(f"\nUsing Table 1 coefs (1.600, 0.928, 0.902):")
print(f"  With ALL respondent props:  {avg_all:.4f} (paper says 1.133)")
print(f"  With VOTER props:          {avg_voters:.4f}")
print(f"  With PARTISAN props:       {avg_partisans:.4f}")

# Now check: which proportions match 0.391/0.376/0.176?
print(f"\nPaper proportions: 0.391, 0.376, 0.176 (sum=0.943)")
print(f"ALL respondents:   {p_strong:.3f}, {p_weak:.3f}, {p_lean:.3f} (sum={p_strong+p_weak+p_lean:.3f})")
print(f"VOTERS:            {pv_strong:.3f}, {pv_weak:.3f}, {pv_lean:.3f} (sum={pv_strong+pv_weak+pv_lean:.3f})")
print(f"PARTISANS:         {pp_strong:.3f}, {pp_weak:.3f}, {pp_lean:.3f} (sum={pp_strong+pp_weak+pp_lean:.3f})")

# Now for Figure 4 -- compute White Non-South and White South proportions
# using the SAME approach as Figure 3 (full electorate proportions)
print("\n\n===== FIGURE 4 APPROACH =====")
print("Using full electorate proportions (all with valid PID, incl pure ind)")

for year in [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]:
    ydf = df[df['VCF0004'] == year].copy()

    # White Non-South electorate
    wns_all = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==2) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    n_wns = len(wns_all)
    if n_wns > 0:
        wns_s = len(wns_all[wns_all['VCF0301'].isin([1,7])]) / n_wns
        wns_w = len(wns_all[wns_all['VCF0301'].isin([2,6])]) / n_wns
        wns_l = len(wns_all[wns_all['VCF0301'].isin([3,5])]) / n_wns

    # White South electorate
    ws_all = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==1) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    n_ws = len(ws_all)
    if n_ws > 0:
        ws_s = len(ws_all[ws_all['VCF0301'].isin([1,7])]) / n_ws
        ws_w = len(ws_all[ws_all['VCF0301'].isin([2,6])]) / n_ws
        ws_l = len(ws_all[ws_all['VCF0301'].isin([3,5])]) / n_ws

    print(f"\n{year}:")
    print(f"  WNS (N={n_wns}): strong={wns_s:.3f} weak={wns_w:.3f} lean={wns_l:.3f}")
    print(f"  WS  (N={n_ws}): strong={ws_s:.3f} weak={ws_w:.3f} lean={ws_l:.3f}")
