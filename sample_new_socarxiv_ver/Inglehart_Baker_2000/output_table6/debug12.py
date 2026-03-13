"""Debug12: Check weighted percentages for discrepant countries."""
import pandas as pd
import numpy as np

MONTHLY = [1, 2, 3]
VALID = [1, 2, 3, 4, 5, 6, 7, 8]

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS', 'S003', 'S017', 'F028'],
                   low_memory=False)

# Check all discrepant countries with weights
tests = [
    (76, 'Brazil', 3, 54),
    (566, 'Nigeria', 3, 87),
    (840, 'USA', 3, 55),
    (32, 'Argentina', 3, 41),
    (756, 'Switzerland', 3, 25),
    (410, 'South Korea', 1, 29),
    (246, 'Finland', 1, 13),  # check if WVS W1 has weights for Finland
]

for s003, name, wave, paper in tests:
    sub = wvs[(wvs['S003'] == s003) & (wvs['S002VS'] == wave)]
    if len(sub) == 0:
        print(f"{name} W{wave}: NO DATA")
        continue

    valid = sub[sub['F028'].isin(VALID)].copy()
    monthly = valid[valid['F028'].isin(MONTHLY)]

    pct_unwt = round(len(monthly)/len(valid)*100) if len(valid) > 0 else 'N/A'

    if valid['S017'].notna().any() and valid['S017'].sum() > 0:
        w_total = valid['S017'].sum()
        w_monthly = monthly['S017'].sum()
        pct_wt = round(w_monthly/w_total*100)
    else:
        pct_wt = 'N/A'

    # Also try with -2 included
    valid_neg2 = sub[sub['F028'].isin(VALID + [-2])].copy()
    monthly_neg2 = valid_neg2[valid_neg2['F028'].isin(MONTHLY)]
    pct_neg2 = round(len(monthly_neg2)/len(valid_neg2)*100) if len(valid_neg2) > 0 else 'N/A'

    if valid_neg2['S017'].notna().any() and valid_neg2['S017'].sum() > 0:
        w_total_neg2 = valid_neg2['S017'].sum()
        w_monthly_neg2 = monthly_neg2['S017'].sum()
        pct_wt_neg2 = round(w_monthly_neg2/w_total_neg2*100)
    else:
        pct_wt_neg2 = 'N/A'

    print(f"{name:<20} W{wave}: unwt={pct_unwt}%, wt={pct_wt}%, unwt+neg2={pct_neg2}%, wt+neg2={pct_wt_neg2}% (paper={paper})")

# Also check EVS weighting
print("\n=== EVS weighting ===")
evs = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False,
                     columns=['c_abrv', 'q336', 'weight_g', 'weight_s'])

# Italy with weights
ita = evs[evs['c_abrv'] == 'IT']
valid = ita[ita['q336'].isin(VALID)]
monthly = ita[ita['q336'].isin(MONTHLY)]

for wt_col in ['weight_g', 'weight_s']:
    if wt_col in valid.columns and valid[wt_col].notna().any():
        w_total = valid[wt_col].sum()
        w_monthly = monthly[wt_col].sum()
        pct = round(w_monthly/w_total*100)
        print(f"Italy EVS {wt_col}: {pct}% (paper=47)")

# Finland with weights
fin = evs[evs['c_abrv'] == 'FI']
valid = fin[fin['q336'].isin(VALID)]
monthly = fin[fin['q336'].isin(MONTHLY)]

for wt_col in ['weight_g', 'weight_s']:
    if wt_col in valid.columns and valid[wt_col].notna().any():
        w_total = valid[wt_col].sum()
        w_monthly = monthly[wt_col].sum()
        pct = round(w_monthly/w_total*100)
        print(f"Finland EVS {wt_col}: {pct}% (paper=13)")

# Hungary with weights
hun = evs[evs['c_abrv'] == 'HU']
for valid_list in [VALID, [1,2,3,4,5,6,7]]:
    valid = hun[hun['q336'].isin(valid_list)]
    monthly = hun[hun['q336'].isin(MONTHLY)]
    for wt_col in ['weight_g', 'weight_s']:
        if wt_col in valid.columns and valid[wt_col].notna().any():
            w_total = valid[wt_col].sum()
            w_monthly = monthly[wt_col].sum()
            pct = round(w_monthly/w_total*100)
            print(f"Hungary EVS {wt_col} denom={valid_list}: {pct}% (paper=34)")
