"""
Investigate the remaining close and miss cells more carefully.
Try different approaches: year filters, valid ranges, different variables.
"""
import pandas as pd
import numpy as np

wvs_path = "data/WVS_Time_Series_1981-2022_csv_v5_0.csv"
evs_dta_path = "data/ZA4460_v3-0-0.dta"

wvs = pd.read_csv(wvs_path, low_memory=False,
                   usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'A006', 'S020', 'G006', 'S017', 'S001'])

# Paper values to compare
targets = {
    ('ZAF', 1): 50,   # South Africa 1981: we get 53
    ('IND', 2): 44,   # India 1990: we get 37
    ('DEU_EAST', 3): 6,  # E. Germany 1995: we get 9
    ('DEU_WEST', 3): 16, # W. Germany 1995: we get 15
    ('MEX', 3): 50,   # Mexico 1995: we get 49
    ('RUS', 3): 19,   # Russia 1995: we get 18
    ('ZAF', 2): 74,   # South Africa 1990: we get 73
    ('ZAF', 3): 71,   # South Africa 1995: we get 70/71
    ('IND', 3): 56,   # India 1995: we get 54
}

def analyze_cell(country, wave, paper_val, wvs_df, do_germany=False):
    """Analyze all possible approaches for a cell."""
    if do_germany:
        sub = wvs_df[(wvs_df['COUNTRY_ALPHA'] == 'DEU') & (wvs_df['S002VS'] == wave)]
        for g006_vals, label in [([2,3], 'EAST'), ([1,4], 'WEST')]:
            subset = sub[sub['G006'].isin(g006_vals)]
            f063_valid = subset[subset['F063'].between(1, 10)]
            if len(f063_valid) > 0:
                pct_uw = (f063_valid['F063'] == 10).mean() * 100
                w = wvs_df.loc[f063_valid.index, 'S017']
                pct_w = (f063_valid['F063'].eq(10) * w).sum() / w.sum() * 100
                print(f"  DEU {label} W{wave}: N={len(f063_valid)}, UW={pct_uw:.4f}%(r{round(pct_uw)},f{int(pct_uw)}), W={pct_w:.4f}%(r{round(pct_w)},f{int(pct_w)})")
        return

    sub = wvs_df[(wvs_df['COUNTRY_ALPHA'] == country) & (wvs_df['S002VS'] == wave)]
    print(f"\n{country} Wave {wave}: N={len(sub)}, paper={paper_val}")
    print(f"  Years: {sorted(sub['S020'].unique())}")
    print(f"  S017 stats: mean={sub['S017'].mean():.4f}, std={sub['S017'].std():.4f}")

    f063_valid = sub[sub['F063'].between(1, 10)]
    if len(f063_valid) > 0:
        pct_uw = (f063_valid['F063'] == 10).mean() * 100
        w = wvs_df.loc[f063_valid.index, 'S017']
        pct_w = (f063_valid['F063'].eq(10) * w).sum() / w.sum() * 100
        print(f"  F063 valid={len(f063_valid)}, UW={pct_uw:.4f}%(r{round(pct_uw)},f{int(pct_uw)}), W={pct_w:.4f}%(r{round(pct_w)},f{int(pct_w)})")
    else:
        print(f"  F063: NO VALID DATA")

    # Try A006 if different scale
    a006_valid = sub[sub['A006'].between(1, 10)]
    if len(a006_valid) > 0:
        pct_a006 = (a006_valid['A006'] == 10).mean() * 100
        print(f"  A006 valid={len(a006_valid)}, %10={pct_a006:.4f}%")

    # Try year-specific subsets for wave 3 countries
    if wave == 3:
        for yr in sorted(sub['S020'].unique()):
            yr_sub = sub[sub['S020'] == yr]
            yr_f063 = yr_sub[yr_sub['F063'].between(1, 10)]
            if len(yr_f063) > 0:
                pct = (yr_f063['F063'] == 10).mean() * 100
                print(f"  Year {yr}: N={len(yr_f063)}, %10={pct:.4f}%")

    # Try different valid range (e.g., only 1-10 with no -4 or -1)
    # Check value distribution
    print(f"  F063 values: {sorted(sub['F063'].value_counts().index.tolist())[:15]}")

print("=== SOUTH AFRICA 1981 ===")
analyze_cell('ZAF', 1, 50, wvs)

print("\n=== INDIA 1990 ===")
analyze_cell('IND', 2, 44, wvs)

print("\n=== INDIA 1995 ===")
analyze_cell('IND', 3, 56, wvs)

print("\n=== EAST/WEST GERMANY 1995 ===")
analyze_cell('DEU', 3, None, wvs, do_germany=True)

print("\n=== MEXICO 1995 ===")
analyze_cell('MEX', 3, 50, wvs)

print("\n=== RUSSIA 1995 ===")
analyze_cell('RUS', 3, 19, wvs)

print("\n=== SOUTH AFRICA 1990 ===")
analyze_cell('ZAF', 2, 74, wvs)

print("\n=== SOUTH AFRICA 1995 ===")
analyze_cell('ZAF', 3, 71, wvs)

# Check S001 - might be different coding
print("\n=== S001 DISTRIBUTION FOR KEY COUNTRIES ===")
for country in ['IND', 'ZAF', 'RUS', 'DEU', 'MEX']:
    sub = wvs[(wvs['COUNTRY_ALPHA'] == country)]
    print(f"{country}: S001 values: {sorted(sub['S001'].unique())}")

# Check if India wave 2 has a subset that gives closer to 44%
print("\n=== INDIA WAVE 2 DETAIL ===")
ind_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'IND') & (wvs['S002VS'] == 2)]
f063_valid = ind_w2[ind_w2['F063'].between(1, 10)]
print(f"India W2 F063 distribution:")
print(f063_valid['F063'].value_counts().sort_index())
print(f"\nTotal valid: {len(f063_valid)}")
print(f"Raw %10: {(f063_valid['F063'] == 10).mean()*100:.4f}%")

# Check if there's a filter that would give ~44%
# India N in paper might be ~1100 not 2474
# Try filtering by year
for yr in sorted(ind_w2['S020'].unique()):
    yr_sub = ind_w2[ind_w2['S020'] == yr]
    yr_f063 = yr_sub[yr_sub['F063'].between(1, 10)]
    if len(yr_f063) > 0:
        pct = (yr_f063['F063'] == 10).mean() * 100
        print(f"India W2 year {yr}: N={len(yr_f063)}, %10={pct:.4f}%")

# Check S001 for India
print("\nIndia S001 values:")
print(ind_w2['S001'].value_counts())
