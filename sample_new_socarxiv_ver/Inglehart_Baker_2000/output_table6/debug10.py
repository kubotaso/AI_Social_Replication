"""Debug10: Check if WVS Time Series includes integrated EVS 1981 data."""
import pandas as pd
import numpy as np

MONTHLY_VALS = [1, 2, 3]
VALID_8PT = [1, 2, 3, 4, 5, 6, 7, 8]

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS', 'S003', 'S020', 'F028', 'S024'],
                   low_memory=False)

# Check wave 1 (S002VS=1) and wave 2 (S002VS=2) for European countries
# that need 1981 data
target_countries = {
    56: 'Belgium', 124: 'Canada', 250: 'France',
    826: 'Great Britain', 352: 'Iceland', 372: 'Ireland',
    380: 'Italy', 528: 'Netherlands', 578: 'Norway',
    724: 'Spain', 752: 'Sweden', 840: 'United States',
    # Also check if N.Ireland is separate
}

print("=== Checking WVS for 1981 European countries ===")
# Check S003 codes we might not have tried
w1 = wvs[wvs['S002VS'] == 1]
print(f"Wave 1 total rows: {len(w1)}")
print(f"Wave 1 S003 codes: {sorted(w1['S003'].unique())}")
print(f"Wave 1 S020 years: {sorted(w1['S020'].unique())}")
print()

# Check wave 2 for early years
w2 = wvs[wvs['S002VS'] == 2]
print(f"Wave 2 total rows: {len(w2)}")
print(f"Wave 2 S020 years: {sorted(w2['S020'].unique())}")
print()

# Check S024 for any 1981-era entries
for s003_code in [56, 124, 250, 826, 352, 372, 380, 528, 578, 724, 752, 840, 276]:
    sub = wvs[(wvs['S003'] == s003_code) & (wvs['S002VS'].isin([1, 2]))]
    if len(sub) > 0:
        years = sorted(sub['S020'].unique())
        waves = sorted(sub['S002VS'].unique())
        print(f"  S003={s003_code}: waves={waves}, years={years}, n={len(sub)}")
    else:
        # Try to find any year < 1985 for this country
        sub_early = wvs[(wvs['S003'] == s003_code) & (wvs['S020'] < 1985)]
        if len(sub_early) > 0:
            print(f"  S003={s003_code}: early data found, years={sorted(sub_early['S020'].unique())}")
        # Also check all waves
        sub_all = wvs[wvs['S003'] == s003_code]
        if len(sub_all) > 0:
            all_waves = sorted(sub_all['S002VS'].unique())
            print(f"  S003={s003_code}: available waves={all_waves}")

# Check if USA has wave 1 data (S003=840)
print("\n=== USA in WVS ===")
usa = wvs[wvs['S003'] == 840]
print(f"USA waves: {sorted(usa['S002VS'].unique())}")
print(f"USA years: {sorted(usa['S020'].unique())}")

# The paper says USA 1981=60%. The EVS (wave 2) gives 59%.
# Maybe the paper used a different data source for USA 1981?
# Check if there's WVS wave 1 USA data under a different code
print("\n=== All S003 codes with S020<=1982 ===")
early = wvs[wvs['S020'] <= 1982]
for s003 in sorted(early['S003'].unique()):
    sub = early[early['S003'] == s003]
    print(f"  S003={s003}: n={len(sub)}, years={sorted(sub['S020'].unique())}")
