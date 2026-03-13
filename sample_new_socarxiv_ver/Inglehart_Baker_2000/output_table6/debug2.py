"""Debug2: Deeper investigation of data issues for Table 6."""
import pandas as pd
import numpy as np

MONTHLY_VALS = [1, 2, 3]
VALID_8PT = [1, 2, 3, 4, 5, 6, 7, 8]

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS', 'S003', 'S020', 'F028', 'S024', 'X048WVS', 'COW_ALPHA'],
                   low_memory=False)

# === GERMANY detailed investigation ===
print("=== GERMANY WVS S024 by wave ===")
deu_wvs = wvs[(wvs['S003'] == 276)]
for w in sorted(deu_wvs['S002VS'].unique()):
    deu_w = deu_wvs[deu_wvs['S002VS'] == w]
    print(f"\nWave {w} (n={len(deu_w)}):")
    for s024_val in sorted(deu_w['S024'].unique()):
        sub = deu_w[deu_w['S024'] == s024_val]
        valid = sub[sub['F028'].isin(VALID_8PT)]
        monthly = sub[sub['F028'].isin(MONTHLY_VALS)]
        x048_vals = sorted(sub['X048WVS'].dropna().unique()[:10])
        pct = round(len(monthly) / len(valid) * 100) if len(valid) > 0 else 'N/A'
        print(f"  S024={s024_val}: n={len(sub)}, valid={len(valid)}, attend%={pct}, X048WVS sample={x048_vals}")

# === Check if WVS has separate East/West Germany entries ===
print("\n\n=== WVS Germany COW_ALPHA ===")
for w in sorted(deu_wvs['S002VS'].unique()):
    deu_w = deu_wvs[deu_wvs['S002VS'] == w]
    print(f"Wave {w}: COW_ALPHA values = {sorted(deu_w['COW_ALPHA'].dropna().unique())}")

# Check if there are separate country codes for East Germany (S003)
print("\n=== Check for separate East Germany code ===")
# DDR might be coded separately
for code in [278, 280, 2276]:  # possible East Germany codes
    ddr = wvs[wvs['S003'] == code]
    if len(ddr) > 0:
        print(f"S003={code}: n={len(ddr)}, waves={sorted(ddr['S002VS'].unique())}")

# === WAVE 1 (1981) data availability ===
print("\n\n=== WAVE 1 (1981) COUNTRIES ===")
w1 = wvs[wvs['S002VS'] == 1]
for s003 in sorted(w1['S003'].unique()):
    n = len(w1[w1['S003'] == s003])
    cow = w1[w1['S003'] == s003]['COW_ALPHA'].dropna().unique()
    valid = w1[(w1['S003'] == s003) & (w1['F028'].isin(VALID_8PT))]
    monthly = w1[(w1['S003'] == s003) & (w1['F028'].isin(MONTHLY_VALS))]
    pct = round(len(monthly)/len(valid)*100) if len(valid) > 0 else 'N/A'
    print(f"  S003={s003} ({cow}): n={n}, valid={len(valid)}, attend%={pct}")

# === Check Great Britain vs Northern Ireland ===
# GB might include Northern Ireland or they might be separate in WVS
print("\n\n=== GB and N.IRELAND in WVS ===")
for w in [1, 2, 3]:
    gb_w = wvs[(wvs['S003'] == 826) & (wvs['S002VS'] == w)]
    nir_w = wvs[(wvs['S003'] == 909) & (wvs['S002VS'] == w)]  # 909 might be N.Ireland
    print(f"Wave {w}: GB(826) n={len(gb_w)}, NIR(909) n={len(nir_w)}")
    # Check other possible codes
    for code in [840, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911]:
        sub = wvs[(wvs['S003'] == code) & (wvs['S002VS'] == w)]
        if len(sub) > 0:
            cow = sub['COW_ALPHA'].dropna().unique()
            print(f"  Also found S003={code} ({cow}): n={len(sub)}")

# === Check all S003 codes in WVS ===
print("\n=== All unique S003 codes across waves 1-3 ===")
w123 = wvs[wvs['S002VS'].isin([1, 2, 3])]
for s003 in sorted(w123['S003'].unique()):
    sub = w123[w123['S003'] == s003]
    cow = sub['COW_ALPHA'].dropna().unique()[:3]
    waves = sorted(sub['S002VS'].unique())
    print(f"  S003={s003} ({cow}): waves={waves}, n={len(sub)}")

# === Check if EVS has wave 1 (1981) data ===
print("\n\n=== EVS wave/year info ===")
evs = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
print("EVS S020 values:", sorted(evs['S020'].unique()))
print("EVS S002VS values:", sorted(evs['S002VS'].dropna().unique()) if 'S002VS' in evs.columns else 'N/A')
