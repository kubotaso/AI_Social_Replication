"""Compare EVS 1990 CSV vs ZA4460 Stata for God importance."""
import pandas as pd
import numpy as np

evs_dta = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
evs_csv = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)

# Map ZA4460 country codes to COUNTRY_ALPHA
za_to_alpha = {
    'US': 'USA', 'GB-GBN': 'GBR', 'GB-NIR': 'NIR', 'IE': 'IRL',
    'BE': 'BEL', 'FR': 'FRA', 'SE': 'SWE', 'NL': 'NLD', 'NO': 'NOR',
    'FI': 'FIN', 'IS': 'ISL', 'ES': 'ESP', 'IT': 'ITA', 'DE': 'DEU',
    'CA': 'CAN', 'HU': 'HUN', 'BG': 'BGR', 'SI': 'SVN', 'LV': 'LVA',
    'AT': 'AUT', 'DK': 'DNK', 'CZ': 'CZE', 'EE': 'EST', 'LT': 'LTU',
    'MT': 'MLT', 'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'SK': 'SVK'
}

print("COMPARISON: EVS CSV (A006) vs ZA4460 (q365)")
print(f"{'Country':<20} {'CSV A006':>10} {'ZA4460 q365':>12} {'Paper':>8}")
print("-" * 55)

paper_1990 = {
    'BEL': 13, 'CAN': 28, 'FIN': 12, 'FRA': 10, 'GBR': 16,
    'ISL': 17, 'IRL': 40, 'NIR': 41, 'ITA': 29, 'NLD': 11,
    'NOR': 15, 'ESP': 18, 'SWE': 8, 'USA': 48, 'HUN': 22,
    'BGR': 7, 'SVN': 14, 'LVA': 9,
}

for za_code, alpha in sorted(za_to_alpha.items(), key=lambda x: x[1]):
    paper_val = paper_1990.get(alpha, '')
    if not paper_val:
        continue

    # ZA4460
    za_sub = evs_dta[evs_dta['c_abrv'] == za_code]
    za_valid = za_sub[(za_sub['q365'] >= 1) & (za_sub['q365'] <= 10)]
    za_pct = (za_valid['q365'] == 10).mean() * 100 if len(za_valid) > 0 else None

    # EVS CSV
    csv_sub = evs_csv[evs_csv['COUNTRY_ALPHA'] == alpha]
    csv_valid = csv_sub[(csv_sub['A006'] >= 1) & (csv_sub['A006'] <= 10)]
    csv_pct = (csv_valid['A006'] == 10).mean() * 100 if len(csv_valid) > 0 else None

    za_str = f"{za_pct:.1f}% ({round(za_pct)})" if za_pct else "N/A"
    csv_str = f"{csv_pct:.1f}% ({round(csv_pct)})" if csv_pct else "N/A"

    print(f"{alpha:<20} {csv_str:>16} {za_str:>16} {paper_val:>6}")

# Now check ZA4460 for East/West Germany
print("\n\nZA4460 GERMANY EAST/WEST q365:")
deu_za = evs_dta[evs_dta['c_abrv'] == 'DE']
print(f"Total Germany: {len(deu_za)}")

# Check for region variable in ZA4460
region_cols = [c for c in evs_dta.columns if 'region' in c.lower() or 'east' in c.lower() or 'west' in c.lower()]
print(f"Region columns: {region_cols}")

# Check studyno or studynoc for different German studies
print(f"studyno: {deu_za['studyno'].unique()}")
print(f"studynoc: {deu_za['studynoc'].unique()}")

# Try to split Germany using weight_s or other indicator
# Actually check if c_abrv has separate E/W germany codes
print(f"\nAll c_abrv values: {sorted(evs_dta['c_abrv'].unique())}")

# Check if there's a variable distinguishing E/W Germany
# The country1 or cntry_y variable might distinguish
for col in ['country', 'country1', 'cntry_y', 'cntry1_y']:
    if col in evs_dta.columns:
        de_vals = deu_za[col].unique()
        print(f"{col}: {de_vals}")

# Weighted analysis for ZA4460
print("\n\nZA4460 WEIGHTED vs UNWEIGHTED for close-match countries:")
for za_code, alpha in [('US','USA'), ('BE','BEL'), ('NL','NLD'), ('ES','ESP'),
                        ('GB-GBN','GBR'), ('HU','HUN'), ('NO','NOR')]:
    sub = evs_dta[evs_dta['c_abrv'] == za_code]
    valid = sub[(sub['q365'] >= 1) & (sub['q365'] <= 10)].copy()
    if len(valid) == 0:
        continue
    is_10 = (valid['q365'] == 10).astype(float)
    pct_uw = is_10.mean() * 100

    for wt_col in ['weight_g', 'weight_s']:
        w = valid[wt_col]
        if w.notna().sum() > 0 and w.sum() > 0:
            pct_w = (is_10 * w).sum() / w.sum() * 100
        else:
            pct_w = pct_uw
        print(f"  {alpha}: unweighted={pct_uw:.2f}%, {wt_col}={pct_w:.2f}%, paper={paper_1990.get(alpha,'')}")

# Check WVS wave 2 USA F063 directly
print("\n\nWVS Wave 2 USA F063:")
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','F063','S017'], low_memory=False)
usa_w2 = wvs[(wvs['COUNTRY_ALPHA']=='USA') & (wvs['S002VS']==2)]
valid = usa_w2[(usa_w2['F063']>=1) & (usa_w2['F063']<=10)]
if len(valid) > 0:
    print(f"  n={len(valid)}, pct10={((valid['F063']==10).mean()*100):.2f}%")
else:
    print("  No valid F063 data for USA wave 2")
    # Check if USA exists in wave 2 at all
    print(f"  USA wave 2 rows: {len(usa_w2)}")
    print(f"  F063 values: {usa_w2['F063'].value_counts().sort_index().to_dict()}")

# Check Finland wave 3
print("\nWVS Finland wave 3:")
fin_w3 = wvs[(wvs['COUNTRY_ALPHA']=='FIN') & (wvs['S002VS']==3)]
valid = fin_w3[(fin_w3['F063']>=1) & (fin_w3['F063']<=10)]
if len(valid) > 0:
    print(f"  n={len(valid)}, pct10={((valid['F063']==10).mean()*100):.2f}%")
else:
    print(f"  No valid F063. F063 values: {fin_w3['F063'].value_counts().sort_index().to_dict()}")

# Check GBR in WVS wave 3
print("\nWVS GBR wave 3:")
gbr_w3 = wvs[(wvs['COUNTRY_ALPHA']=='GBR') & (wvs['S002VS']==3)]
print(f"  rows: {len(gbr_w3)}")
if len(gbr_w3) > 0:
    valid = gbr_w3[(gbr_w3['F063']>=1) & (gbr_w3['F063']<=10)]
    print(f"  F063 valid: {len(valid)}")
    if len(valid) > 0:
        print(f"  pct10={((valid['F063']==10).mean()*100):.2f}%")
