"""Deep check of WVS Time Series for any European 1981 data or EVS integration"""
import pandas as pd
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, 'data', 'WVS_Time_Series_1981-2022_csv_v5_0.csv')

# Read just the columns we need
wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'F001', 'S020', 'S017'])

# Check ALL European countries we need (Belgium, Canada, France, Germany, etc.)
# Use S003 country codes
target_countries = {
    56: 'Belgium',
    124: 'Canada',
    250: 'France',
    276: 'Germany',
    826: 'Great Britain',
    352: 'Iceland',
    372: 'Ireland',
    909: 'Northern Ireland',
    380: 'Italy',
    528: 'Netherlands',
    578: 'Norway',
    724: 'Spain',
    752: 'Sweden',
    840: 'United States',
}

print("Checking target countries in WVS Time Series v5:")
print(f"Total rows in dataset: {len(wvs)}")
print()

for s003, name in sorted(target_countries.items(), key=lambda x: x[1]):
    sub = wvs[wvs['S003'] == s003]
    if len(sub) > 0:
        print(f"{name} (S003={s003}):")
        for wave in sorted(sub['S002VS'].unique()):
            wsub = sub[sub['S002VS'] == wave]
            years = sorted(wsub['S020'].unique())
            f001_valid = (wsub['F001'] > 0).sum()
            print(f"  Wave {wave}: n={len(wsub)}, F001_valid={f001_valid}, years={years}")
    else:
        # Try COUNTRY_ALPHA
        alpha_map = {56:'BEL', 124:'CAN', 250:'FRA', 276:'DEU', 826:'GBR',
                     352:'ISL', 372:'IRL', 909:'NIR', 380:'ITA', 528:'NLD',
                     578:'NOR', 724:'ESP', 752:'SWE', 840:'USA'}
        alpha = alpha_map.get(s003, '')
        sub2 = wvs[wvs['COUNTRY_ALPHA'] == alpha]
        if len(sub2) > 0:
            print(f"{name} (alpha={alpha}):")
            for wave in sorted(sub2['S002VS'].unique()):
                wsub = sub2[sub2['S002VS'] == wave]
                years = sorted(wsub['S020'].unique())
                f001_valid = (wsub['F001'] > 0).sum()
                print(f"  Wave {wave}: n={len(wsub)}, F001_valid={f001_valid}, years={years}")
        else:
            print(f"{name}: NOT FOUND (S003={s003}, alpha={alpha})")
    print()

# Check USA specifically - it HAS 1981 data in the paper (48%)
print("=" * 60)
print("Checking USA in detail:")
usa = wvs[(wvs['COUNTRY_ALPHA'] == 'USA') | (wvs['S003'] == 840)]
print(f"USA total rows: {len(usa)}")
for wave in sorted(usa['S002VS'].unique()):
    wsub = usa[usa['S002VS'] == wave]
    print(f"  Wave {wave}: n={len(wsub)}, years={sorted(wsub['S020'].unique())}")

# Check S003=840 vs COUNTRY_ALPHA=USA
print()
print("S003=840:", len(wvs[wvs['S003'] == 840]))
print("COUNTRY_ALPHA=USA:", len(wvs[wvs['COUNTRY_ALPHA'] == 'USA']))

# Also check if there's any data with S020=1981-1982 for any country not in wave 1
print()
print("=" * 60)
print("All entries with year 1981-1985 (any wave):")
early = wvs[(wvs['S020'] >= 1981) & (wvs['S020'] <= 1985)]
for alpha in sorted(early['COUNTRY_ALPHA'].unique()):
    sub = early[early['COUNTRY_ALPHA'] == alpha]
    print(f"  {alpha}: waves={sorted(sub['S002VS'].unique())}, years={sorted(sub['S020'].unique())}, n={len(sub)}")
