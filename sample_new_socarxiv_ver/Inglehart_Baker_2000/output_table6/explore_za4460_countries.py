import pandas as pd
import numpy as np

evs_long = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

# Country map for ZA4460
country_map = {
    40: 'Austria', 56: 'Belgium', 100: 'Bulgaria', 124: 'Canada',
    203: 'Czech Republic', 208: 'Denmark', 233: 'Estonia', 246: 'Finland',
    250: 'France', 276: 'Germany', 348: 'Hungary', 352: 'Iceland',
    372: 'Ireland', 380: 'Italy', 428: 'Latvia', 440: 'Lithuania',
    470: 'Malta', 528: 'Netherlands', 578: 'Norway', 616: 'Poland',
    620: 'Portugal', 642: 'Romania', 703: 'Slovakia', 705: 'Slovenia',
    724: 'Spain', 752: 'Sweden', 826: 'Great Britain', 840: 'United States',
    909: 'Northern Ireland'
}

# Church attendance variable in ZA4460 is q336
# Check against paper values and EVS_1990_wvs_format F063

evs_formatted = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)

# Focus on countries with discrepancies: Hungary, Italy, Finland
for code, name in sorted(country_map.items()):
    subset = evs_long[evs_long['country'] == code]
    if len(subset) == 0:
        continue

    valid = subset[subset['q336'].isin([1,2,3,4,5,6,7,8])]
    monthly = subset[subset['q336'].isin([1,2,3])]

    if len(valid) == 0:
        continue

    pct_q336 = round(len(monthly)/len(valid)*100)

    # Compare with EVS formatted
    alpha = {'Austria':'AUT','Belgium':'BEL','Bulgaria':'BGR','Canada':'CAN',
             'Czech Republic':'CZE','Denmark':'DNK','Estonia':'EST','Finland':'FIN',
             'France':'FRA','Germany':'DEU','Hungary':'HUN','Iceland':'ISL',
             'Ireland':'IRL','Italy':'ITA','Latvia':'LVA','Lithuania':'LTU',
             'Malta':'MLT','Netherlands':'NLD','Norway':'NOR','Poland':'POL',
             'Portugal':'PRT','Romania':'ROU','Slovakia':'SVK','Slovenia':'SVN',
             'Spain':'ESP','Sweden':'SWE','Great Britain':'GBR','United States':'USA',
             'Northern Ireland':'NIR'}.get(name, '')

    evs_sub = evs_formatted[evs_formatted['COUNTRY_ALPHA'] == alpha] if alpha else pd.DataFrame()
    pct_f063 = None
    if len(evs_sub) > 0:
        ev = evs_sub[evs_sub['F063'].isin([1,2,3,4,5,6,7,8])]
        em = evs_sub[evs_sub['F063'].isin([1,2,3])]
        if len(ev) > 0:
            pct_f063 = round(len(em)/len(ev)*100)

    same = "SAME" if pct_q336 == pct_f063 else "DIFF"
    print(f"{name:<22} q336={pct_q336:>3}%  F063={str(pct_f063) if pct_f063 is not None else 'N/A':>4}%  {same}")
