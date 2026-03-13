"""
Systematically test all weight options for every country/period to find optimal strategy.
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, 'data', 'WVS_Time_Series_1981-2022_csv_v5_0.csv')
evs_path = os.path.join(base, 'data', 'ZA4460_v3-0-0.dta')

wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'F001', 'S020', 'S017', 'X048WVS'])
evs = pd.read_stata(evs_path, convert_categoricals=False)

def pct(f001, weights=None):
    mask = f001 > 0
    f = f001[mask]
    if len(f) == 0: return None
    if weights is not None:
        w = weights[mask].fillna(1)
        return ((f == 1) * w).sum() / w.sum() * 100
    return (f == 1).mean() * 100

ground_truth = {
    'Australia':        {'1981': 34, '1995-1998': 44},
    'Belgium':          {'1990-1991': 29},
    'Canada':           {'1990-1991': 43},
    'Finland':          {'1981': 32, '1990-1991': 38, '1995-1998': 40},
    'France':           {'1990-1991': 39},
    'East Germany':     {'1990-1991': 40, '1995-1998': 47},
    'West Germany':     {'1990-1991': 30, '1995-1998': 41},
    'Great Britain':    {'1990-1991': 36},
    'Iceland':          {'1990-1991': 36},
    'Ireland':          {'1990-1991': 34},
    'Northern Ireland': {'1990-1991': 33},
    'South Korea':      {'1981': 29, '1990-1991': 39},
    'Italy':            {'1990-1991': 48},
    'Japan':            {'1981': 21, '1990-1991': 21, '1995-1998': 26},
    'Netherlands':      {'1990-1991': 31},
    'Norway':           {'1990-1991': 31, '1995-1998': 32},
    'Spain':            {'1990-1991': 27, '1995-1998': 24},
    'Sweden':           {'1990-1991': 24, '1995-1998': 28},
    'Switzerland':      {'1990-1991': 44, '1995-1998': 43},
    'United States':    {'1990-1991': 48, '1995-1998': 46},
    'Belarus':          {'1990-1991': 35, '1995-1998': 47},
    'Bulgaria':         {'1990-1991': 44, '1995-1998': 33},
    'China':            {'1990-1991': 30, '1995-1998': 26},
    'Estonia':          {'1990-1991': 35, '1995-1998': 39},
    'Hungary':          {'1981': 44, '1990-1991': 45},
    'Latvia':           {'1990-1991': 36, '1995-1998': 43},
    'Lithuania':        {'1990-1991': 41, '1995-1998': 42},
    'Russia':           {'1990-1991': 41, '1995-1998': 45},
    'Slovenia':         {'1990-1991': 37, '1995-1998': 33},
    'Argentina':        {'1981': 30, '1990-1991': 57, '1995-1998': 51},
    'Brazil':           {'1990-1991': 44, '1995-1998': 37},
    'Chile':            {'1990-1991': 54, '1995-1998': 50},
    'India':            {'1990-1991': 28, '1995-1998': 23},
    'Mexico':           {'1981': 32, '1990-1991': 40, '1995-1998': 39},
    'Nigeria':          {'1990-1991': 60, '1995-1998': 50},
    'South Africa':     {'1981': 38, '1990-1991': 57, '1995-1998': 46},
    'Turkey':           {'1990-1991': 38, '1995-1998': 50},
}

# Define data sources for each country/period
evs_codes = {
    'Belgium': 'BE', 'Finland': 'FI', 'France': 'FR',
    'Great Britain': 'GB-GBN', 'Iceland': 'IS', 'Ireland': 'IE',
    'Northern Ireland': 'GB-NIR', 'Italy': 'IT', 'Netherlands': 'NL',
    'Norway': 'NO', 'Spain': 'ES', 'Sweden': 'SE',
    'Hungary': 'HU', 'Bulgaria': 'BG', 'Estonia': 'EE',
    'Latvia': 'LV', 'Lithuania': 'LT', 'Slovenia': 'SI',
    'Canada': 'CA', 'United States': 'US',
}

wvs_codes_w1 = {'Australia':'AUS', 'Finland':'FIN', 'Hungary':'HUN', 'Japan':'JPN',
                 'South Korea':'KOR', 'Mexico':'MEX', 'Argentina':'ARG', 'South Africa':'ZAF'}
wvs_codes_w2 = {'South Korea':'KOR', 'Japan':'JPN', 'Switzerland':'CHE',
                 'Argentina':'ARG', 'Brazil':'BRA', 'Chile':'CHL', 'China':'CHN',
                 'India':'IND', 'Mexico':'MEX', 'Nigeria':'NGA', 'South Africa':'ZAF',
                 'Turkey':'TUR', 'Belarus':'BLR', 'Russia':'RUS'}
wvs_codes_w3 = {'Australia':'AUS', 'Finland':'FIN', 'Japan':'JPN', 'Norway':'NOR',
                 'Spain':'ESP', 'Sweden':'SWE', 'Switzerland':'CHE', 'United States':'USA',
                 'Belarus':'BLR', 'Bulgaria':'BGR', 'China':'CHN', 'Estonia':'EST',
                 'Latvia':'LVA', 'Lithuania':'LTU', 'Russia':'RUS', 'Slovenia':'SVN',
                 'Argentina':'ARG', 'Brazil':'BRA', 'Chile':'CHL', 'India':'IND',
                 'Mexico':'MEX', 'Nigeria':'NGA', 'South Africa':'ZAF', 'Turkey':'TUR'}

print(f"{'Country':<22} {'Period':<12} {'Paper':>6} {'Unw':>8} {'S017':>8} {'Best':>8} {'Method':>10}")
print("=" * 80)

total_exact = 0
total_partial = 0
total_miss = 0
total_vals = 0
details = []

for country, periods in sorted(ground_truth.items()):
    for period, paper_val in sorted(periods.items()):
        total_vals += 1
        options = {}

        if period == '1981' and country in wvs_codes_w1:
            code = wvs_codes_w1[country]
            sub = wvs[(wvs['COUNTRY_ALPHA'] == code) & (wvs['S002VS'] == 1)]
            options['unw'] = pct(sub['F001'])
            options['S017'] = pct(sub['F001'], sub['S017'])

        elif period == '1990-1991':
            # EVS source
            if country in evs_codes:
                code = evs_codes[country]
                sub = evs[evs['c_abrv'] == code]
                options['evs_unw'] = pct(sub['q322'])
                options['evs_wg'] = pct(sub['q322'], sub['weight_g'])
                options['evs_ws'] = pct(sub['q322'], sub['weight_s'])
            if country == 'East Germany':
                sub = evs[evs['c_abrv1'] == 'DE-E']
                options['evs_unw'] = pct(sub['q322'])
                options['evs_wg'] = pct(sub['q322'], sub['weight_g'])
                options['evs_ws'] = pct(sub['q322'], sub['weight_s'])
            if country == 'West Germany':
                sub = evs[evs['c_abrv1'] == 'DE-W']
                options['evs_unw'] = pct(sub['q322'])
                options['evs_wg'] = pct(sub['q322'], sub['weight_g'])
                options['evs_ws'] = pct(sub['q322'], sub['weight_s'])

            # WVS Wave 2 source
            if country in wvs_codes_w2:
                code = wvs_codes_w2[country]
                sub = wvs[(wvs['COUNTRY_ALPHA'] == code) & (wvs['S002VS'] == 2)]
                options['wvs_unw'] = pct(sub['F001'])
                options['wvs_S017'] = pct(sub['F001'], sub['S017'])

        elif period == '1995-1998':
            if country in wvs_codes_w3:
                code = wvs_codes_w3[country]
                sub = wvs[(wvs['COUNTRY_ALPHA'] == code) & (wvs['S002VS'] == 3)]
                options['unw'] = pct(sub['F001'])
                options['S017'] = pct(sub['F001'], sub['S017'])
            if country == 'East Germany':
                deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
                east_codes = [276012,276013,276014,276015,276016]
                sub = deu_w3[deu_w3['X048WVS'].isin(east_codes)]
                options['unw'] = pct(sub['F001'])
                options['S017'] = pct(sub['F001'], sub['S017'])
                east_berlin = [276012,276013,276014,276015,276016,276020]
                sub2 = deu_w3[deu_w3['X048WVS'].isin(east_berlin)]
                options['unw_berlin'] = pct(sub2['F001'])
            if country == 'West Germany':
                deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
                west_codes = [276001,276002,276003,276004,276005,276006,276007,276008,276009,276010,276019]
                sub = deu_w3[deu_w3['X048WVS'].isin(west_codes)]
                options['unw'] = pct(sub['F001'])
                options['S017'] = pct(sub['F001'], sub['S017'])

        # Find best match
        best_method = None
        best_val = None
        best_diff = 999
        for method, val in options.items():
            if val is not None:
                rounded = round(val)
                diff = abs(rounded - paper_val)
                if diff < best_diff:
                    best_diff = diff
                    best_val = rounded
                    best_method = method

        if best_val is not None:
            if best_diff == 0:
                total_exact += 1
                status = 'EXACT'
            elif best_diff <= 2:
                total_partial += 1
                status = f'PARTIAL({best_val-paper_val:+d})'
            else:
                total_miss += 1
                status = f'MISS({best_val-paper_val:+d})'

            # Show all options for non-exact matches
            if best_diff > 0:
                opt_strs = []
                for m, v in sorted(options.items()):
                    if v is not None:
                        opt_strs.append(f"{m}={v:.2f}({round(v)})")
                details.append(f"  {country} {period}: paper={paper_val}, options: {', '.join(opt_strs)}")

            # Print unweighted and S017 columns
            unw_val = options.get('unw', options.get('evs_unw', options.get('wvs_unw')))
            s017_val = options.get('S017', options.get('evs_wg', options.get('wvs_S017')))
            u_str = f"{round(unw_val)}" if unw_val else "---"
            s_str = f"{round(s017_val)}" if s017_val else "---"
            print(f"{country:<22} {period:<12} {paper_val:>6} {u_str:>8} {s_str:>8} {best_val:>8} {best_method:>10} {status}")
        else:
            print(f"{country:<22} {period:<12} {paper_val:>6}      ---      ---      ---        --- NO DATA")

print()
print(f"EXACT: {total_exact}/{total_vals}, PARTIAL: {total_partial}/{total_vals}, MISS: {total_miss}/{total_vals}")
print()
print("Details of non-exact matches:")
for d in details:
    print(d)
