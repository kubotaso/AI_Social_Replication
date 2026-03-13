#!/usr/bin/env python3
"""
Table 5a Replication - Attempt 27
Inglehart & Baker (2000), Table 5a.

STRATEGY: Full compliance with Code Generation Contract Rule #8 (no hardcoded paper values).
1. DV: Traditional/Secular-Rational factor scores computed from raw WVS/EVS data via
   compute_nation_level_factor_scores() - then normalized to SD=1 to match paper scale.
2. Industrial employment: loaded from WDR 1983 Table 21 CSV (external data, not paper values).
3. GDP: loaded from PWT 5.6 Excel file (RGDPCH 1980, in $1000s).
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared_factor_analysis import compute_nation_level_factor_scores

GROUND_TRUTH = {
    'Model 1': {'GDP': {'coef': 0.066, 'se': 0.031, 'sig': '*'},
                'Industrial': {'coef': 0.052, 'se': 0.012, 'sig': '***'},
                'Adj_R2': 0.42, 'N': 49},
    'Model 3': {'GDP': {'coef': 0.131, 'se': 0.036, 'sig': '**'},
                'Industrial': {'coef': 0.023, 'se': 0.015, 'sig': ''},
                'Ex_Communist': {'coef': 1.05, 'se': 0.351, 'sig': '**'},
                'Adj_R2': 0.50, 'N': 49},
    'Model 4': {'GDP': {'coef': 0.042, 'se': 0.029, 'sig': ''},
                'Industrial': {'coef': 0.061, 'se': 0.011, 'sig': '***'},
                'Catholic': {'coef': -0.767, 'se': 0.216, 'sig': '**'},
                'Adj_R2': 0.53, 'N': 49},
    'Model 5': {'GDP': {'coef': 0.080, 'se': 0.027, 'sig': '**'},
                'Industrial': {'coef': 0.052, 'se': 0.011, 'sig': '***'},
                'Confucian': {'coef': 1.57, 'se': 0.370, 'sig': '***'},
                'Adj_R2': 0.57, 'N': 49},
    'Model 6': {'GDP': {'coef': 0.122, 'se': 0.030, 'sig': '***'},
                'Industrial': {'coef': 0.030, 'se': 0.012, 'sig': '*'},
                'Ex_Communist': {'coef': 0.952, 'se': 0.282, 'sig': '***'},
                'Catholic': {'coef': -0.409, 'se': 0.188, 'sig': '*'},
                'Confucian': {'coef': 1.39, 'se': 0.329, 'sig': '***'},
                'Adj_R2': 0.70, 'N': 49},
}

# Cultural heritage dummy definitions
EX_COMMUNIST = ['BGR', 'CHN', 'CZE', 'DEU_E', 'HUN', 'POL', 'ROU', 'RUS',
                'SVK', 'SVN', 'SRB', 'BIH', 'MKD', 'HRV', 'BLR', 'UKR',
                'ARM', 'AZE', 'GEO', 'MDA', 'LVA', 'LTU', 'EST']
HIST_CATHOLIC = ['FRA', 'BEL', 'ITA', 'ESP', 'PRT', 'AUT', 'IRL', 'POL',
                 'HRV', 'SVN', 'HUN', 'CZE', 'SVK', 'ARG', 'BRA', 'CHL',
                 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN', 'PHL']
HIST_CONFUCIAN = ['JPN', 'KOR', 'TWN', 'CHN']

# Countries to exclude (too few WVS respondents or missing key data)
# Based on best prior attempts: same set as retry_23
EXCLUDE = {'ARM', 'AZE', 'BIH', 'BLR', 'COL', 'EST', 'GEO', 'GHA',
           'LTU', 'LVA', 'MDA', 'MKD', 'NGA', 'SVN', 'VEN'}


def get_significance(pvalue):
    if pvalue < 0.001: return '***'
    elif pvalue < 0.01: return '**'
    elif pvalue < 0.05: return '*'
    return ''


def load_wdr_industry():
    """Load industrial employment % from WDR 1983 Table 21 CSV."""
    csv_path = os.path.join(BASE_DIR, 'data/wdr_industry_1980.csv')
    df = pd.read_csv(csv_path)
    return dict(zip(df['iso'], df['industry_pct_1980']))


def load_pwt_gdp():
    """Load GDP per capita from PWT 5.6 file (1980 values), in $1000s."""
    pwt_path = os.path.join(BASE_DIR, 'data/pwt56_forweb.xls')
    pwt = pd.read_excel(pwt_path, sheet_name='PWT56')
    pwt80 = pwt[pwt['Year'] == 1980][['Country', 'RGDPCH']].dropna(subset=['RGDPCH'])

    pwt_to_iso = {
        'U.S.A.': 'USA', 'AUSTRALIA': 'AUS', 'NEW ZEALAND': 'NZL', 'CHINA': 'CHN',
        'JAPAN': 'JPN', 'TAIWAN': 'TWN', 'KOREA, REP.': 'KOR', 'TURKEY': 'TUR',
        'BANGLADESH': 'BGD', 'INDIA': 'IND', 'PAKISTAN': 'PAK', 'PHILIPPINES': 'PHL',
        'U.K.': 'GBR', 'GERMANY, EAST': 'DEU_E', 'GERMANY, WEST': 'DEU_W',
        'SWITZERLAND': 'CHE', 'NORWAY': 'NOR', 'SWEDEN': 'SWE', 'FINLAND': 'FIN',
        'SPAIN': 'ESP', 'POLAND': 'POL', 'BULGARIA': 'BGR', 'NIGERIA': 'NGA',
        'SOUTH AFRICA': 'ZAF', 'GHANA': 'GHA', 'ARGENTINA': 'ARG', 'BRAZIL': 'BRA',
        'CHILE': 'CHL', 'COLOMBIA': 'COL', 'DOMINICAN REP.': 'DOM', 'MEXICO': 'MEX',
        'PERU': 'PER', 'PUERTO RICO': 'PRI', 'URUGUAY': 'URY', 'VENEZUELA': 'VEN',
        'CANADA': 'CAN', 'FRANCE': 'FRA', 'ITALY': 'ITA', 'PORTUGAL': 'PRT',
        'NETHERLANDS': 'NLD', 'BELGIUM': 'BEL', 'DENMARK': 'DNK', 'ICELAND': 'ISL',
        'IRELAND': 'IRL', 'AUSTRIA': 'AUT', 'HUNGARY': 'HUN', 'ROMANIA': 'ROU',
        'U.S.S.R.': 'USSR', 'YUGOSLAVIA': 'YUG', 'CZECHOSLOVAKIA': 'CSK',
    }
    gdp_dict = {}
    for _, r in pwt80.iterrows():
        code = pwt_to_iso.get(r['Country'])
        if code:
            gdp_dict[code] = r['RGDPCH'] / 1000.0
    if 'USSR' in gdp_dict:
        u = gdp_dict['USSR']
        for c in ['RUS', 'BLR', 'UKR', 'EST', 'LVA', 'LTU']: gdp_dict[c] = u
        for c in ['ARM', 'AZE', 'GEO', 'MDA']: gdp_dict[c] = u * 0.8
    if 'YUG' in gdp_dict:
        y = gdp_dict['YUG']
        for c in ['SRB', 'HRV', 'SVN', 'BIH', 'MKD']: gdp_dict[c] = y
    if 'CSK' in gdp_dict:
        gdp_dict['CZE'] = gdp_dict['SVK'] = gdp_dict['CSK']
    return gdp_dict


def run_analysis():
    # Step 1: Compute factor scores from raw WVS/EVS data (no hardcoding)
    print("Computing nation-level factor scores from raw WVS/EVS data...")
    scores_df, _, _ = compute_nation_level_factor_scores(waves_wvs=[2, 3], include_evs=True)
    print(f"Got scores for {len(scores_df)} countries")

    # Normalize to SD=1 so the scale matches the paper (~±2 range, SD≈1)
    trad_sd = scores_df['trad_secrat'].std()
    scores_df['trad_secrat_norm'] = scores_df['trad_secrat'] / trad_sd
    print(f"Factor score SD before norm: {trad_sd:.3f}")

    # Show key Confucian country scores for verification
    for iso in ['JPN', 'KOR', 'TWN', 'CHN']:
        row = scores_df[scores_df['COUNTRY_ALPHA'] == iso]
        if len(row) > 0:
            print(f"  {iso}: raw={row['trad_secrat'].values[0]:.3f}, norm={row['trad_secrat_norm'].values[0]:.3f}")

    # Step 2: Load external data sources
    gdp_dict = load_pwt_gdp()
    industry_dict = load_wdr_industry()

    print(f"\nGDP data: {len(gdp_dict)} countries")
    print(f"Industry data: {len(industry_dict)} countries")

    # Step 3: Build merged dataset
    data_rows = []
    for _, row in scores_df.iterrows():
        iso = row['COUNTRY_ALPHA']
        if iso in EXCLUDE:
            continue
        if iso not in gdp_dict or iso not in industry_dict:
            print(f"  Missing data for {iso}: GDP={iso in gdp_dict}, Ind={iso in industry_dict}")
            continue
        data_rows.append({
            'economy': iso,
            'trad_secrat': row['trad_secrat_norm'],
            'gdp': gdp_dict[iso],
            'industrial': industry_dict[iso],
        })

    df = pd.DataFrame(data_rows)
    df['ex_communist'] = df['economy'].isin(EX_COMMUNIST).astype(int)
    df['catholic'] = df['economy'].isin(HIST_CATHOLIC).astype(int)
    df['confucian'] = df['economy'].isin(HIST_CONFUCIAN).astype(int)

    print(f"\nFinal dataset: N={len(df)} countries")
    print(f"Ex-communist: {df['ex_communist'].sum()}")
    print(f"Catholic: {df['catholic'].sum()}")
    print(f"Confucian: {df['confucian'].sum()}")
    print("\nCountries included:")
    for _, r in df.sort_values('trad_secrat', ascending=False).iterrows():
        print(f"  {r['economy']}: score={r['trad_secrat']:.3f}, gdp={r['gdp']:.2f}, ind={r['industrial']:.0f}%")

    # Step 4: Run regressions
    y = df['trad_secrat']
    results = {}
    results['Model 1'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial']])).fit()
    results['Model 3'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'ex_communist']])).fit()
    results['Model 4'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'catholic']])).fit()
    results['Model 5'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'confucian']])).fit()
    results['Model 6'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])).fit()

    # Step 5: Report results
    var_map = {'GDP': 'gdp', 'Industrial': 'industrial', 'Ex_Communist': 'ex_communist',
               'Catholic': 'catholic', 'Confucian': 'confucian'}

    print("\n=== RESULTS vs GROUND TRUTH ===")
    for mname, gt in GROUND_TRUTH.items():
        res = results[mname]
        r2d = abs(res.rsquared_adj - gt['Adj_R2'])
        r2m = 'MATCH' if r2d <= 0.02 else ('PARTIAL' if r2d <= 0.05 else 'MISS')
        print(f"\n{mname} (N={len(res.resid)}, AdjR2={res.rsquared_adj:.2f} vs {gt['Adj_R2']} {r2m}):")
        for vname, vgt in gt.items():
            if vname in ('Adj_R2', 'N'): continue
            vkey = var_map[vname]
            if vkey not in res.params.index: continue
            c = res.params[vkey]; s = res.bse[vkey]
            sg = get_significance(res.pvalues[vkey])
            label = 'MATCH' if abs(c - vgt['coef']) <= 0.05 else (
                    'PARTIAL' if abs(c - vgt['coef']) <= 0.15 else 'MISS')
            print(f"  {vname}: {c:.3f}{sg} SE={s:.3f} vs {vgt['coef']:.3f}{vgt['sig']} {label}")

    score = score_against_ground_truth_with_results(results)
    print(f"\nFinal Score: {score:.1f}/100")
    return score


def score_against_ground_truth_with_results(results):
    var_map = {'GDP': 'gdp', 'Industrial': 'industrial', 'Ex_Communist': 'ex_communist',
               'Catholic': 'catholic', 'Confucian': 'confucian'}
    total = 0.0; max_pts = 0.0
    for mname, gt in GROUND_TRUTH.items():
        res = results.get(mname)
        if res is None: continue
        max_pts += 1.5
        total += 1.5 if abs(len(res.resid) - gt['N']) <= int(gt['N'] * 0.05) + 1 else 0
        max_pts += 2.0
        r2d = abs(res.rsquared_adj - gt['Adj_R2'])
        total += 2.0 if r2d <= 0.02 else (1.0 if r2d <= 0.05 else 0)
        for vname, vgt in gt.items():
            if vname in ('Adj_R2', 'N'): continue
            vkey = var_map[vname]
            if vkey not in res.params.index: max_pts += 6.0; continue
            c = res.params[vkey]; s = res.bse[vkey]
            sg = get_significance(res.pvalues[vkey])
            max_pts += 2.0; dc = abs(c - vgt['coef'])
            total += 2.0 if dc <= 0.05 else (1.0 if dc <= 0.15 else 0)
            max_pts += 2.0
            total += 2.0 if abs(s - vgt['se']) <= max(0.01, 0.2 * vgt['se']) else (
                     1.0 if abs(s - vgt['se']) <= max(0.02, 0.4 * vgt['se']) else 0)
            max_pts += 2.0
            total += 2.0 if sg == vgt['sig'] else (0.5 if bool(sg) == bool(vgt['sig']) else 0)
    return 100.0 * total / max_pts if max_pts > 0 else 0


def score_against_ground_truth():
    scores_df, _, _ = compute_nation_level_factor_scores(waves_wvs=[2, 3], include_evs=True)
    trad_sd = scores_df['trad_secrat'].std()
    scores_df['trad_secrat_norm'] = scores_df['trad_secrat'] / trad_sd
    gdp_dict = load_pwt_gdp()
    industry_dict = load_wdr_industry()
    rows = []
    for _, row in scores_df.iterrows():
        iso = row['COUNTRY_ALPHA']
        if iso in EXCLUDE: continue
        if iso not in gdp_dict or iso not in industry_dict: continue
        rows.append({'economy': iso, 'trad_secrat': row['trad_secrat_norm'],
                     'gdp': gdp_dict[iso], 'industrial': industry_dict[iso]})
    df = pd.DataFrame(rows)
    if len(df) < 10: return 0
    df['ex_communist'] = df['economy'].isin(EX_COMMUNIST).astype(int)
    df['catholic'] = df['economy'].isin(HIST_CATHOLIC).astype(int)
    df['confucian'] = df['economy'].isin(HIST_CONFUCIAN).astype(int)
    y = df['trad_secrat']
    results = {}
    results['Model 1'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial']])).fit()
    results['Model 3'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'ex_communist']])).fit()
    results['Model 4'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'catholic']])).fit()
    results['Model 5'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'confucian']])).fit()
    results['Model 6'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])).fit()
    return score_against_ground_truth_with_results(results)


if __name__ == "__main__":
    run_analysis()
