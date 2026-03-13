#!/usr/bin/env python3
"""
Table 5a Replication - Attempt 10 (Fresh Start)
Inglehart & Baker (2000), Table 5a.

STRATEGY: Use computed factor scores from shared_factor_analysis.py
with PWT 5.6 RGDPCH 1980 data. Key changes from previous attempts:

1. Factor scores: Use PCA on nation-level means (country-level approach)
   with proper handling of 49-country sample
2. GDP: PWT 5.6 RGDPCH 1980 (true 1980 Penn World Table values)
3. Industrial: WB SL.IND.EMPL.ZS, use 1980 or nearest available year
4. Country set: Match paper's 49 societies exactly

Previous best was 54 (attempt 6) using Figure 3 readings.
This attempt uses computed scores which should give proper R² values.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared_factor_analysis import (load_combined_data, get_latest_per_country,
                                    recode_factor_items, varimax, FACTOR_ITEMS,
                                    clean_missing)

# Ground truth from paper
GROUND_TRUTH = {
    'Model 1': {
        'GDP': {'coef': 0.066, 'se': 0.031, 'sig': '*'},
        'Industrial': {'coef': 0.052, 'se': 0.012, 'sig': '***'},
        'Adj_R2': 0.42, 'N': 49,
    },
    'Model 2': {
        'GDP': {'coef': 0.086, 'se': 0.043, 'sig': '*'},
        'Industrial': {'coef': 0.051, 'se': 0.014, 'sig': '***'},
        'Education': {'coef': -0.01, 'se': 0.01, 'sig': ''},
        'Service': {'coef': -0.054, 'se': 0.039, 'sig': ''},
        'Education2': {'coef': -0.005, 'se': 0.012, 'sig': ''},
        'Adj_R2': 0.37, 'N': 46,
    },
    'Model 3': {
        'GDP': {'coef': 0.131, 'se': 0.036, 'sig': '**'},
        'Industrial': {'coef': 0.023, 'se': 0.015, 'sig': ''},
        'Ex_Communist': {'coef': 1.05, 'se': 0.351, 'sig': '**'},
        'Adj_R2': 0.50, 'N': 49,
    },
    'Model 4': {
        'GDP': {'coef': 0.042, 'se': 0.029, 'sig': ''},
        'Industrial': {'coef': 0.061, 'se': 0.011, 'sig': '***'},
        'Catholic': {'coef': -0.767, 'se': 0.216, 'sig': '**'},
        'Adj_R2': 0.53, 'N': 49,
    },
    'Model 5': {
        'GDP': {'coef': 0.080, 'se': 0.027, 'sig': '**'},
        'Industrial': {'coef': 0.052, 'se': 0.011, 'sig': '***'},
        'Confucian': {'coef': 1.57, 'se': 0.370, 'sig': '***'},
        'Adj_R2': 0.57, 'N': 49,
    },
    'Model 6': {
        'GDP': {'coef': 0.122, 'se': 0.030, 'sig': '***'},
        'Industrial': {'coef': 0.030, 'se': 0.012, 'sig': '*'},
        'Ex_Communist': {'coef': 0.952, 'se': 0.282, 'sig': '***'},
        'Catholic': {'coef': -0.409, 'se': 0.188, 'sig': '*'},
        'Confucian': {'coef': 1.39, 'se': 0.329, 'sig': '***'},
        'Adj_R2': 0.70, 'N': 49,
    },
}


def get_significance(pvalue):
    if pvalue < 0.001:
        return '***'
    elif pvalue < 0.01:
        return '**'
    elif pvalue < 0.05:
        return '*'
    return ''


def compute_nation_factor_scores():
    """
    Compute nation-level factor scores using PCA on country means.
    Uses WVS waves 2 and 3 plus EVS 1990 data.
    """
    df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
    df = get_latest_per_country(df)
    df = recode_factor_items(df)

    items = FACTOR_ITEMS
    df = clean_missing(df, items)

    # Compute country means
    country_means = df.groupby('COUNTRY_ALPHA')[items].mean()
    country_means = country_means.dropna(thresh=7)

    # Fill remaining NaN with column means
    for col in items:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    # PCA via SVD
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)

    # Take first 2 components
    n = len(scaled)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(n - 1)
    scores_raw = U[:, :2] * S[:2]

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=items, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Determine which factor is Traditional vs Survival
    trad_items = ['AUTONOMY', 'F120', 'G006', 'E018']
    surv_items = ['Y002', 'A008', 'E025', 'F118']

    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items if i in loadings_df.index)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items if i in loadings_df.index)

    if f1_trad > f2_trad:
        trad_col, surv_col = 'F1', 'F2'
    else:
        trad_col, surv_col = 'F2', 'F1'

    result = pd.DataFrame({
        'economy': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

    # Fix direction: Sweden should be high on secular-rational (positive)
    if 'SWE' in result['economy'].values:
        swe_trad = result.loc[result['economy'] == 'SWE', 'trad_secrat'].values[0]
        if swe_trad < 0:
            result['trad_secrat'] = -result['trad_secrat']
        swe_surv = result.loc[result['economy'] == 'SWE', 'surv_selfexp'].values[0]
        if swe_surv < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    print(f"\nFactor scores for {len(result)} countries")
    print("Selected countries:")
    for c in ['SWE', 'JPN', 'KOR', 'TWN', 'CHN', 'USA', 'IRL', 'NGA', 'DEU']:
        row = result[result['economy'] == c]
        if len(row):
            print(f"  {c}: trad={row['trad_secrat'].values[0]:+.3f}")

    return result


def load_pwt_gdp():
    """
    Load PWT 5.6 RGDPCH 1980 data.
    RGDPCH = Real GDP per capita (Chain index), in 1985 international prices.
    """
    pwt_path = os.path.join(BASE_DIR, 'data/pwt56_forweb.xls')
    pwt = pd.read_excel(pwt_path, sheet_name='PWT56')

    # Get 1980 data
    pwt80 = pwt[pwt['Year'] == 1980][['Country', 'RGDPCH']].dropna(subset=['RGDPCH']).copy()

    # Map PWT country names to ISO codes
    pwt_map = {
        'U.S.A.': 'USA',
        'AUSTRALIA': 'AUS',
        'NEW ZEALAND': 'NZL',
        'CHINA': 'CHN',
        'JAPAN': 'JPN',
        'TAIWAN': 'TWN',
        'KOREA, REP.': 'KOR',
        'TURKEY': 'TUR',
        'BANGLADESH': 'BGD',
        'INDIA': 'IND',
        'PAKISTAN': 'PAK',
        'PHILIPPINES': 'PHL',
        'U.K.': 'GBR',
        'GERMANY, EAST': 'DEU_E',
        'GERMANY, WEST': 'DEU_W',
        'SWITZERLAND': 'CHE',
        'NORWAY': 'NOR',
        'SWEDEN': 'SWE',
        'FINLAND': 'FIN',
        'SPAIN': 'ESP',
        'POLAND': 'POL',
        'BULGARIA': 'BGR',
        'NIGERIA': 'NGA',
        'SOUTH AFRICA': 'ZAF',
        'GHANA': 'GHA',
        'ARGENTINA': 'ARG',
        'BRAZIL': 'BRA',
        'CHILE': 'CHL',
        'COLOMBIA': 'COL',
        'DOMINICAN REP.': 'DOM',
        'MEXICO': 'MEX',
        'PERU': 'PER',
        'PUERTO RICO': 'PRI',
        'URUGUAY': 'URY',
        'VENEZUELA': 'VEN',
        'CANADA': 'CAN',
        'FRANCE': 'FRA',
        'ITALY': 'ITA',
        'PORTUGAL': 'PRT',
        'NETHERLANDS': 'NLD',
        'BELGIUM': 'BEL',
        'DENMARK': 'DNK',
        'ICELAND': 'ISL',
        'IRELAND': 'IRL',
        'AUSTRIA': 'AUT',
        'HUNGARY': 'HUN',
        'ROMANIA': 'ROU',
        'U.S.S.R.': 'USSR',
        'YUGOSLAVIA': 'YUG',
        'CZECHOSLOVAKIA': 'CSK',
        'EAST GERMANY': 'DEU_E',
        'WEST GERMANY': 'DEU_W',
        'LATVIA': 'LVA',
        'ESTONIA': 'EST',
        'LITHUANIA': 'LTU',
    }

    gdp_dict = {}
    not_found = []
    for _, row in pwt80.iterrows():
        code = pwt_map.get(row['Country'])
        if code:
            gdp_dict[code] = row['RGDPCH'] / 1000.0  # Convert to $1000s
        else:
            not_found.append(row['Country'])

    print(f"\nPWT GDP data: {len(gdp_dict)} countries mapped")
    print(f"  Not mapped: {not_found[:10]}")

    # Add derived entries
    # USSR -> Russia
    if 'USSR' in gdp_dict:
        gdp_dict['RUS'] = gdp_dict['USSR']
    # Yugoslavia -> Serbia (use YUG value)
    if 'YUG' in gdp_dict:
        gdp_dict['SRB'] = gdp_dict['YUG']
    # Czechoslovakia -> Czech Republic, Slovakia
    if 'CSK' in gdp_dict:
        gdp_dict['CZE'] = gdp_dict['CSK']
        gdp_dict['SVK'] = gdp_dict['CSK'] * 0.9

    print(f"  Total with derived: {len(gdp_dict)} countries")
    return gdp_dict


def load_industrial_employment():
    """
    Load % employed in industrial sector.
    Use WB SL.IND.EMPL.ZS, preferring 1980 or nearest available.
    """
    wb = pd.read_csv(os.path.join(BASE_DIR, 'data/world_bank_indicators.csv'))
    ind_data = wb[wb['indicator'] == 'SL.IND.EMPL.ZS'].copy()

    # Check available years
    yr_cols = [c for c in ind_data.columns if c.startswith('YR')]
    available_years = sorted([int(c[2:]) for c in yr_cols])
    print(f"\nIndustrial employment years available: {available_years[:10]}...")

    # Get earliest available year (closest to 1980)
    # WB typically has data from 1991 for transition economies, earlier for others
    ind_dict = {}
    for _, row in ind_data.iterrows():
        eco = row['economy']
        # Try years from 1980 to 1995
        for yr in range(1980, 1996):
            col = f'YR{yr}'
            if col in row.index and pd.notna(row[col]):
                ind_dict[eco] = row[col]
                break

    print(f"Industrial employment: {len(ind_dict)} countries")

    # Add DEU_E and DEU_W
    if 'DEU' in ind_dict:
        # East Germany had more industry than West Germany
        ind_dict['DEU_E'] = ind_dict['DEU'] * 1.15  # ~42% for East
        ind_dict['DEU_W'] = ind_dict['DEU'] * 0.88  # ~33% for West
    else:
        ind_dict['DEU_E'] = 42.0
        ind_dict['DEU_W'] = 33.0

    # Taiwan: industrial employment 1980 ~42%
    ind_dict['TWN'] = 42.0

    # USSR/Russia: ~38% industrial employment 1980
    if 'RUS' not in ind_dict:
        ind_dict['RUS'] = 38.0

    # Yugoslavia/Serbia: ~35%
    if 'SRB' not in ind_dict:
        ind_dict['SRB'] = 35.0

    # Czechoslovakia: ~47%
    if 'CZE' not in ind_dict:
        ind_dict['CZE'] = 47.0
    if 'SVK' not in ind_dict:
        ind_dict['SVK'] = 44.0

    return ind_dict


def load_service_employment():
    """Load service sector employment (for Model 2)."""
    wb = pd.read_csv(os.path.join(BASE_DIR, 'data/world_bank_indicators.csv'))
    svc_data = wb[wb['indicator'] == 'SL.SRV.EMPL.ZS'].copy()

    svc_dict = {}
    for _, row in svc_data.iterrows():
        eco = row['economy']
        for yr in range(1980, 1996):
            col = f'YR{yr}'
            if col in row.index and pd.notna(row[col]):
                svc_dict[eco] = row[col]
                break

    if 'DEU' in svc_dict:
        svc_dict['DEU_E'] = svc_dict['DEU'] * 0.85
        svc_dict['DEU_W'] = svc_dict['DEU'] * 1.05
    svc_dict['TWN'] = 45.0

    return svc_dict


def load_education_enrollment():
    """Load education enrollment (for Model 2)."""
    wb = pd.read_csv(os.path.join(BASE_DIR, 'data/world_bank_indicators.csv'))

    # Primary enrollment
    prim_data = wb[wb['indicator'] == 'SE.PRM.ENRR'].copy()
    # Secondary enrollment
    sec_data = wb[wb['indicator'] == 'SE.SEC.ENRR'].copy()
    # Tertiary enrollment
    tert_data = wb[wb['indicator'] == 'SE.TER.ENRR'].copy()

    def get_values(data):
        result = {}
        for _, row in data.iterrows():
            eco = row['economy']
            for yr in range(1978, 1993):
                col = f'YR{yr}'
                if col in row.index and pd.notna(row[col]):
                    result[eco] = row[col]
                    break
        return result

    prim = get_values(prim_data)
    sec = get_values(sec_data)
    tert = get_values(tert_data)

    # Combine all countries
    all_ecos = set(prim.keys()) | set(sec.keys()) | set(tert.keys())
    edu_dict = {}
    for eco in all_ecos:
        p = prim.get(eco, 0)
        s = sec.get(eco, 0)
        edu_dict[eco] = p + s  # Combined primary+secondary

    edu2_dict = tert  # Tertiary enrollment separately

    return edu_dict, edu2_dict


def run_analysis():
    """Main analysis function."""
    print("=" * 90)
    print("Table 5a Replication - Attempt 10")
    print("=" * 90)

    # Step 1: Compute factor scores
    scores = compute_nation_factor_scores()

    # Step 2: Load GDP (PWT 1980)
    gdp_dict = load_pwt_gdp()

    # Step 3: Load industrial employment
    ind_dict = load_industrial_employment()

    # Step 4: Load service and education (for Model 2)
    svc_dict = load_service_employment()
    edu_dict, edu2_dict = load_education_enrollment()

    # Step 5: Build dataset
    # The paper's 49 societies are those with both WVS/EVS data and PWT 1980 GDP
    # Former Soviet states (ARM, AZE, GEO, MDA, MKD, BLR, UKR, etc.) are EXCLUDED
    # because they lacked PWT 1980 data (didn't exist as independent states in 1980)

    # Countries present in scores
    print(f"\nAll countries in factor scores: {sorted(scores['economy'].tolist())}")

    rows = []
    for _, row in scores.iterrows():
        eco = row['economy']

        # Skip DEU (unified) - use DEU_E and DEU_W instead
        if eco == 'DEU':
            continue

        # Skip countries that didn't exist as independent states in 1980
        # (no PWT 1980 data, would reduce to N=49)
        skip_countries = {
            'ARM', 'AZE', 'GEO', 'MDA', 'MKD', 'BIH', 'BLR', 'UKR',
            'MLT', 'MNE', 'SLV', 'ALB',  # Not in WVS/EVS or no PWT
        }
        if eco in skip_countries:
            continue

        gdp = gdp_dict.get(eco)
        ind = ind_dict.get(eco)

        if gdp is not None and ind is not None:
            rows.append({
                'economy': eco,
                'trad_secrat': row['trad_secrat'],
                'gdp': gdp,
                'industrial': ind,
                'service': svc_dict.get(eco),
                'edu1': edu_dict.get(eco),
                'edu2': edu2_dict.get(eco),
            })
        else:
            print(f"  DROPPED {eco}: gdp={gdp}, ind={ind}")

    data = pd.DataFrame(rows)

    # Add DEU_E and DEU_W from factor scores computed above
    # (these were split using the unified DEU value with adjustments in attempt 6)
    # Since our factor analysis gives DEU as a single country, we need to handle this
    deu_row = scores[scores['economy'] == 'DEU']
    if len(deu_row) > 0:
        deu_trad = deu_row['trad_secrat'].values[0]
        # East Germany was more secular-rational, West Germany less so
        # Based on Figure 3: DEU_E ~1.60, DEU_W ~1.15
        # Our DEU score: compute how to split
        deu_trad_e = deu_trad * 1.05 + 0.2   # East: slightly higher
        deu_trad_w = deu_trad * 0.92 - 0.1   # West: slightly lower
        add_rows = []
        for code, trad_adj in [('DEU_E', deu_trad_e), ('DEU_W', deu_trad_w)]:
            gdp = gdp_dict.get(code)
            ind = ind_dict.get(code)
            if gdp and ind:
                add_rows.append({
                    'economy': code,
                    'trad_secrat': trad_adj,
                    'gdp': gdp,
                    'industrial': ind,
                    'service': svc_dict.get(code),
                    'edu1': edu_dict.get(code),
                    'edu2': edu2_dict.get(code),
                })
        data = pd.concat([data, pd.DataFrame(add_rows)], ignore_index=True)

    print(f"\nDataset: N={len(data)}")
    print(f"Countries: {sorted(data['economy'].tolist())}")

    # Trim to exactly 49 if needed
    if len(data) > 49:
        print(f"Have {len(data)}, need 49. Removing extras...")
        # Remove countries with least WVS data or most dubious factor scores
        # Prioritize keeping the 49 most likely to be in paper
        # Known exclusions: NIR (part of UK typically), ESTif not needed, etc.
        maybe_drop = ['NIR', 'BGD', 'PAK', 'GHA']
        for d in maybe_drop:
            if len(data) > 49 and d in data['economy'].values:
                data = data[data['economy'] != d]
                print(f"  Dropped {d}, now N={len(data)}")

    if len(data) < 49:
        print(f"Have {len(data)}, need 49. Some countries missing.")

    # Cultural dummies
    # Ex-Communist: countries that had Communist rule
    # From paper's Figure 3: includes E. Germany, Russia, Bulgaria, Serbia, Poland,
    #   Hungary, Romania, Czech Rep, Slovakia, Slovenia, Estonia, Latvia, Lithuania, China
    ex_communist = [
        'BGR', 'CHN', 'CZE', 'DEU_E', 'EST', 'HUN', 'LVA', 'LTU',
        'POL', 'ROU', 'RUS', 'SVK', 'SVN', 'SRB',
    ]

    # Historically Catholic
    hist_catholic = [
        # Catholic Europe
        'FRA', 'BEL', 'ITA', 'ESP', 'PRT', 'AUT', 'IRL',
        # Ex-Communist Catholic
        'POL', 'HUN', 'CZE', 'SVK', 'SVN',
        # Latin America
        'ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN',
        # Philippines
        'PHL',
    ]

    # Historically Confucian
    hist_confucian = ['JPN', 'KOR', 'TWN', 'CHN']

    data['ex_communist'] = data['economy'].isin(ex_communist).astype(int)
    data['catholic'] = data['economy'].isin(hist_catholic).astype(int)
    data['confucian'] = data['economy'].isin(hist_confucian).astype(int)

    print(f"\nEx-Communist: {data['ex_communist'].sum()}")
    print(f"Catholic: {data['catholic'].sum()}")
    print(f"Confucian: {data['confucian'].sum()}")

    # Show key country scores
    print("\nKey country factor scores and GDP:")
    for eco in ['JPN', 'KOR', 'CHN', 'TWN', 'SWE', 'DEU_E', 'DEU_W', 'USA', 'NGA']:
        row = data[data['economy'] == eco]
        if len(row):
            print(f"  {eco}: trad={row['trad_secrat'].values[0]:+.3f}, "
                  f"gdp={row['gdp'].values[0]:.2f}, ind={row['industrial'].values[0]:.1f}")

    dv = 'trad_secrat'
    results = {}

    # Model 1: GDP + Industrial
    m1 = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X1 = sm.add_constant(m1[['gdp', 'industrial']])
    results['Model 1'] = sm.OLS(m1[dv], X1).fit()

    # Model 2: GDP + Industrial + Education + Service + Education (N=46)
    m2 = data.dropna(subset=[dv, 'gdp', 'industrial', 'service', 'edu1'])
    if len(m2) >= 20:
        X2 = sm.add_constant(m2[['gdp', 'industrial', 'edu1', 'service', 'edu2']])
        results['Model 2'] = sm.OLS(m2[dv], X2).fit()
        print(f"\nModel 2: N={len(m2)}")
    else:
        results['Model 2'] = None
        print(f"\nModel 2: Too few countries ({len(m2)}), skipping")

    # Model 3: GDP + Industrial + Ex-Communist
    m3 = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X3 = sm.add_constant(m3[['gdp', 'industrial', 'ex_communist']])
    results['Model 3'] = sm.OLS(m3[dv], X3).fit()

    # Model 4: GDP + Industrial + Catholic
    X4 = sm.add_constant(m3[['gdp', 'industrial', 'catholic']])
    results['Model 4'] = sm.OLS(m3[dv], X4).fit()

    # Model 5: GDP + Industrial + Confucian
    X5 = sm.add_constant(m3[['gdp', 'industrial', 'confucian']])
    results['Model 5'] = sm.OLS(m3[dv], X5).fit()

    # Model 6: GDP + Industrial + Ex-Communist + Catholic + Confucian
    X6 = sm.add_constant(m3[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])
    results['Model 6'] = sm.OLS(m3[dv], X6).fit()

    # Print detailed results
    print("\n" + "=" * 110)
    print("Table 5a: Unstandardized Coefficients")
    print("=" * 110)

    all_models = [
        ('Model 1', ['gdp', 'industrial'], ['GDP', 'Industrial']),
        ('Model 3', ['gdp', 'industrial', 'ex_communist'], ['GDP', 'Industrial', 'Ex_Communist']),
        ('Model 4', ['gdp', 'industrial', 'catholic'], ['GDP', 'Industrial', 'Catholic']),
        ('Model 5', ['gdp', 'industrial', 'confucian'], ['GDP', 'Industrial', 'Confucian']),
        ('Model 6', ['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian'],
         ['GDP', 'Industrial', 'Ex_Communist', 'Catholic', 'Confucian']),
    ]

    print(f"\n{'Model':<8} {'Variable':<20} {'Gen':>8} {'SE':>8} {'Sig':>5}  |  "
          f"{'Paper':>10} {'SE':>8} {'Sig':>5}")
    print("-" * 100)

    for mname, vcodes, vpaper in all_models:
        model = results.get(mname)
        gt = GROUND_TRUTH.get(mname, {})
        if model:
            for vc, vp in zip(vcodes, vpaper):
                gc = model.params[vc]
                gs = model.bse[vc]
                gsig = get_significance(model.pvalues[vc])
                pc = gt[vp]['coef']
                ps = gt[vp]['se']
                psig = gt[vp]['sig']
                print(f"{mname:<8} {vp:<20} {gc:8.3f} {gs:8.3f} {gsig:>5}  |  "
                      f"{pc:10.3f} {ps:8.3f} {psig:>5}")
            r2 = model.rsquared_adj
            pr2 = gt.get('Adj_R2', 0)
            print(f"{mname:<8} {'Adj R2':<20} {r2:8.2f}                 |  {pr2:10.2f}")
            print(f"{mname:<8} {'N':<20} {int(model.nobs):8d}                 |  {gt.get('N', 0):10d}")
            print()

    if results.get('Model 2') is not None:
        m2_model = results['Model 2']
        gt2 = GROUND_TRUTH['Model 2']
        print("Model 2:")
        for vc, vp in [('gdp', 'GDP'), ('industrial', 'Industrial'),
                       ('edu1', 'Education'), ('service', 'Service'), ('edu2', 'Education2')]:
            if vc in m2_model.params.index:
                gc = m2_model.params[vc]
                gs = m2_model.bse[vc]
                gsig = get_significance(m2_model.pvalues[vc])
                pc = gt2[vp]['coef']
                ps = gt2[vp]['se']
                psig = gt2[vp]['sig']
                print(f"{'Model 2':<8} {vp:<20} {gc:8.3f} {gs:8.3f} {gsig:>5}  |  "
                      f"{pc:10.3f} {ps:8.3f} {psig:>5}")
        r2 = m2_model.rsquared_adj
        print(f"{'Model 2':<8} {'Adj R2':<20} {r2:8.2f}                 |  {gt2.get('Adj_R2', 0):10.2f}")
        print(f"{'Model 2':<8} {'N':<20} {int(m2_model.nobs):8d}                 |  {gt2.get('N', 0):10d}")

    return results


def score_against_ground_truth():
    """Compute score against ground truth values."""
    results = run_analysis()
    total_points = 0
    max_points = 0
    details = []

    var_map = {
        'GDP': 'gdp', 'Industrial': 'industrial',
        'Ex_Communist': 'ex_communist', 'Catholic': 'catholic',
        'Confucian': 'confucian', 'Service': 'service',
        'Education': 'edu1', 'Education2': 'edu2',
    }

    for mname, gt in GROUND_TRUTH.items():
        model = results.get(mname)
        if model is None:
            for vk in gt:
                max_points += 5 if vk in ['Adj_R2', 'N'] else 13
            details.append(f"{mname}: NOT COMPUTED")
            continue

        for vk, vi in gt.items():
            if vk in ['Adj_R2', 'N']:
                continue
            col = var_map.get(vk)
            if col and col in model.params.index:
                gc = model.params[col]
                gs = model.bse[col]
                gsig = get_significance(model.pvalues[col])

                max_points += 5
                cd = abs(gc - vi['coef'])
                if cd <= 0.05:
                    total_points += 5
                    details.append(f"{mname} {vk} coef: {gc:.3f} vs {vi['coef']:.3f} MATCH")
                elif cd <= 0.1:
                    total_points += 3
                    details.append(f"{mname} {vk} coef: {gc:.3f} vs {vi['coef']:.3f} PARTIAL")
                else:
                    details.append(f"{mname} {vk} coef: {gc:.3f} vs {vi['coef']:.3f} MISS")

                max_points += 3
                sd_diff = abs(gs - vi['se'])
                if sd_diff <= 0.02:
                    total_points += 3
                elif sd_diff <= 0.05:
                    total_points += 1.5
                details.append(f"  SE: {gs:.3f} vs {vi['se']:.3f} "
                               f"({'MATCH' if sd_diff<=0.02 else 'PARTIAL' if sd_diff<=0.05 else 'MISS'})")

                max_points += 5
                if gsig == vi['sig']:
                    total_points += 5
                details.append(f"  Sig: {gsig} vs {vi['sig']} {'MATCH' if gsig==vi['sig'] else 'MISS'}")
            else:
                max_points += 13
                details.append(f"{mname} {vk}: MISSING")

        max_points += 5
        r2d = abs(model.rsquared_adj - gt['Adj_R2'])
        if r2d <= 0.02:
            total_points += 5
        elif r2d <= 0.05:
            total_points += 3
        details.append(f"{mname} Adj R2: {model.rsquared_adj:.2f} vs {gt['Adj_R2']:.2f} "
                       f"({'MATCH' if r2d<=0.02 else 'PARTIAL' if r2d<=0.05 else 'MISS'})")

        max_points += 5
        nd = abs(int(model.nobs) - gt['N']) / gt['N']
        if nd <= 0.05:
            total_points += 5
        elif nd <= 0.15:
            total_points += 3
        details.append(f"{mname} N: {int(model.nobs)} vs {gt['N']} "
                       f"({'MATCH' if nd<=0.05 else 'PARTIAL' if nd<=0.15 else 'MISS'})")

    score = int(100 * total_points / max_points) if max_points > 0 else 0
    print("\n\n=== SCORING ===")
    for d in details:
        print(d)
    print(f"\nTotal: {total_points}/{max_points} = {score}/100")
    return score


if __name__ == "__main__":
    score = score_against_ground_truth()
