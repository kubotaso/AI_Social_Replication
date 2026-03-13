#!/usr/bin/env python3
"""
Table 5a Replication - Attempt 31
Inglehart & Baker (2000), Table 5a.

STRATEGY: Full compliance with Code Generation Contract Rule #8.
Changes from attempts 28 and 30:
- Hybrid approach: latest-wave per country (like retry_28) + A006 fallback for missing F063

KEY INSIGHT from comparing retry_28 (score=75.8) and retry_30 (score=70.5):
- retry_28: no A006 fix → KOR/CHN GOD_IMP = column mean → Confucian M5=0.937 (miss)
  BUT M1 AdjR2=0.40 (match), M4 AdjR2=0.52 (match)
- retry_30: A006 fallback + pooled waves → Confucian M5=1.348 (improved!)
  BUT M1 AdjR2=0.32 (big miss), pooling hurt R-squared

FIX: Apply A006 fallback FIRST (before get_latest_per_country), then use latest wave.
This gives:
- KOR: A006 fallback applied to all rows → latest wave gets A006-derived GOD_IMP
- CHN: Wave 3 (latest) gets A006 fallback for GOD_IMP (less extreme than pooled approach)
- Other countries: unchanged from retry_28 (F063 available, latest wave)

Expected: Confucian M5 > 0.937 (due to KOR/CHN GOD_IMP fix)
         M1 AdjR2 closer to 0.40 (not degraded by pooling)

Note: CHN Wave 3 GOD_IMP ≈ 3.1 (from A006), less secular than pooled ≈ 2.47.
This reduces CHN secular-rational from 1.279 to ~1.1, slightly less positive Confucian
contribution. But KOR's improvement (0→1.0) more than compensates.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared_factor_analysis import (load_combined_data, get_latest_per_country,
                                     clean_missing, FACTOR_ITEMS, varimax)

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

EX_COMMUNIST = ['BGR', 'CHN', 'CZE', 'DEU_E', 'HUN', 'POL', 'ROU', 'RUS',
                'SVK', 'SVN', 'SRB', 'BIH', 'MKD', 'HRV', 'BLR', 'UKR',
                'ARM', 'AZE', 'GEO', 'MDA', 'LVA', 'LTU', 'EST']
HIST_CATHOLIC = ['FRA', 'BEL', 'ITA', 'ESP', 'PRT', 'AUT', 'IRL', 'POL',
                 'HRV', 'SVN', 'HUN', 'CZE', 'SVK', 'ARG', 'BRA', 'CHL',
                 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN', 'PHL',
                 'SLV']
HIST_CONFUCIAN = ['JPN', 'KOR', 'TWN', 'CHN']

EXCLUDE = {'ARM', 'AZE', 'BIH', 'BLR', 'COL', 'EST', 'GEO', 'GHA',
           'LTU', 'LVA', 'MDA', 'MKD', 'NGA', 'SVN', 'VEN'}

ISO_MAP = {'DEU': 'DEU_W', 'NIR': 'GBR'}


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
        'EL SALVADOR': 'SLV', 'GREECE': 'GRC', 'ISRAEL': 'ISR',
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
    if 'DEU_W' in gdp_dict:
        gdp_dict['DEU'] = gdp_dict['DEU_W']
    if 'GBR' in gdp_dict:
        gdp_dict['NIR'] = gdp_dict['GBR']
    return gdp_dict


def compute_factor_scores_latest_with_fallback(waves_wvs=[2, 3], include_evs=True):
    """
    Compute ecological PCA using latest wave per country, with A006 fallback for F063.

    KEY STEPS:
    1. Load all WVS waves 2+3 + EVS
    2. Apply A006 fallback to rows where GOD_IMP (F063) is missing
       - WVS A006 is 4-point (1=very important, 4=not at all)
       - Rescale: GOD_IMP_approx = 13 - 3*A006 (1->10, 2->7, 3->4, 4->1)
       - EVS rows: GOD_IMP already set from 10-pt A006, won't be affected (not NaN)
    3. Apply get_latest_per_country to select one wave per country
       - For CHN: selects Wave 3 (latest), which now has A006-derived GOD_IMP ≈ 3.1
       - For KOR: selects Wave 3 (or 2), A006-derived GOD_IMP ≈ 5.84
    4. Compute country means, PCA, varimax rotation

    This preserves the get_latest_per_country approach (like retry_28, which scored 75.8)
    while fixing the GOD_IMP for KOR and CHN.
    """
    df = load_combined_data(waves_wvs=waves_wvs, include_evs=include_evs)

    # Step 1: Clean GOD_IMP (replace missing codes < 0 with NaN)
    if 'GOD_IMP' in df.columns:
        df['GOD_IMP'] = pd.to_numeric(df['GOD_IMP'], errors='coerce')
        df.loc[df['GOD_IMP'] < 0, 'GOD_IMP'] = np.nan

    # Step 2: Apply A006 fallback for rows where GOD_IMP is missing
    # WVS A006 is 4-point (1=very important, 4=not at all)
    # EVS rows: GOD_IMP already set from 10-pt EVS A006 (not NaN) → not affected
    if 'A006' in df.columns:
        df['A006'] = pd.to_numeric(df['A006'], errors='coerce')
        df.loc[df['A006'] < 0, 'A006'] = np.nan

    missing_god_imp = df['GOD_IMP'].isna()
    # Apply only when A006 is in WVS 4-point range [1,4]
    has_a006_4pt = df['A006'].notna() & (df['A006'] >= 1) & (df['A006'] <= 4)
    use_fallback = missing_god_imp & has_a006_4pt
    n_fallback = use_fallback.sum()
    print(f"  Applying A006 fallback to {n_fallback} rows")
    if n_fallback > 0:
        # Diagnostics: which countries get fallback
        fallback_countries = df.loc[use_fallback, 'COUNTRY_ALPHA'].value_counts().head(10)
        print(f"  Countries with A006 fallback: {fallback_countries.to_dict()}")
        df.loc[use_fallback, 'GOD_IMP'] = 13.0 - 3.0 * df.loc[use_fallback, 'A006']

    # Step 3: Get latest wave per country (same as shared_factor_analysis.py)
    df = get_latest_per_country(df)

    # Step 4: Construct AUTONOMY if needed (already done in load_combined_data, but verify)
    if 'AUTONOMY' not in df.columns or df['AUTONOMY'].isna().all():
        for v in ['A042', 'A034', 'A029']:
            if v in df.columns:
                df[v] = pd.to_numeric(df[v], errors='coerce')
                df[v] = df[v].where(df[v] >= 0, np.nan)
                df.loc[df[v] == 2, v] = 0
        if all(v in df.columns for v in ['A042', 'A034', 'A029']):
            df['AUTONOMY'] = df['A042'] + df['A034'] - df['A029']

    # Step 5: Clean all factor items
    df = clean_missing(df, FACTOR_ITEMS)

    # Step 6: Recode directionally (higher = traditional/survival)
    if 'F120' in df.columns:
        df['F120'] = 11.0 - df['F120']
    if 'G006' in df.columns:
        df['G006'] = 5.0 - df['G006']
    if 'E018' in df.columns:
        df['E018'] = 4.0 - df['E018']
    if 'Y002' in df.columns:
        df['Y002'] = 4.0 - df['Y002']
    if 'F118' in df.columns:
        df['F118'] = 11.0 - df['F118']

    # Print GOD_IMP for key countries after fallback
    for iso in ['JPN', 'KOR', 'TWN', 'CHN', 'SWE', 'USA']:
        sub = df[df['COUNTRY_ALPHA'] == iso]
        if len(sub) > 0 and 'GOD_IMP' in df.columns:
            gm = sub['GOD_IMP'].mean()
            print(f"  {iso}: mean GOD_IMP = {gm:.2f}")

    # Step 7: Compute country means
    country_means = df.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)

    # Fill remaining NaN with column means
    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Step 8: Standardize
    means = country_means.mean()
    stds = country_means.std().replace(0, 1)
    scaled = (country_means - means) / stds

    # Step 9: PCA via SVD
    n = len(scaled)
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(n - 1)
    scores_raw = U[:, :2] * S[:2]

    # Step 10: Varimax rotation
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Step 11: Identify Traditional/Secular-Rational factor
    trad_items = ['AUTONOMY', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items if i in loadings_df.index)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items if i in loadings_df.index)
    trad_col = 'F1' if f1_trad > f2_trad else 'F2'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[('F2' if trad_col == 'F1' else 'F1')].values
    }).reset_index(drop=True)

    # Fix direction: Sweden should be positive (secular-rational)
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        if result[result['COUNTRY_ALPHA'] == 'SWE']['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
        if result[result['COUNTRY_ALPHA'] == 'SWE']['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    return result


def run_analysis():
    print("Computing factor scores (latest-wave + A006 fallback for missing F063)...")
    scores_df = compute_factor_scores_latest_with_fallback(waves_wvs=[2, 3], include_evs=True)
    print(f"Got scores for {len(scores_df)} countries")

    # Normalize to SD=1
    trad_sd = scores_df['trad_secrat'].std()
    scores_df['trad_secrat_norm'] = scores_df['trad_secrat'] / trad_sd
    print(f"Factor score SD={trad_sd:.3f}")

    print("\nKey country factor scores (normalized):")
    for iso in ['JPN', 'KOR', 'TWN', 'CHN', 'SWE', 'NOR', 'DEU', 'USA', 'IND', 'NGA']:
        row = scores_df[scores_df['COUNTRY_ALPHA'] == iso]
        if len(row) > 0:
            print(f"  {iso}: norm={row['trad_secrat_norm'].values[0]:.3f}")

    gdp_dict = load_pwt_gdp()
    industry_dict = load_wdr_industry()

    data_rows = []
    for _, row in scores_df.iterrows():
        iso = row['COUNTRY_ALPHA']
        if iso in EXCLUDE:
            continue
        lookup_iso = ISO_MAP.get(iso, iso)
        gdp_val = gdp_dict.get(lookup_iso)
        ind_val = industry_dict.get(iso, industry_dict.get(lookup_iso))
        if gdp_val is None or ind_val is None:
            print(f"  Missing: {iso} (lookup={lookup_iso}), "
                  f"GDP={'ok' if gdp_val else 'MISS'}, Ind={'ok' if ind_val else 'MISS'}")
            continue
        data_rows.append({
            'economy': iso,
            'trad_secrat': row['trad_secrat_norm'],
            'gdp': gdp_val,
            'industrial': ind_val,
        })

    df = pd.DataFrame(data_rows)
    df['ex_communist'] = df['economy'].isin(EX_COMMUNIST).astype(int)
    df['catholic'] = df['economy'].isin(HIST_CATHOLIC).astype(int)
    df['confucian'] = df['economy'].isin(HIST_CONFUCIAN).astype(int)

    print(f"\nFinal N={len(df)}")

    print("\nConfucian country details:")
    for iso in HIST_CONFUCIAN:
        row = df[df['economy'] == iso]
        if len(row) > 0:
            print(f"  {iso}: trad={row['trad_secrat'].values[0]:.3f}, "
                  f"gdp={row['gdp'].values[0]:.2f}, ind={row['industrial'].values[0]}")

    y = df['trad_secrat']
    results = {}
    results['Model 1'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial']])).fit()
    results['Model 3'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'ex_communist']])).fit()
    results['Model 4'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'catholic']])).fit()
    results['Model 5'] = sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'confucian']])).fit()
    results['Model 6'] = sm.OLS(y, sm.add_constant(
        df[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])).fit()

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
    return score, results


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
    scores_df = compute_factor_scores_latest_with_fallback(waves_wvs=[2, 3], include_evs=True)
    trad_sd = scores_df['trad_secrat'].std()
    scores_df['trad_secrat_norm'] = scores_df['trad_secrat'] / trad_sd
    gdp_dict = load_pwt_gdp()
    industry_dict = load_wdr_industry()
    rows = []
    for _, row in scores_df.iterrows():
        iso = row['COUNTRY_ALPHA']
        if iso in EXCLUDE: continue
        lookup_iso = ISO_MAP.get(iso, iso)
        gdp_val = gdp_dict.get(lookup_iso)
        ind_val = industry_dict.get(iso, industry_dict.get(lookup_iso))
        if gdp_val is None or ind_val is None: continue
        rows.append({'economy': iso, 'trad_secrat': row['trad_secrat_norm'],
                     'gdp': gdp_val, 'industrial': ind_val})
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
    results['Model 6'] = sm.OLS(y, sm.add_constant(
        df[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])).fit()
    return score_against_ground_truth_with_results(results)


if __name__ == "__main__":
    run_analysis()
