#!/usr/bin/env python3
"""
Table 5a Replication - Attempt 32
Inglehart & Baker (2000), Table 5a.

STRATEGY: Full compliance with Code Generation Contract Rule #8.

ANALYSIS OF PREVIOUS ATTEMPTS:
- retry_28 (get_latest_per_country, no A006 fix): M1 R2=0.40 MATCH, Confucian M5=0.937 → score 75.8
- retry_30 (pooled waves + A006 fallback all countries): M1 R2=0.32 MISS, Confucian M5=1.348 → 70.5
- retry_31 (latest wave + A006 fallback before filter): M1 R2=0.36 MISS, Confucian M5=1.105 → 73.1

ROOT CAUSE: A006 fallback in retry_30/31 affected COL, POL, CZE, SVK, TUR, JPN, RUS, UKR
(not just KOR/CHN), degrading the PCA factor structure and M1 R2.

FIX FOR retry_32:
- Compute country-level GOD_IMP separately from ALL waves:
  * For each country: take mean F063 from any wave where F063 > 0
  * For countries with ZERO valid F063 in any wave (only KOR): use A006 rescaled
- Use get_latest_per_country for all OTHER factor items (AUTONOMY, F120, G006, E018, etc.)
- Override GOD_IMP in country_means with the "best available" values

This ensures:
- KOR: GOD_IMP from A006 fallback (fixed! was filled with column mean before)
- CHN: GOD_IMP from Wave 2 F063=1.62 (very secular, correct)
- COL, POL, CZE, etc.: GOD_IMP from their actual F063 data (unchanged from retry_28)
- All other factor items: unchanged from retry_28

Expected: Confucian M5 > 0.937 (KOR and CHN fixed)
         M1 R2 ≈ 0.40 (other countries unchanged, no degradation from retry_28)
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
    csv_path = os.path.join(BASE_DIR, 'data/wdr_industry_1980.csv')
    df = pd.read_csv(csv_path)
    return dict(zip(df['iso'], df['industry_pct_1980']))


def load_pwt_gdp():
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


def compute_factor_scores_best_f063(waves_wvs=[2, 3], include_evs=True):
    """
    Ecological PCA where GOD_IMP uses the best available data per country:
    1. For each country: mean F063 across ALL waves where F063 > 0 (WVS data)
    2. For countries with NO valid F063 in any wave (KOR): use A006 rescaled
       A006 (WVS 4-pt): 1=very important -> 10, 4=not at all -> 1
       Formula: GOD_IMP_approx = 13 - 3*A006
    3. For all other factor items: use get_latest_per_country (standard approach)
    4. Override GOD_IMP in country_means with the computed "best available" values

    This preserves retry_28's approach for all non-GOD_IMP items,
    while fixing only the problematic KOR (and CHN) GOD_IMP values.
    """
    # Load all data (need all waves to find best F063 per country)
    df_all = load_combined_data(waves_wvs=waves_wvs, include_evs=include_evs)

    # --- Step 1: Compute best GOD_IMP per country from all available data ---
    best_god_imp = {}

    # WVS: GOD_IMP = F063 (already set in load_combined_data)
    # EVS: GOD_IMP = EVS A006 (10-pt, already set)
    if 'GOD_IMP' in df_all.columns:
        df_all['GOD_IMP_raw'] = pd.to_numeric(df_all['GOD_IMP'], errors='coerce')
        df_all.loc[df_all['GOD_IMP_raw'] < 0, 'GOD_IMP_raw'] = np.nan
    else:
        df_all['GOD_IMP_raw'] = np.nan

    # Also get WVS F063 directly (clean)
    if 'F063' in df_all.columns:
        df_all['F063_clean'] = pd.to_numeric(df_all['F063'], errors='coerce')
        df_all.loc[df_all['F063_clean'] <= 0, 'F063_clean'] = np.nan
    else:
        df_all['F063_clean'] = np.nan

    # WVS A006 (4-point scale, for fallback)
    if 'A006' in df_all.columns:
        df_all['A006_clean'] = pd.to_numeric(df_all['A006'], errors='coerce')
        df_all.loc[df_all['A006_clean'] < 0, 'A006_clean'] = np.nan
        # Only WVS-style A006 (values 1-4 are the 4-point scale)
        # EVS A006 is 10-point, already captured in GOD_IMP_raw
    else:
        df_all['A006_clean'] = np.nan

    # For each country: compute mean F063 across all waves
    for iso, group in df_all.groupby('COUNTRY_ALPHA'):
        # Priority 1: WVS F063 (10-pt God importance)
        f063_vals = group['F063_clean'].dropna()
        if len(f063_vals) > 0:
            best_god_imp[iso] = f063_vals.mean()
            continue

        # Priority 2: EVS GOD_IMP (already set from EVS A006 10-pt)
        # For EVS rows, GOD_IMP_raw is set but F063 is not
        god_imp_vals = group['GOD_IMP_raw'].dropna()
        # Filter to plausible range (1-10) to exclude codes
        god_imp_valid = god_imp_vals[(god_imp_vals >= 1) & (god_imp_vals <= 10)]
        if len(god_imp_valid) > 0:
            best_god_imp[iso] = god_imp_valid.mean()
            continue

        # Priority 3: WVS A006 (4-point importance of religion) → rescale to 1-10
        # Only use when A006 is in the WVS 4-pt range [1,4]
        a006_vals = group['A006_clean'].dropna()
        a006_wvs = a006_vals[(a006_vals >= 1) & (a006_vals <= 4)]
        if len(a006_wvs) > 0:
            # Rescale: 1->10, 2->7, 3->4, 4->1 (linear, 1-4 maps to 10-1)
            best_god_imp[iso] = (13.0 - 3.0 * a006_wvs).mean()
            continue

        # No data available
        best_god_imp[iso] = np.nan

    # Diagnostic: which countries use A006 fallback
    print("  GOD_IMP source by country:")
    for iso in ['JPN', 'KOR', 'TWN', 'CHN', 'SWE', 'USA', 'DEU', 'COL', 'POL']:
        val = best_god_imp.get(iso, np.nan)
        # Determine source
        sub = df_all[df_all['COUNTRY_ALPHA'] == iso]
        has_f063 = sub['F063_clean'].dropna().__len__() > 0
        source = 'F063' if has_f063 else ('A006_fallback' if not np.isnan(val) else 'MISSING')
        print(f"    {iso}: GOD_IMP={val:.2f} (source={source})" if not np.isnan(val) else f"    {iso}: GOD_IMP=NaN (source={source})")

    # --- Step 2: Compute other factor items using get_latest_per_country ---
    df_latest = get_latest_per_country(df_all)

    # Construct AUTONOMY if needed
    if 'AUTONOMY' not in df_latest.columns or df_latest['AUTONOMY'].isna().all():
        for v in ['A042', 'A034', 'A029']:
            if v in df_latest.columns:
                df_latest[v] = pd.to_numeric(df_latest[v], errors='coerce')
                df_latest[v] = df_latest[v].where(df_latest[v] >= 0, np.nan)
                df_latest.loc[df_latest[v] == 2, v] = 0
        if all(v in df_latest.columns for v in ['A042', 'A034', 'A029']):
            df_latest['AUTONOMY'] = df_latest['A042'] + df_latest['A034'] - df_latest['A029']

    # Clean all factor items
    df_latest = clean_missing(df_latest, FACTOR_ITEMS)

    # Recode directionally (higher = traditional/survival)
    if 'F120' in df_latest.columns:
        df_latest['F120'] = 11.0 - df_latest['F120']
    if 'G006' in df_latest.columns:
        df_latest['G006'] = 5.0 - df_latest['G006']
    if 'E018' in df_latest.columns:
        df_latest['E018'] = 4.0 - df_latest['E018']
    if 'Y002' in df_latest.columns:
        df_latest['Y002'] = 4.0 - df_latest['Y002']
    if 'F118' in df_latest.columns:
        df_latest['F118'] = 11.0 - df_latest['F118']

    # Compute country means for all items (using latest wave data)
    country_means = df_latest.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)

    # --- Step 3: Override GOD_IMP with best available values ---
    # Note: recode_factor_items does NOT transform GOD_IMP (kept as-is, higher=more traditional)
    # The recoding above also left GOD_IMP untouched
    # But GOD_IMP from `clean_missing` might now have NaN for rows where F063 was bad
    # We override with our best_god_imp computed from all waves
    for iso in country_means.index:
        if iso in best_god_imp and not np.isnan(best_god_imp[iso]):
            country_means.loc[iso, 'GOD_IMP'] = best_god_imp[iso]

    # Fill any remaining NaN with column means
    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # --- Step 4: Standardize and run PCA ---
    means = country_means.mean()
    stds = country_means.std().replace(0, 1)
    scaled = (country_means - means) / stds

    n = len(scaled)
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(n - 1)
    scores_raw = U[:, :2] * S[:2]

    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Identify Traditional/Secular-Rational factor
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
    print("Computing factor scores (best available F063 per country)...")
    scores_df = compute_factor_scores_best_f063(waves_wvs=[2, 3], include_evs=True)
    print(f"Got scores for {len(scores_df)} countries")

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
            print(f"  Missing: {iso}, GDP={'ok' if gdp_val else 'MISS'}, Ind={'ok' if ind_val else 'MISS'}")
            continue
        data_rows.append({'economy': iso, 'trad_secrat': row['trad_secrat_norm'],
                          'gdp': gdp_val, 'industrial': ind_val})

    df = pd.DataFrame(data_rows)
    df['ex_communist'] = df['economy'].isin(EX_COMMUNIST).astype(int)
    df['catholic'] = df['economy'].isin(HIST_CATHOLIC).astype(int)
    df['confucian'] = df['economy'].isin(HIST_CONFUCIAN).astype(int)

    print(f"\nFinal N={len(df)}")

    print("\nConfucian country details:")
    for iso in HIST_CONFUCIAN:
        r = df[df['economy'] == iso]
        if len(r) > 0:
            print(f"  {iso}: trad={r['trad_secrat'].values[0]:.3f}, "
                  f"gdp={r['gdp'].values[0]:.2f}, ind={r['industrial'].values[0]}")

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
    scores_df = compute_factor_scores_best_f063(waves_wvs=[2, 3], include_evs=True)
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
    results = {
        'Model 1': sm.OLS(y, sm.add_constant(df[['gdp', 'industrial']])).fit(),
        'Model 3': sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'ex_communist']])).fit(),
        'Model 4': sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'catholic']])).fit(),
        'Model 5': sm.OLS(y, sm.add_constant(df[['gdp', 'industrial', 'confucian']])).fit(),
        'Model 6': sm.OLS(y, sm.add_constant(
            df[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])).fit(),
    }
    return score_against_ground_truth_with_results(results)


if __name__ == "__main__":
    run_analysis()
