#!/usr/bin/env python3
"""
Shared factor analysis computation for Inglehart & Baker (2000) replication.
Computes the two key dimensions:
1. Traditional vs. Secular-Rational Values
2. Survival vs. Self-Expression Values

Uses combined WVS + EVS 1990 data.
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")
WB_PATH = os.path.join(BASE_DIR, "data/world_bank_indicators.csv")

FACTOR_ITEMS = ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018',  # Trad/Sec-Rat
                'Y002', 'A008', 'E025', 'F118', 'A165']   # Surv/Self-Exp

# Legacy mapping for scripts that still use old names
FACTOR_ITEMS_LEGACY = ['A006', 'A042', 'F120', 'G006', 'E018',
                       'Y002', 'A008', 'E025', 'F118', 'A165']


def load_combined_data(waves_wvs=[2, 3], include_evs=True):
    """Load WVS + EVS combined data."""
    import csv
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024',
              'A001', 'A002', 'A003', 'A004', 'A005', 'A006', 'A008', 'A009',
              'A025', 'A029', 'A030', 'A032', 'A034', 'A035', 'A038', 'A039',
              'A040', 'A041', 'A042', 'A165', 'C001', 'C006',
              'D017', 'D018', 'D022', 'D054', 'D058', 'D059', 'D060',
              'E012', 'E014', 'E018', 'E023', 'E025', 'E026', 'E033', 'E036', 'E037',
              'E114', 'E117',
              'F001', 'F024', 'F028', 'F034', 'F050', 'F051', 'F063',
              'F114', 'F118', 'F119', 'F120', 'F121', 'F122', 'F125',
              'G006', 'G007', 'S017', 'S018',
              'X001', 'X002', 'X003', 'Y002']
    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    if waves_wvs:
        wvs = wvs[wvs['S002VS'].isin(waves_wvs)]

    # Create GOD_IMP from WVS F063 (10-point God importance scale)
    # In WVS Time Series v5.0, F063 is "How important is God in your life" (1-10)
    # NOT church attendance. A006 in WVS is "Importance of religion" (1-4 scale).
    if 'F063' in wvs.columns:
        wvs['GOD_IMP'] = wvs['F063']
    else:
        wvs['GOD_IMP'] = np.nan

    # Create AUTONOMY index from child quality items
    # AUTONOMY = (obedience + religious_faith) - (independence + determination)
    # Higher = more traditional (emphasis on obedience/faith over autonomy)
    child_vars_wvs = ['A042', 'A034', 'A029', 'A032']
    for v in child_vars_wvs:
        if v in wvs.columns:
            wvs[v] = pd.to_numeric(wvs[v], errors='coerce')
            wvs[v] = wvs[v].where(wvs[v] >= 0, np.nan)

    # WVS child qualities: some waves use 0/1, others 1/2. Harmonize to 0/1.
    for v in child_vars_wvs:
        if v in wvs.columns:
            # Map 2→0 (not mentioned) for waves that use 1/2 coding
            wvs.loc[wvs[v] == 2, v] = 0

    # AUTONOMY = (obedience + religious_faith) - independence
    # Using 3-item version for consistency (A032/determination not in EVS)
    if all(v in wvs.columns for v in ['A042', 'A034', 'A029']):
        wvs['AUTONOMY'] = (wvs['A042'] + wvs['A034'] - wvs['A029'])
    else:
        wvs['AUTONOMY'] = np.nan

    if include_evs and os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)

        # EVS A006 is God importance 10-point scale (from q365)
        if 'A006' in evs.columns:
            evs['GOD_IMP'] = evs['A006']

        # EVS child qualities: A042 is 1=mentioned, 2=not mentioned; A029/A034 are 0/1
        # A032 (determination) not available in EVS
        for v in ['A042', 'A034', 'A029']:
            if v in evs.columns:
                evs[v] = pd.to_numeric(evs[v], errors='coerce')
                evs[v] = evs[v].where(evs[v] >= 0, np.nan)

        if 'A042' in evs.columns:
            # Recode EVS A042: 1=mentioned→1, 2=not mentioned→0
            evs['A042_bin'] = evs['A042'].map({1: 1, 2: 0})
        else:
            evs['A042_bin'] = np.nan

        if all(v in evs.columns for v in ['A042', 'A034', 'A029']):
            evs['AUTONOMY'] = (evs['A042_bin'] + evs['A034'] - evs['A029'])
        else:
            evs['AUTONOMY'] = np.nan

        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs

    return combined


def clean_missing(df, cols):
    """Replace WVS/EVS missing value codes (<0) with NaN."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col] >= 0, np.nan)
    return df


def get_latest_per_country(df):
    """For each country, keep only the latest available wave/year."""
    if 'S020' in df.columns:
        latest = df.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_year']
        df = df.merge(latest, on='COUNTRY_ALPHA')
        df = df[df['S020'] == df['latest_year']].drop('latest_year', axis=1)
    elif 'S002VS' in df.columns:
        latest = df.groupby('COUNTRY_ALPHA')['S002VS'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_wave']
        df = df.merge(latest, on='COUNTRY_ALPHA')
        df = df[df['S002VS'] == df['latest_wave']].drop('latest_wave', axis=1)
    return df


def varimax(Phi, gamma=1.0, q=100, tol=1e-8):
    """Varimax rotation."""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        Lambda = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.sum(Lambda**2, axis=0)))
        )
        R = u @ vt
        d_new = np.sum(s)
        if d_new - d < tol:
            break
        d = d_new
    return Phi @ R, R


def recode_factor_items(df):
    """Recode the 10 factor items so higher = traditional/survival."""
    df = df.copy()
    items = FACTOR_ITEMS
    df = clean_missing(df, items)

    # GOD_IMP: God importance 1-10 (1=not important, 10=very important).
    # Higher = more important = traditional. Keep as-is.

    # AUTONOMY: Already constructed as (obedience + religious_faith) - (independence + determination).
    # Higher = more traditional. Keep as-is.

    # F120: Abortion 1-10 (1=never, 10=always). Reverse: higher=never=traditional
    if 'F120' in df.columns:
        df['F120'] = 11 - df['F120']

    # G006: National pride 1=Very proud...4=Not at all. Reverse: higher=proud=traditional
    if 'G006' in df.columns:
        df['G006'] = 5 - df['G006']

    # E018: Respect authority 1=Good, 2=Don't mind, 3=Bad. Reverse: higher=good=traditional
    if 'E018' in df.columns:
        df['E018'] = 4 - df['E018']

    # Y002: Post-materialist 1=Materialist, 2=Mixed, 3=Postmaterialist. Reverse: higher=materialist=survival
    if 'Y002' in df.columns:
        df['Y002'] = 4 - df['Y002']

    # A008: Happiness 1=Very happy...4=Not at all. Higher = unhappy = survival. Keep.
    # E025: Petition 1=Have done, 2=Might do, 3=Would never. Higher = never = survival. Keep.

    # F118: Homosexuality 1-10 (1=never, 10=always). Reverse: higher=never=survival
    if 'F118' in df.columns:
        df['F118'] = 11 - df['F118']

    # A165: Trust 1=Most can be trusted, 2=Careful. Higher = careful = survival. Keep.

    return df


def compute_nation_level_factor_scores(waves_wvs=[2, 3], include_evs=True):
    """
    Compute two-dimensional factor scores at the nation level.
    Returns: (scores_df, loadings_df, country_means_df)
    """
    df = load_combined_data(waves_wvs=waves_wvs, include_evs=include_evs)
    df = get_latest_per_country(df)
    df = recode_factor_items(df)

    # Compute country means
    country_means = df.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)  # Require at least 7 of 10 items

    # Fill remaining NaN with column means for PCA
    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    # PCA via SVD
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)

    # Take first 2 components
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Determine which factor is which by checking total loading patterns
    # Traditional items: GOD_IMP, AUTONOMY, F120, G006, E018
    # Survival items: Y002, A008, E025, F118, A165
    trad_items = ['AUTONOMY', 'F120', 'G006', 'E018']  # Items that should load high on Trad
    surv_items = ['Y002', 'A008', 'E025', 'F118']  # Items that should load high on Surv

    f1_trad_load = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items if i in loadings_df.index)
    f2_trad_load = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items if i in loadings_df.index)

    if f1_trad_load > f2_trad_load:
        trad_col, surv_col = 'F1', 'F2'
    else:
        trad_col, surv_col = 'F2', 'F1'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

    loadings_result = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings_df[trad_col].values,
        'surv_selfexp': loadings_df[surv_col].values
    })

    # Fix direction: Paper convention is secular-rational = positive, self-expression = positive.
    # Sweden should be high positive on both dimensions.
    # Our recoding made higher = traditional/survival, so PCA gives traditional/survival = positive.
    # We need to FLIP so secular-rational/self-expression = positive.
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
            loadings_result['trad_secrat'] = -loadings_result['trad_secrat']

        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']
            loadings_result['surv_selfexp'] = -loadings_result['surv_selfexp']

    return result, loadings_result, country_means


def compute_individual_factor_scores(df, nation_loadings):
    """Compute individual-level factor scores using nation-level loadings."""
    df = recode_factor_items(df)

    items = FACTOR_ITEMS
    data = df[items].copy()

    # Standardize using the same scale
    for col in items:
        col_mean = data[col].mean()
        col_std = data[col].std()
        if col_std > 0:
            data[col] = (data[col] - col_mean) / col_std

    # Fill NaN with 0 (mean after standardization)
    data = data.fillna(0)

    # Compute scores as weighted sum using loadings
    trad_loadings = nation_loadings.set_index('item')['trad_secrat']
    surv_loadings = nation_loadings.set_index('item')['surv_selfexp']

    trad_scores = data[items].values @ trad_loadings[items].values
    surv_scores = data[items].values @ surv_loadings[items].values

    return trad_scores, surv_scores


def load_world_bank_data():
    """Load World Bank indicators and reshape for easy use."""
    wb = pd.read_csv(WB_PATH)
    return wb


def get_wb_indicator(wb, indicator, year, countries=None):
    """Get a specific World Bank indicator for a year, as a dict of country->value."""
    sub = wb[wb['indicator'] == indicator].copy()
    # Find the column for the given year
    yr_col = f'YR{year}'
    if yr_col not in sub.columns:
        # Try nearby years
        for delta in range(0, 6):
            for y in [year + delta, year - delta]:
                c = f'YR{y}'
                if c in sub.columns:
                    yr_col = c
                    break
            if yr_col != f'YR{year}':
                break

    if yr_col in sub.columns:
        result = sub.set_index('economy')[yr_col].to_dict()
        return {k: v for k, v in result.items() if pd.notna(v)}
    return {}


def get_cultural_zones():
    """Return cultural zone assignments based on Huntington (1993, 1996)."""
    return {
        'Ex-Communist': ['ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'CHN', 'HRV', 'CZE',
                         'EST', 'GEO', 'HUN', 'LVA', 'LTU', 'MKD', 'MDA', 'POL',
                         'ROU', 'RUS', 'SVK', 'SVN', 'UKR', 'SRB'],
        'Protestant Europe': ['DNK', 'FIN', 'ISL', 'NLD', 'NOR', 'SWE', 'CHE'],
        'English-speaking': ['AUS', 'CAN', 'GBR', 'NIR', 'IRL', 'NZL', 'USA'],
        'Latin America': ['ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN'],
        'Africa': ['GHA', 'NGA', 'ZAF'],
        'South Asia': ['BGD', 'IND', 'PAK', 'PHL', 'TUR'],
        'Orthodox': ['ARM', 'AZE', 'BLR', 'BGR', 'GEO', 'MDA', 'ROU', 'RUS', 'UKR', 'SRB', 'MKD'],
        'Confucian': ['CHN', 'JPN', 'KOR', 'TWN'],
        'Catholic Europe': ['FRA', 'BEL', 'ITA', 'ESP', 'PRT', 'AUT']
    }


def get_society_type_groups():
    """Return society type groupings for Figures 7-8."""
    return {
        'Ex-Communist': ['ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'HRV', 'CZE', 'EST',
                         'GEO', 'HUN', 'LVA', 'LTU', 'MKD', 'MDA', 'POL', 'ROU',
                         'RUS', 'SVK', 'SVN', 'UKR', 'SRB'],
        'Advanced industrial': ['AUS', 'AUT', 'BEL', 'CAN', 'DNK', 'FIN', 'FRA', 'DEU',
                                'GBR', 'ISL', 'IRL', 'ITA', 'JPN', 'KOR', 'NLD', 'NZL',
                                'NOR', 'PRT', 'ESP', 'SWE', 'CHE', 'USA'],
        'Developing': ['ARG', 'BRA', 'CHL', 'MEX', 'ZAF', 'PRI', 'TUR', 'URY', 'VEN'],
        'Low-income': ['DOM', 'GHA', 'IND', 'NGA', 'PER', 'PHL']
    }


# Country name mapping for display
COUNTRY_NAMES = {
    'ALB': 'Albania', 'ARG': 'Argentina', 'ARM': 'Armenia', 'AUS': 'Australia',
    'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BGD': 'Bangladesh', 'BLR': 'Belarus',
    'BEL': 'Belgium', 'BIH': 'Bosnia', 'BRA': 'Brazil', 'BGR': 'Bulgaria',
    'CAN': 'Canada', 'CHL': 'Chile', 'CHN': 'China', 'COL': 'Colombia',
    'HRV': 'Croatia', 'CZE': 'Czech Republic', 'DNK': 'Denmark', 'DOM': 'Dominican Republic',
    'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France', 'GEO': 'Georgia',
    'DEU': 'Germany', 'GHA': 'Ghana', 'GBR': 'Great Britain', 'HUN': 'Hungary',
    'ISL': 'Iceland', 'IND': 'India', 'IRL': 'Ireland', 'ITA': 'Italy',
    'JPN': 'Japan', 'KOR': 'South Korea', 'LVA': 'Latvia', 'LTU': 'Lithuania',
    'MKD': 'Macedonia', 'MEX': 'Mexico', 'MDA': 'Moldova', 'NLD': 'Netherlands',
    'NZL': 'New Zealand', 'NGA': 'Nigeria', 'NIR': 'N. Ireland', 'NOR': 'Norway',
    'PAK': 'Pakistan', 'PER': 'Peru', 'PHL': 'Philippines', 'POL': 'Poland',
    'PRT': 'Portugal', 'PRI': 'Puerto Rico', 'ROU': 'Romania', 'RUS': 'Russia',
    'SRB': 'Serbia', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ZAF': 'South Africa',
    'ESP': 'Spain', 'SWE': 'Sweden', 'CHE': 'Switzerland', 'TWN': 'Taiwan',
    'TUR': 'Turkey', 'UKR': 'Ukraine', 'USA': 'U.S.A.', 'URY': 'Uruguay',
    'VEN': 'Venezuela'
}


if __name__ == "__main__":
    print("Computing factor scores with WVS + EVS data...")
    scores, loadings, means = compute_nation_level_factor_scores()

    print(f"\nFactor loadings (nation-level):")
    print(loadings.to_string(index=False))

    print(f"\nFactor scores for {len(scores)} countries:")
    scores['name'] = scores['COUNTRY_ALPHA'].map(COUNTRY_NAMES).fillna('')
    for _, row in scores.sort_values('trad_secrat', ascending=False).iterrows():
        name = str(row.get('name', ''))
        print(f"  {row['COUNTRY_ALPHA']:4s} {name:20s} trad={row['trad_secrat']:+.3f}  surv={row['surv_selfexp']:+.3f}")

    print(f"\nVerification:")
    print(f"  Japan should be high secular-rational (positive)")
    print(f"  Sweden should be high secular-rational AND high self-expression")
    print(f"  Nigeria should be low on both (negative)")
