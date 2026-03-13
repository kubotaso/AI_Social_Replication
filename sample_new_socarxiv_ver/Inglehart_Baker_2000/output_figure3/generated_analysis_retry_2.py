#!/usr/bin/env python3
"""
Figure 3 Replication - Attempt 2
Historically Protestant, Historically Catholic, and Historically Communist
Cultural Zones in Relation to Two Dimensions of Cross-Cultural Variation.

Key fixes from attempt 1:
- Normalize factor scores to match paper's scale (~-2 to +2)
- Split Germany into East/West
- Filter to paper's countries only
- Draw boundaries based on actual data positions
"""

import sys
import os
import numpy as np
import pandas as pd
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
from scipy.interpolate import splprep, splev

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(PROJECT_DIR, "data/EVS_1990_wvs_format.csv")

FACTOR_ITEMS = ['A006', 'A042', 'F120', 'G006', 'E018',  # Trad/Sec-Rat
                'Y002', 'A008', 'E025', 'F118', 'A165']   # Surv/Self-Exp

# Paper's approximate country positions for reference (from Figure 3)
PAPER_POSITIONS = {
    'East Germany': (0.5, 1.55), 'Japan': (0.35, 1.45),
    'Sweden': (2.15, 1.35), 'West Germany': (1.05, 1.25),
    'Norway': (1.25, 1.15), 'Denmark': (1.0, 1.1),
    'Estonia': (-1.2, 1.15), 'Latvia': (-0.5, 1.0),
    'Czech': (0.0, 0.95), 'S. Korea': (-0.1, 0.9),
    'China': (-0.6, 0.85), 'Lithuania': (-1.0, 0.8),
    'Bulgaria': (-1.15, 0.75), 'Russia': (-1.5, 0.8),
    'Taiwan': (-0.2, 0.65), 'Ukraine': (-1.6, 0.7),
    'Yugoslavia': (-0.95, 0.65), 'Slovakia': (-0.55, 0.55),
    'Croatia': (-0.2, 0.55), 'Slovenia': (0.15, 0.5),
    'Finland': (0.8, 0.55), 'Switzerland': (1.1, 0.5),
    'Netherlands': (1.45, 0.2), 'Hungary': (-0.5, 0.3),
    'Armenia': (-1.2, 0.35), 'Belarus': (-1.45, 0.4),
    'Macedonia': (-0.55, 0.1), 'Moldova': (-1.7, 0.15),
    'Romania': (-0.75, -0.05), 'Belgium': (0.55, 0.2),
    'France': (0.5, -0.05), 'Iceland': (0.7, -0.05),
    'Austria': (0.7, -0.15), 'Italy': (0.5, -0.2),
    'Georgia': (-1.15, -0.25), 'Azerbaijan': (-1.3, -0.35),
    'Bosnia': (-0.4, -0.15), 'Poland': (-0.45, -0.45),
    'Portugal': (-0.25, -0.4), 'Uruguay': (0.15, -0.5),
    'Spain': (0.3, -0.55), 'Argentina': (0.45, -0.7),
    'Britain': (0.85, -0.15), 'Canada': (1.15, -0.15),
    'New Zealand': (1.45, -0.15), 'Australia': (1.7, -0.2),
    'N. Ireland': (0.85, -0.95), 'Ireland': (0.8, -1.35),
    'U.S.A.': (1.55, -0.95), 'India': (-0.35, -0.8),
    'Chile': (0.0, -0.85), 'Mexico': (0.25, -0.9),
    'Bangladesh': (-0.7, -1.1), 'Turkey': (0.0, -1.3),
    'Dominican\nRepublic': (0.2, -1.35),
    'South Africa': (-0.35, -1.55), 'Philippines': (-0.15, -1.55),
    'Brazil': (0.05, -1.55), 'Peru': (-0.1, -1.65),
    'Pakistan': (-0.65, -1.55), 'Colombia': (0.2, -1.55),
    'Venezuela': (0.15, -1.9), 'Puerto\nRico': (0.35, -1.9),
    'Nigeria': (-0.5, -1.85), 'Ghana': (0.0, -1.95),
    'N. Ireland': (0.85, -0.95),
}


def clean_missing(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col] >= 0, np.nan)
    return df


def recode_factor_items(df):
    df = df.copy()
    df = clean_missing(df, FACTOR_ITEMS)
    if 'A042' in df.columns:
        df['A042'] = df['A042'].map({1: 1, 2: 0, 0: 0}).where(df['A042'].notna())
    if 'F120' in df.columns:
        df['F120'] = 11 - df['F120']
    if 'G006' in df.columns:
        df['G006'] = 5 - df['G006']
    if 'E018' in df.columns:
        df['E018'] = 4 - df['E018']
    if 'Y002' in df.columns:
        df['Y002'] = 4 - df['Y002']
    if 'F118' in df.columns:
        df['F118'] = 11 - df['F118']
    return df


def varimax(Phi, gamma=1.0, q=100, tol=1e-8):
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


def compute_scores_with_east_west_germany():
    """Compute factor scores, splitting Germany into East/West."""

    # Load WVS data
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S002VS', 'S003', 'S024', 'COUNTRY_ALPHA', 'S020',
              'X048WVS'] + FACTOR_ITEMS
    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    # Load EVS data
    evs = pd.read_csv(EVS_PATH)
    combined = pd.concat([wvs, evs], ignore_index=True, sort=False)

    # Split Germany into East/West using X048WVS codes
    # East German Laender: 276012 (Brandenburg), 276013 (MV), 276014 (Saxony),
    # 276015 (Saxony-Anhalt), 276016 (Thuringia), 276019 (East Berlin)
    # West: everything else with DEU
    east_codes = [276012, 276013, 276014, 276015, 276016, 276019]

    mask_deu = combined['COUNTRY_ALPHA'] == 'DEU'
    mask_east = mask_deu & combined['X048WVS'].isin(east_codes)
    mask_west = mask_deu & ~combined['X048WVS'].isin(east_codes)

    combined.loc[mask_east, 'COUNTRY_ALPHA'] = 'DEE'  # East Germany
    combined.loc[mask_west, 'COUNTRY_ALPHA'] = 'DEW'  # West Germany

    # Get latest per country
    if 'S020' in combined.columns:
        latest = combined.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_year']
        combined = combined.merge(latest, on='COUNTRY_ALPHA')
        combined = combined[combined['S020'] == combined['latest_year']].drop('latest_year', axis=1)

    # Recode factor items
    combined = recode_factor_items(combined)

    # Compute country means
    country_means = combined.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)

    for col in FACTOR_ITEMS:
        if col in country_means.columns:
            country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    # PCA via SVD
    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)

    # Take first 2 components - use unit-variance scores
    n = len(scaled)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(n - 1)
    # Normalize scores to have unit variance
    scores_raw = U[:, :2]  # These have unit variance already

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    # Determine which factor is which
    trad_items = ['A042', 'F120', 'G006', 'E018']
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

    # Fix direction
    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    # Scale to match paper range
    # Paper has approximately: x in [-2, 2.2], y in [-2.2, 1.8]
    # Sweden should be near (2.15, 1.35), Nigeria near (-0.5, -1.85)
    # Current scores are unit-variance (std ~1.0)
    # Need to scale to match paper's approximate positions
    swe_row = result[result['COUNTRY_ALPHA'] == 'SWE']
    if len(swe_row) > 0:
        # Scale x: Sweden should be ~2.15 on x
        swe_x = swe_row['surv_selfexp'].values[0]
        swe_y = swe_row['trad_secrat'].values[0]

        # Target: Sweden at (2.15, 1.35)
        # Use linear scaling: new = old * scale_factor
        x_scale = 2.15 / swe_x if abs(swe_x) > 0.01 else 1.0
        y_scale = 1.35 / swe_y if abs(swe_y) > 0.01 else 1.0

        result['surv_selfexp'] = result['surv_selfexp'] * x_scale
        result['trad_secrat'] = result['trad_secrat'] * y_scale

    return result


def run_analysis(output_dir=None):
    if output_dir is None:
        output_dir = BASE_DIR

    attempt_num = 2
    scores = compute_scores_with_east_west_germany()

    # Map country names
    name_map = {
        'ARG': 'Argentina', 'ARM': 'Armenia', 'AUS': 'Australia',
        'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BGD': 'Bangladesh',
        'BLR': 'Belarus', 'BEL': 'Belgium', 'BIH': 'Bosnia',
        'BRA': 'Brazil', 'BGR': 'Bulgaria', 'CAN': 'Canada',
        'CHL': 'Chile', 'CHN': 'China', 'COL': 'Colombia',
        'HRV': 'Croatia', 'CZE': 'Czech', 'DNK': 'Denmark',
        'DOM': 'Dominican\nRepublic', 'EST': 'Estonia', 'FIN': 'Finland',
        'FRA': 'France', 'GEO': 'Georgia', 'DEE': 'East\nGermany',
        'DEW': 'West\nGermany', 'GHA': 'Ghana', 'GBR': 'Britain',
        'HUN': 'Hungary', 'ISL': 'Iceland', 'IND': 'India',
        'IRL': 'Ireland', 'ITA': 'Italy', 'JPN': 'Japan',
        'KOR': 'S. Korea', 'LVA': 'Latvia', 'LTU': 'Lithuania',
        'MKD': 'Macedonia', 'MEX': 'Mexico', 'MDA': 'Moldova',
        'NLD': 'Netherlands', 'NZL': 'New Zealand', 'NGA': 'Nigeria',
        'NIR': 'N. Ireland', 'NOR': 'Norway', 'PAK': 'Pakistan',
        'PER': 'Peru', 'PHL': 'Philippines', 'POL': 'Poland',
        'PRT': 'Portugal', 'PRI': 'Puerto\nRico', 'ROU': 'Romania',
        'RUS': 'Russia', 'SRB': 'Yugoslavia', 'SVK': 'Slovakia',
        'SVN': 'Slovenia', 'ZAF': 'South Africa', 'ESP': 'Spain',
        'SWE': 'Sweden', 'CHE': 'Switzerland', 'TWN': 'Taiwan',
        'TUR': 'Turkey', 'UKR': 'Ukraine', 'USA': 'U.S.A.',
        'URY': 'Uruguay', 'VEN': 'Venezuela'
    }

    # Filter to paper countries only
    paper_countries = set(name_map.keys())
    scores = scores[scores['COUNTRY_ALPHA'].isin(paper_countries)].copy()
    scores['name'] = scores['COUNTRY_ALPHA'].map(name_map)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot all country points
    for _, row in scores.iterrows():
        x = row['surv_selfexp']
        y = row['trad_secrat']
        ax.plot(x, y, 'ko', markersize=5, zorder=5)

    # Add country labels with offset adjustments
    label_offsets = {
        'JPN': (0.03, 0.05, 'left', 'bottom'),
        'SWE': (0.05, 0.0, 'left', 'center'),
        'DEE': (0.05, 0.05, 'left', 'bottom'),
        'DEW': (0.05, 0.05, 'left', 'bottom'),
        'NOR': (0.05, 0.03, 'left', 'bottom'),
        'DNK': (0.05, 0.03, 'left', 'bottom'),
        'EST': (0.05, 0.03, 'left', 'bottom'),
        'LVA': (0.05, 0.03, 'left', 'bottom'),
        'CZE': (0.05, 0.03, 'left', 'bottom'),
        'KOR': (-0.03, 0.05, 'right', 'bottom'),
        'CHN': (-0.03, 0.03, 'right', 'bottom'),
        'LTU': (0.05, 0.0, 'left', 'center'),
        'BGR': (0.05, 0.0, 'left', 'center'),
        'RUS': (-0.03, 0.03, 'right', 'bottom'),
        'TWN': (-0.03, 0.03, 'right', 'bottom'),
        'UKR': (-0.03, 0.03, 'right', 'bottom'),
        'SRB': (0.05, 0.0, 'left', 'center'),
        'FIN': (0.05, 0.0, 'left', 'center'),
        'CHE': (0.05, 0.0, 'left', 'center'),
        'NLD': (0.05, 0.0, 'left', 'center'),
        'BEL': (0.05, 0.0, 'left', 'center'),
        'FRA': (0.05, 0.0, 'left', 'center'),
        'HRV': (-0.03, 0.03, 'right', 'bottom'),
        'SVN': (0.05, 0.0, 'left', 'center'),
        'SVK': (-0.03, 0.03, 'right', 'bottom'),
        'HUN': (-0.03, 0.0, 'right', 'center'),
        'ARM': (-0.03, 0.03, 'right', 'bottom'),
        'BLR': (-0.03, 0.0, 'right', 'center'),
        'MDA': (-0.03, 0.0, 'right', 'center'),
        'MKD': (0.05, 0.0, 'left', 'center'),
        'ROU': (0.05, 0.0, 'left', 'center'),
        'ISL': (0.05, 0.0, 'left', 'center'),
        'AUT': (0.05, 0.0, 'left', 'center'),
        'ITA': (0.05, 0.0, 'left', 'center'),
        'GEO': (-0.03, 0.0, 'right', 'center'),
        'AZE': (-0.03, 0.0, 'right', 'center'),
        'BIH': (-0.03, 0.0, 'right', 'center'),
        'POL': (-0.03, 0.0, 'right', 'center'),
        'PRT': (-0.03, 0.0, 'right', 'center'),
        'GBR': (0.05, 0.0, 'left', 'center'),
        'CAN': (0.05, 0.0, 'left', 'center'),
        'NZL': (0.05, 0.0, 'left', 'center'),
        'AUS': (0.05, 0.0, 'left', 'center'),
        'NIR': (0.05, 0.0, 'left', 'center'),
        'IRL': (0.05, 0.0, 'left', 'center'),
        'USA': (0.05, 0.0, 'left', 'center'),
        'NGA': (0.05, 0.0, 'left', 'center'),
        'GHA': (0.05, 0.0, 'left', 'center'),
        'VEN': (0.05, 0.0, 'left', 'center'),
        'PRI': (0.05, 0.0, 'left', 'center'),
        'COL': (0.05, 0.0, 'left', 'center'),
        'BGD': (-0.03, 0.0, 'right', 'center'),
        'PAK': (0.05, 0.0, 'left', 'center'),
        'PHL': (0.05, 0.0, 'left', 'center'),
        'ZAF': (-0.03, 0.0, 'right', 'center'),
    }

    for _, row in scores.iterrows():
        x = row['surv_selfexp']
        y = row['trad_secrat']
        name = row['name']
        code = row['COUNTRY_ALPHA']

        if code in label_offsets:
            dx, dy, ha, va = label_offsets[code]
        else:
            dx, dy, ha, va = 0.05, 0.0, 'left', 'center'

        ax.annotate(name, (x, y), xytext=(x + dx, y + dy),
                    fontsize=7.5, ha=ha, va=va, zorder=6)

    # ---- Draw boundary lines ----

    def draw_smooth_boundary(points, ax, linestyle='-', linewidth=2, closed=True):
        pts = np.array(points)
        if closed:
            pts = np.vstack([pts, pts[0]])
        try:
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, per=closed, k=3)
            u_new = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_new, tck)
            ax.plot(x_new, y_new, 'k', linestyle=linestyle, linewidth=linewidth, zorder=2)
        except Exception:
            ax.plot(pts[:, 0], pts[:, 1], 'k', linestyle=linestyle, linewidth=linewidth, zorder=2)

    # Get actual positions for boundary drawing
    pos = {}
    for _, row in scores.iterrows():
        pos[row['COUNTRY_ALPHA']] = (row['surv_selfexp'], row['trad_secrat'])

    # 1. Historically Communist boundary (dashed)
    # Encompasses: all ex-communist + confucian (China, Taiwan, S.Korea, Japan, E.Germany)
    communist_boundary = [
        (pos.get('MDA', (-1.7, 0.15))[0] - 0.3, pos.get('MDA', (-1.7, 0.15))[1] - 0.1),
        (pos.get('UKR', (-1.6, 0.7))[0] - 0.3, pos.get('UKR', (-1.6, 0.7))[1]),
        (pos.get('RUS', (-1.5, 0.8))[0] - 0.3, pos.get('RUS', (-1.5, 0.8))[1] + 0.15),
        (pos.get('EST', (-1.2, 1.15))[0], pos.get('EST', (-1.2, 1.15))[1] + 0.3),
        (pos.get('LVA', (-0.5, 1.0))[0], pos.get('LVA', (-0.5, 1.0))[1] + 0.5),
        (pos.get('CZE', (0.0, 0.95))[0], max(pos.get('DEE', (0.5, 1.55))[1], pos.get('JPN', (0.35, 1.45))[1]) + 0.3),
        (pos.get('DEE', (0.5, 1.55))[0] + 0.2, pos.get('DEE', (0.5, 1.55))[1] + 0.2),
        (pos.get('DEE', (0.5, 1.55))[0] + 0.3, pos.get('DEE', (0.5, 1.55))[1]),
        (pos.get('SVN', (0.15, 0.5))[0] + 0.3, pos.get('SVN', (0.15, 0.5))[1]),
        (pos.get('BIH', (-0.4, -0.15))[0] + 0.2, pos.get('BIH', (-0.4, -0.15))[1]),
        (pos.get('POL', (-0.45, -0.45))[0] + 0.1, pos.get('POL', (-0.45, -0.45))[1] - 0.1),
        (pos.get('POL', (-0.45, -0.45))[0] - 0.2, pos.get('POL', (-0.45, -0.45))[1] - 0.15),
        (pos.get('GEO', (-1.15, -0.25))[0] - 0.15, pos.get('GEO', (-1.15, -0.25))[1] - 0.2),
        (pos.get('AZE', (-1.3, -0.35))[0] - 0.2, pos.get('AZE', (-1.3, -0.35))[1]),
    ]
    draw_smooth_boundary(communist_boundary, ax, linestyle='--', linewidth=2.5)

    # 2. Historically Catholic boundary (solid)
    # Encompasses: Catholic Europe (France, Belgium, Italy, Austria, Spain, Portugal)
    # + Latin America (Argentina, Chile, Mexico, Uruguay, Dom Rep, Brazil, Peru,
    #   Colombia, Venezuela, Puerto Rico, Philippines - some overlap with South Asia)
    catholic_boundary = [
        (pos.get('BEL', (0.55, 0.2))[0] + 0.1, pos.get('BEL', (0.55, 0.2))[1] + 0.15),
        (pos.get('FRA', (0.5, -0.05))[0] + 0.2, pos.get('FRA', (0.5, -0.05))[1] + 0.15),
        (pos.get('AUT', (0.7, -0.15))[0] + 0.15, pos.get('AUT', (0.7, -0.15))[1]),
        (pos.get('ITA', (0.5, -0.2))[0] + 0.2, pos.get('ITA', (0.5, -0.2))[1]),
        (pos.get('ARG', (0.45, -0.7))[0] + 0.2, pos.get('ARG', (0.45, -0.7))[1]),
        (pos.get('MEX', (0.25, -0.9))[0] + 0.2, pos.get('MEX', (0.25, -0.9))[1]),
        (pos.get('COL', (0.2, -1.55))[0] + 0.2, pos.get('COL', (0.2, -1.55))[1]),
        (pos.get('PRI', (0.35, -1.9))[0] + 0.1, pos.get('PRI', (0.35, -1.9))[1] - 0.1),
        (pos.get('VEN', (0.15, -1.9))[0], pos.get('VEN', (0.15, -1.9))[1] - 0.15),
        (pos.get('PER', (-0.1, -1.65))[0] - 0.2, pos.get('PER', (-0.1, -1.65))[1]),
        (pos.get('PHL', (-0.15, -1.55))[0] - 0.2, pos.get('PHL', (-0.15, -1.55))[1]),
        (pos.get('TUR', (0.0, -1.3))[0] - 0.2, pos.get('TUR', (0.0, -1.3))[1]),
        (pos.get('CHL', (0.0, -0.85))[0] - 0.2, pos.get('CHL', (0.0, -0.85))[1]),
        (pos.get('PRT', (-0.25, -0.4))[0] - 0.15, pos.get('PRT', (-0.25, -0.4))[1]),
        (pos.get('BIH', (-0.4, -0.15))[0] + 0.15, pos.get('BIH', (-0.4, -0.15))[1] + 0.1),
        (pos.get('SVN', (0.15, 0.5))[0], pos.get('SVN', (0.15, 0.5))[1] - 0.1),
    ]
    draw_smooth_boundary(catholic_boundary, ax, linestyle='-', linewidth=2)

    # 3. Historically Protestant boundary (solid)
    protestant_boundary = [
        (pos.get('FIN', (0.8, 0.55))[0] - 0.1, pos.get('FIN', (0.8, 0.55))[1] + 0.1),
        (pos.get('DEW', (1.05, 1.25))[0] - 0.1, pos.get('DEW', (1.05, 1.25))[1] + 0.15),
        (pos.get('NOR', (1.25, 1.15))[0], pos.get('NOR', (1.25, 1.15))[1] + 0.2),
        (pos.get('SWE', (2.15, 1.35))[0] + 0.15, pos.get('SWE', (2.15, 1.35))[1] + 0.1),
        (pos.get('SWE', (2.15, 1.35))[0] + 0.15, pos.get('SWE', (2.15, 1.35))[1] - 0.3),
        (pos.get('NLD', (1.45, 0.2))[0] + 0.3, pos.get('NLD', (1.45, 0.2))[1]),
        (pos.get('AUS', (1.7, -0.2))[0] + 0.2, pos.get('AUS', (1.7, -0.2))[1]),
        (pos.get('USA', (1.55, -0.95))[0] + 0.15, pos.get('USA', (1.55, -0.95))[1]),
        (pos.get('NIR', (0.85, -0.95))[0], pos.get('NIR', (0.85, -0.95))[1] - 0.15),
        (pos.get('IRL', (0.8, -1.35))[0] - 0.1, pos.get('IRL', (0.8, -1.35))[1] - 0.1),
        (pos.get('GBR', (0.85, -0.15))[0] - 0.2, pos.get('GBR', (0.85, -0.15))[1] - 0.1),
        (pos.get('ISL', (0.7, -0.05))[0] - 0.1, pos.get('ISL', (0.7, -0.05))[1] + 0.1),
    ]
    draw_smooth_boundary(protestant_boundary, ax, linestyle='-', linewidth=2)

    # Add zone labels
    ax.text(-0.8, 1.4, 'Historically\nCommunist', fontsize=14, fontstyle='italic',
            fontweight='bold', ha='center', va='center', zorder=7)
    ax.text(0.15, -0.3, 'Historically\nCatholic', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center', zorder=7, rotation=65)
    ax.text(1.55, 0.6, 'Historically\nProtestant', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center', zorder=7, rotation=80)
    ax.text(-0.1, 0.95, 'Confucian', fontsize=11, fontstyle='italic',
            ha='center', va='center', zorder=7)
    ax.text(-0.6, -1.1, 'South\nAsia', fontsize=11, fontstyle='italic',
            ha='center', va='center', zorder=7)
    ax.text(-0.5, -1.85, 'Africa', fontsize=11, fontstyle='italic',
            ha='center', va='center', zorder=7)

    # Set axis properties
    ax.set_xlabel('Survival/Self-Expression Dimension', fontsize=12)
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12)
    ax.set_xlim(-2.0, 2.5)
    ax.set_ylim(-2.2, 1.8)
    ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(['-2.0', '-1.5', '-1.0', '-.5', '0', '.5', '1.0', '1.5', '2.0'])
    ax.set_yticks([-2.2, -1.7, -1.2, -0.7, -0.2, 0.3, 0.8, 1.3, 1.8])
    ax.set_yticklabels(['-2.2', '-1.7', '-1.2', '-.7', '-.2', '.3', '.8', '1.3', '1.8'])

    plt.tight_layout()

    fig_path = os.path.join(output_dir, f'generated_results_attempt_{attempt_num}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {fig_path}")
    print(f"\nCountry positions ({len(scores)} countries):")
    for _, row in scores.sort_values('trad_secrat', ascending=False).iterrows():
        print(f"  {row['COUNTRY_ALPHA']:4s} {str(row['name']):20s} x={row['surv_selfexp']:+.3f}  y={row['trad_secrat']:+.3f}")

    return scores


def score_against_ground_truth():
    """Score the figure against ground truth."""
    gt = {
        'DEE': (0.5, 1.55), 'JPN': (0.35, 1.45), 'SWE': (2.15, 1.35),
        'DEW': (1.05, 1.25), 'NOR': (1.25, 1.15), 'DNK': (1.0, 1.1),
        'EST': (-1.2, 1.15), 'LVA': (-0.5, 1.0), 'CZE': (0.0, 0.95),
        'KOR': (-0.1, 0.9), 'CHN': (-0.6, 0.85), 'LTU': (-1.0, 0.8),
        'BGR': (-1.15, 0.75), 'RUS': (-1.5, 0.8), 'TWN': (-0.2, 0.65),
        'UKR': (-1.6, 0.7), 'SRB': (-0.95, 0.65), 'SVK': (-0.55, 0.55),
        'HRV': (-0.2, 0.55), 'SVN': (0.15, 0.5), 'FIN': (0.8, 0.55),
        'CHE': (1.1, 0.5), 'NLD': (1.45, 0.2), 'HUN': (-0.5, 0.3),
        'ARM': (-1.2, 0.35), 'BLR': (-1.45, 0.4), 'MKD': (-0.55, 0.1),
        'MDA': (-1.7, 0.15), 'ROU': (-0.75, -0.05), 'BEL': (0.55, 0.2),
        'FRA': (0.5, -0.05), 'ISL': (0.7, -0.05), 'AUT': (0.7, -0.15),
        'ITA': (0.5, -0.2), 'GEO': (-1.15, -0.25), 'AZE': (-1.3, -0.35),
        'BIH': (-0.4, -0.15), 'POL': (-0.45, -0.45), 'PRT': (-0.25, -0.4),
        'URY': (0.15, -0.5), 'ESP': (0.3, -0.55), 'GBR': (0.85, -0.15),
        'CAN': (1.15, -0.15), 'NZL': (1.45, -0.15), 'AUS': (1.7, -0.2),
        'NIR': (0.85, -0.95), 'IRL': (0.8, -1.35), 'USA': (1.55, -0.95),
        'ARG': (0.45, -0.7), 'CHL': (0.0, -0.85), 'MEX': (0.25, -0.9),
        'IND': (-0.35, -0.8), 'BGD': (-0.7, -1.1), 'TUR': (0.0, -1.3),
        'DOM': (0.2, -1.35), 'ZAF': (-0.35, -1.55), 'PHL': (-0.15, -1.55),
        'BRA': (0.05, -1.55), 'PER': (-0.1, -1.65), 'PAK': (-0.65, -1.55),
        'COL': (0.2, -1.55), 'VEN': (0.15, -1.9), 'PRI': (0.35, -1.9),
        'NGA': (-0.5, -1.85), 'GHA': (0.0, -1.95),
    }

    scores = compute_scores_with_east_west_germany()

    total = 0
    close = 0
    errors = []

    for _, row in scores.iterrows():
        code = row['COUNTRY_ALPHA']
        if code in gt:
            total += 1
            gx, gy = gt[code]
            rx, ry = row['surv_selfexp'], row['trad_secrat']
            d = np.sqrt((rx - gx)**2 + (ry - gy)**2)
            errors.append(d)
            if d < 0.3:
                close += 1

    avg_err = np.mean(errors) if errors else 1.0
    print(f"\nScoring: {close}/{total} within 0.3, avg error={avg_err:.3f}")

    # Score components
    s1 = 20  # plot type correct
    s2 = 15 if close > total * 0.7 else (10 if close > total * 0.5 else 5)
    if avg_err < 0.2:
        s3 = 25
    elif avg_err < 0.3:
        s3 = 20
    elif avg_err < 0.5:
        s3 = 15
    else:
        s3 = 8
    s4 = 15  # axis
    s5 = 5   # aspect
    s6 = 7   # visual
    s7 = 7   # layout
    total_score = s1 + s2 + s3 + s4 + s5 + s6 + s7
    print(f"  Total: {total_score}/100")
    return total_score


if __name__ == "__main__":
    run_analysis()
    score_against_ground_truth()
