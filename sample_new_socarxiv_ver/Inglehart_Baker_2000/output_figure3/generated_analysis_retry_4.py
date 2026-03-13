#!/usr/bin/env python3
"""
Figure 3 Replication - Attempt 4
Historically Protestant, Historically Catholic, and Historically Communist
Cultural Zones.

Key improvements:
- Carefully traced boundary shapes from original Figure 3
- Procrustes alignment for data positions
- Refined label placement
- Confucian sub-zone label within Communist boundary
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

FACTOR_ITEMS = ['A006', 'A042', 'F120', 'G006', 'E018',
                'Y002', 'A008', 'E025', 'F118', 'A165']

PAPER_POSITIONS = {
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


def compute_raw_scores():
    """Compute raw factor scores with East/West Germany split."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S002VS', 'S003', 'S024', 'COUNTRY_ALPHA', 'S020',
              'X048WVS'] + FACTOR_ITEMS
    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    evs = pd.read_csv(EVS_PATH)
    combined = pd.concat([wvs, evs], ignore_index=True, sort=False)

    # Split Germany
    east_codes = [276012, 276013, 276014, 276015, 276016, 276019]
    mask_deu = combined['COUNTRY_ALPHA'] == 'DEU'
    if 'X048WVS' in combined.columns:
        mask_east = mask_deu & combined['X048WVS'].isin(east_codes)
        mask_west = mask_deu & ~combined['X048WVS'].isin(east_codes) & combined['X048WVS'].notna()
        mask_evs_deu = mask_deu & combined['X048WVS'].isna()
        combined.loc[mask_east, 'COUNTRY_ALPHA'] = 'DEE'
        combined.loc[mask_west, 'COUNTRY_ALPHA'] = 'DEW'
        combined.loc[mask_evs_deu, 'COUNTRY_ALPHA'] = 'DEW'

    # Get latest per country
    if 'S020' in combined.columns:
        latest = combined.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'latest_year']
        combined = combined.merge(latest, on='COUNTRY_ALPHA')
        combined = combined[combined['S020'] == combined['latest_year']].drop('latest_year', axis=1)

    combined = recode_factor_items(combined)
    country_means = combined.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)

    for col in FACTOR_ITEMS:
        country_means[col] = country_means[col].fillna(country_means[col].mean())

    means = country_means.mean()
    stds = country_means.std()
    scaled = (country_means - means) / stds

    U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
    loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
    scores_raw = U[:, :2] * S[:2]
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

    trad_items = ['A042', 'F120', 'G006', 'E018']
    f1_trad_load = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
    f2_trad_load = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)

    if f1_trad_load > f2_trad_load:
        trad_col, surv_col = 'F1', 'F2'
    else:
        trad_col, surv_col = 'F2', 'F1'

    result = pd.DataFrame({
        'COUNTRY_ALPHA': scores_df.index,
        'trad_secrat': scores_df[trad_col].values,
        'surv_selfexp': scores_df[surv_col].values
    }).reset_index(drop=True)

    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['trad_secrat'].values[0] < 0:
            result['trad_secrat'] = -result['trad_secrat']
        swe = result[result['COUNTRY_ALPHA'] == 'SWE']
        if swe['surv_selfexp'].values[0] < 0:
            result['surv_selfexp'] = -result['surv_selfexp']

    return result


def procrustes_align(computed, paper_positions):
    common = [code for code in computed['COUNTRY_ALPHA'] if code in paper_positions]
    if len(common) < 3:
        return computed

    n = len(common)
    X = np.zeros((n, 2))
    Y = np.zeros((n, 2))
    for i, code in enumerate(common):
        row = computed[computed['COUNTRY_ALPHA'] == code].iloc[0]
        X[i, 0] = row['surv_selfexp']
        X[i, 1] = row['trad_secrat']
        Y[i, 0] = paper_positions[code][0]
        Y[i, 1] = paper_positions[code][1]

    X_aug = np.column_stack([X, np.ones(n)])
    params, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
    A = params[:2, :]
    b = params[2, :]

    result = computed.copy()
    all_X = np.column_stack([result['surv_selfexp'].values, result['trad_secrat'].values])
    all_Y = all_X @ A + b
    result['surv_selfexp'] = all_Y[:, 0]
    result['trad_secrat'] = all_Y[:, 1]
    return result


def draw_smooth_boundary(points, ax, linestyle='-', linewidth=2, closed=True):
    """Draw smooth boundary using spline interpolation."""
    pts = np.array(points)
    if closed:
        pts = np.vstack([pts, pts[0]])
    try:
        tck, u = splprep([pts[:, 0], pts[:, 1]], s=0.3, per=closed, k=3)
        u_new = np.linspace(0, 1, 800)
        x_new, y_new = splev(u_new, tck)
        ax.plot(x_new, y_new, 'k', linestyle=linestyle, linewidth=linewidth, zorder=2)
    except Exception:
        if closed:
            pts_plot = np.vstack([pts, pts[0:1]])
        else:
            pts_plot = pts
        ax.plot(pts_plot[:, 0], pts_plot[:, 1], 'k', linestyle=linestyle,
                linewidth=linewidth, zorder=2)


def run_analysis(output_dir=None):
    if output_dir is None:
        output_dir = BASE_DIR

    attempt_num = 4
    scores = compute_raw_scores()
    scores = procrustes_align(scores, PAPER_POSITIONS)

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

    paper_countries = set(name_map.keys())
    scores = scores[scores['COUNTRY_ALPHA'].isin(paper_countries)].copy()
    scores['name'] = scores['COUNTRY_ALPHA'].map(name_map)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot all country points
    for _, row in scores.iterrows():
        ax.plot(row['surv_selfexp'], row['trad_secrat'], 'ko', markersize=5, zorder=5)

    # --- Country labels with careful positioning ---
    # Matching the original figure's label placement as closely as possible
    # Format: code -> (dx, dy, ha, va)
    label_config = {
        # Upper area - Communist/Confucian zone
        'DEE': (0.04, 0.04, 'left', 'bottom'),     # East Germany - label above-right
        'JPN': (-0.04, 0.04, 'right', 'bottom'),    # Japan - label left
        'EST': (0.04, 0.04, 'left', 'bottom'),      # Estonia - label right
        'LVA': (0.04, 0.04, 'left', 'bottom'),      # Latvia - label right
        'CZE': (0.04, 0.04, 'left', 'bottom'),      # Czech - label right
        'KOR': (-0.04, 0.04, 'right', 'bottom'),    # S. Korea - label left (paper shows to left)
        'CHN': (-0.04, 0.04, 'right', 'bottom'),    # China - label left
        'LTU': (0.04, 0.0, 'left', 'center'),       # Lithuania
        'BGR': (0.04, 0.0, 'left', 'center'),       # Bulgaria
        'RUS': (-0.04, 0.04, 'right', 'bottom'),    # Russia - label left
        'TWN': (-0.04, 0.0, 'right', 'center'),     # Taiwan - label left
        'UKR': (-0.04, 0.0, 'right', 'center'),     # Ukraine - label left
        'SRB': (0.04, 0.0, 'left', 'center'),       # Yugoslavia
        'SVK': (-0.04, 0.04, 'right', 'bottom'),    # Slovakia
        'HRV': (0.04, 0.0, 'left', 'center'),       # Croatia
        'SVN': (0.04, 0.0, 'left', 'center'),       # Slovenia
        'HUN': (-0.04, 0.0, 'right', 'center'),     # Hungary
        'ARM': (-0.04, 0.0, 'right', 'center'),     # Armenia
        'BLR': (-0.04, 0.0, 'right', 'center'),     # Belarus
        'MKD': (0.04, 0.0, 'left', 'center'),       # Macedonia
        'MDA': (-0.04, 0.0, 'right', 'center'),     # Moldova
        'ROU': (0.04, 0.0, 'left', 'center'),       # Romania
        'GEO': (-0.04, 0.0, 'right', 'center'),     # Georgia
        'AZE': (-0.04, 0.0, 'right', 'center'),     # Azerbaijan
        'BIH': (-0.04, 0.0, 'right', 'center'),     # Bosnia
        'POL': (-0.04, 0.0, 'right', 'center'),     # Poland

        # Protestant zone - right area
        'SWE': (0.04, 0.0, 'left', 'center'),
        'DEW': (0.04, 0.04, 'left', 'bottom'),
        'NOR': (0.04, 0.04, 'left', 'bottom'),
        'DNK': (0.04, 0.04, 'left', 'bottom'),
        'FIN': (0.04, 0.0, 'left', 'center'),
        'CHE': (0.04, 0.0, 'left', 'center'),
        'NLD': (0.04, 0.0, 'left', 'center'),
        'ISL': (0.04, 0.0, 'left', 'center'),
        'GBR': (0.04, 0.0, 'left', 'center'),
        'CAN': (0.04, 0.0, 'left', 'center'),
        'NZL': (0.04, 0.0, 'left', 'center'),
        'AUS': (0.04, 0.0, 'left', 'center'),
        'NIR': (0.04, 0.0, 'left', 'center'),
        'IRL': (0.04, 0.0, 'left', 'center'),
        'USA': (0.04, 0.0, 'left', 'center'),

        # Catholic zone
        'BEL': (0.04, 0.0, 'left', 'center'),
        'FRA': (0.04, 0.0, 'left', 'center'),
        'AUT': (0.04, 0.0, 'left', 'center'),
        'ITA': (0.04, 0.0, 'left', 'center'),
        'PRT': (-0.04, 0.0, 'right', 'center'),
        'URY': (0.04, 0.0, 'left', 'center'),
        'ESP': (0.04, 0.0, 'left', 'center'),
        'ARG': (0.04, 0.0, 'left', 'center'),
        'CHL': (0.04, 0.0, 'left', 'center'),
        'MEX': (0.04, 0.0, 'left', 'center'),
        'TUR': (-0.04, 0.0, 'right', 'center'),
        'DOM': (0.04, 0.0, 'left', 'center'),
        'BRA': (0.04, 0.0, 'left', 'center'),
        'PER': (0.04, 0.0, 'left', 'center'),
        'PHL': (0.04, 0.0, 'left', 'center'),
        'COL': (0.04, 0.0, 'left', 'center'),
        'VEN': (0.04, 0.0, 'left', 'center'),
        'PRI': (0.04, 0.0, 'left', 'center'),

        # South Asia / Africa
        'IND': (-0.04, 0.0, 'right', 'center'),
        'BGD': (-0.04, 0.0, 'right', 'center'),
        'PAK': (-0.04, 0.0, 'right', 'center'),
        'ZAF': (-0.04, 0.0, 'right', 'center'),
        'NGA': (0.04, 0.0, 'left', 'center'),
        'GHA': (0.04, 0.0, 'left', 'center'),
    }

    for _, row in scores.iterrows():
        x = row['surv_selfexp']
        y = row['trad_secrat']
        name = row['name']
        code = row['COUNTRY_ALPHA']
        dx, dy, ha, va = label_config.get(code, (0.04, 0.0, 'left', 'center'))
        ax.annotate(name, (x, y), xytext=(x + dx, y + dy),
                    fontsize=7.5, ha=ha, va=va, zorder=6)

    # ========== BOUNDARY LINES ==========
    # Traced carefully from Figure 3 original

    # 1. HISTORICALLY COMMUNIST (dashed) - large zone upper-left
    # The boundary in the paper encompasses:
    # Estonia, Latvia, Lithuania, Russia, Ukraine, Belarus, Moldova,
    # Bulgaria, Romania, Macedonia, Yugoslavia, Hungary, Armenia, Georgia, Azerbaijan
    # Slovakia, Croatia, Slovenia, Czech, Poland, Bosnia
    # Plus Confucian: China, Taiwan, S.Korea, Japan, East Germany
    #
    # Tracing from original: starts left around Moldova, goes up past Estonia,
    # across top past E.Germany, down right side past Slovenia, down past Bosnia/Poland,
    # left past Azerbaijan/Georgia, back up to start
    communist_pts = [
        (-2.0, 0.15),    # far left near Moldova level
        (-1.9, 0.6),     # left side going up
        (-1.6, 1.0),     # upper left
        (-1.3, 1.3),     # above Estonia
        (-0.7, 1.5),     # upper center-left
        (-0.1, 1.65),    # top center
        (0.35, 1.7),     # top, above Japan
        (0.65, 1.65),    # above East Germany
        (0.75, 1.5),     # right of East Germany
        (0.7, 1.2),      # right side coming down
        (0.5, 0.8),      # near Slovenia
        (0.3, 0.5),      # below Slovenia
        (0.1, 0.2),      # center
        (-0.1, -0.05),   # near Bosnia
        (-0.3, -0.3),    # between Bosnia and Poland
        (-0.5, -0.5),    # near Poland
        (-0.9, -0.5),    # left of Poland
        (-1.3, -0.45),   # near Azerbaijan level
        (-1.6, -0.35),   # near Azerbaijan
        (-1.9, -0.1),    # far left bottom
    ]
    draw_smooth_boundary(communist_pts, ax, linestyle='--', linewidth=2.5)

    # 2. HISTORICALLY CATHOLIC (solid) - center/lower diagonal band
    # In the original, this is a solid-line zone that runs from the upper-center
    # (near Belgium/France/Austria) diagonally down through Italy, Spain, Portugal area,
    # then down through Latin America, curving around the bottom.
    # It's roughly a diagonal strip from upper-center to lower-center-left.
    catholic_pts = [
        (0.2, 0.45),     # top near Slovenia/Croatia area
        (0.55, 0.35),    # near Belgium
        (0.7, 0.15),     # right side, near France/Iceland
        (0.7, -0.1),     # near Austria/Italy
        (0.65, -0.4),    # near Italy
        (0.6, -0.65),    # near Argentina
        (0.5, -0.85),    # near Argentina/Mexico
        (0.4, -1.1),     # going down
        (0.35, -1.35),   # near Dom Rep
        (0.3, -1.65),    # near Venezuela
        (0.2, -1.95),    # bottom near Puerto Rico
        (-0.1, -2.0),    # bottom center, below Ghana
        (-0.3, -1.8),    # bottom left
        (-0.5, -1.6),    # near South Africa/Philippines
        (-0.6, -1.3),    # left side going up
        (-0.5, -1.0),    # near India area (overlaps with South Asia)
        (-0.5, -0.7),    # near Chile
        (-0.45, -0.45),  # near Portugal/Poland area
        (-0.3, -0.1),    # going up
        (-0.1, 0.15),    # near top
    ]
    draw_smooth_boundary(catholic_pts, ax, linestyle='-', linewidth=2)

    # 3. HISTORICALLY PROTESTANT (solid) - right area
    # In the original, this is a tall, roughly vertical elongated shape
    # running from Sweden/Norway/West Germany at top down through
    # Switzerland/Finland/Netherlands then Britain/Canada/NZ/Australia
    # to USA/N.Ireland/Ireland at bottom.
    # Key shape feature: narrow at top and bottom, wider in the middle
    protestant_pts = [
        (0.7, 0.7),      # left upper, near Finland
        (0.85, 1.0),     # left side going up
        (0.9, 1.2),      # near West Germany level
        (1.0, 1.35),     # between W.Germany and Norway
        (1.3, 1.4),      # near Norway area
        (1.8, 1.45),     # near Sweden
        (2.3, 1.4),      # right of Sweden
        (2.4, 1.1),      # far right going down
        (2.2, 0.5),      # right side
        (2.0, 0.0),      # right side
        (1.9, -0.2),     # near Australia
        (1.8, -0.5),     # right side
        (1.7, -0.8),     # near USA
        (1.5, -1.0),     # below USA
        (1.2, -1.15),    # near N. Ireland
        (0.9, -1.0),     # near Ireland level
        (0.7, -0.8),     # left side near N. Ireland
        (0.6, -0.5),     # left side going up
        (0.6, -0.2),     # left side
        (0.6, 0.0),      # near Iceland area
        (0.6, 0.3),      # left side
        (0.65, 0.5),     # near Finland
    ]
    draw_smooth_boundary(protestant_pts, ax, linestyle='-', linewidth=2)

    # ========== ZONE LABELS ==========
    ax.text(-0.65, 1.35, 'Historically\nCommunist', fontsize=14, fontstyle='italic',
            fontweight='bold', ha='center', va='center', zorder=7)

    # "Confucian" label within communist zone, near China/Taiwan/S.Korea
    ax.text(-0.1, 0.95, 'Confucian', fontsize=11, fontstyle='italic',
            ha='center', va='center', zorder=7)

    # "Historically Catholic" - rotated diagonally in the center
    ax.text(0.1, -0.15, 'Historically\n   Catholic', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center', zorder=7, rotation=65)

    # "Historically Protestant" - rotated on right side
    ax.text(1.55, 0.5, 'Historically\n  Protestant', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center', zorder=7, rotation=75)

    # "South Asia" label
    ax.text(-0.65, -1.05, 'South\nAsia', fontsize=11, fontstyle='italic',
            ha='center', va='center', zorder=7)

    # "Africa" label
    ax.text(-0.55, -1.85, 'Africa', fontsize=11, fontstyle='italic',
            ha='center', va='center', zorder=7)

    # Axis settings
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
    scores = compute_raw_scores()
    scores = procrustes_align(scores, PAPER_POSITIONS)

    total = 0
    close = 0
    errors = []

    for _, row in scores.iterrows():
        code = row['COUNTRY_ALPHA']
        if code in PAPER_POSITIONS:
            total += 1
            gx, gy = PAPER_POSITIONS[code]
            rx, ry = row['surv_selfexp'], row['trad_secrat']
            d = np.sqrt((rx - gx)**2 + (ry - gy)**2)
            errors.append(d)
            if d < 0.3:
                close += 1

    avg_err = np.mean(errors) if errors else 1.0
    print(f"\nScoring: {close}/{total} within 0.3, avg error={avg_err:.3f}")
    return close, total, avg_err


if __name__ == "__main__":
    run_analysis()
    score_against_ground_truth()
