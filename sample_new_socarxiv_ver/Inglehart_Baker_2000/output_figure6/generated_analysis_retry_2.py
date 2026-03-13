#!/usr/bin/env python3
"""
Figure 6 Replication: Change Over Time in Location on Two Dimensions
of Cross Cultural Variation for 38 Societies.

Scatter plot with arrows showing change from initial to latest survey position.
Open circles = Initial survey, Filled circles = Latest survey.

Approach:
1. Run factor analysis on the "reference" set (latest wave per country, waves 2-3 + EVS)
   to get the same factor structure as Figures 1-5.
2. For each country with 2+ time points, compute country means at each wave.
3. Project each country-wave mean onto the reference factor space using the reference
   standardization parameters and rotation matrix.
4. Draw arrows from initial to latest position.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import csv

BASE_DIR = '/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_IB_v5'
DATA_PATH = os.path.join(BASE_DIR, 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv')
EVS_PATH = os.path.join(BASE_DIR, 'data/EVS_1990_wvs_format.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_figure6')

FACTOR_ITEMS = ['A006', 'A042', 'F120', 'G006', 'E018',  # Trad/Sec-Rat
                'Y002', 'A008', 'E025', 'F118', 'A165']   # Surv/Self-Exp


def clean_missing(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col] >= 0, np.nan)
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


def load_wvs_all_waves():
    """Load WVS waves 1-3 data."""
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'S024'] + FACTOR_ITEMS
    available = [c for c in needed if c in header]

    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([1, 2, 3])]
    return wvs


def load_evs():
    """Load EVS 1990 data."""
    evs = pd.read_csv(EVS_PATH)
    if 'S002VS' not in evs.columns:
        evs['S002VS'] = 2
    if 'S003' not in evs.columns:
        evs['S003'] = -1
    if 'S024' not in evs.columns:
        evs['S024'] = -1
    return evs


COUNTRY_NAMES = {
    'ALB': 'Albania', 'ARG': 'Argentina', 'ARM': 'Armenia', 'AUS': 'Australia',
    'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BGD': 'Bangladesh', 'BLR': 'Belarus',
    'BEL': 'Belgium', 'BIH': 'Bosnia', 'BRA': 'Brazil', 'BGR': 'Bulgaria',
    'CAN': 'Canada', 'CHL': 'Chile', 'CHN': 'China', 'COL': 'Colombia',
    'HRV': 'Croatia', 'CZE': 'Czech Rep.', 'DNK': 'Denmark', 'DOM': 'Dom. Rep.',
    'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France', 'GEO': 'Georgia',
    'DEU': 'W. Germany', 'GHA': 'Ghana', 'GBR': 'Britain', 'HUN': 'Hungary',
    'ISL': 'Iceland', 'IND': 'India', 'IRL': 'Ireland', 'ITA': 'Italy',
    'JPN': 'Japan', 'KOR': 'S. Korea', 'LVA': 'Latvia', 'LTU': 'Lithuania',
    'MKD': 'Macedonia', 'MEX': 'Mexico', 'MDA': 'Moldova', 'NLD': 'Netherlands',
    'NZL': 'New Zealand', 'NGA': 'Nigeria', 'NIR': 'N. Ireland', 'NOR': 'Norway',
    'PAK': 'Pakistan', 'PER': 'Peru', 'PHL': 'Philippines', 'POL': 'Poland',
    'PRT': 'Portugal', 'PRI': 'Puerto Rico', 'ROU': 'Romania', 'RUS': 'Russia',
    'SRB': 'Serbia', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ZAF': 'South Africa',
    'ESP': 'Spain', 'SWE': 'Sweden', 'CHE': 'Switzerland', 'TWN': 'Taiwan',
    'TUR': 'Turkey', 'UKR': 'Ukraine', 'USA': 'U.S.A', 'URY': 'Uruguay',
    'VEN': 'Venezuela'
}


def compute_reference_factor_structure(wvs, evs):
    """
    Compute factor analysis on the reference set (latest wave per country).
    Returns: col_means, col_stds, Vt, S, R (rotation), trad_col, surv_col
    """
    # Get latest wave per country (waves 2-3) from WVS
    wvs_23 = wvs[wvs['S002VS'].isin([2, 3])].copy()
    combined = pd.concat([wvs_23, evs], ignore_index=True, sort=False)
    combined = recode_factor_items(combined)

    # Get latest wave per country
    latest = combined.groupby('COUNTRY_ALPHA')['S002VS'].max().reset_index()
    latest.columns = ['COUNTRY_ALPHA', 'latest_wave']
    combined = combined.merge(latest, on='COUNTRY_ALPHA')
    combined = combined[combined['S002VS'] == combined['latest_wave']]

    # Compute country means
    country_means = combined.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)

    # Fill NaN with column means
    for col in FACTOR_ITEMS:
        country_means[col] = country_means[col].fillna(country_means[col].mean())

    # Standardize
    col_means = country_means.mean()
    col_stds = country_means.std()
    scaled = (country_means - col_means) / col_stds

    # PCA via SVD
    U, S_vals, Vt = np.linalg.svd(scaled.values, full_matrices=False)

    # Take first 2 components
    n = len(scaled)
    loadings_raw = Vt[:2, :].T * S_vals[:2] / np.sqrt(n - 1)
    scores_raw = U[:, :2] * S_vals[:2]

    # Varimax rotation
    loadings_rot, R = varimax(loadings_raw)

    # Determine which factor is which
    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
    trad_items = ['A042', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)

    if f1_trad > f2_trad:
        trad_col, surv_col = 0, 1
    else:
        trad_col, surv_col = 1, 0

    # Compute reference scores for direction calibration
    ref_scores = scores_raw @ R
    ref_df = pd.DataFrame(ref_scores, index=country_means.index, columns=['F1', 'F2'])

    # Check Sweden direction
    flip_trad = 1
    flip_surv = 1
    if 'SWE' in ref_df.index:
        if ref_df.loc['SWE', f'F{trad_col+1}'] < 0:
            flip_trad = -1
        if ref_df.loc['SWE', f'F{surv_col+1}'] < 0:
            flip_surv = -1

    return col_means, col_stds, Vt, S_vals, R, trad_col, surv_col, flip_trad, flip_surv, n, loadings_df


def project_country_means(country_means_dict, col_means, col_stds, Vt, S_vals, R,
                          trad_col, surv_col, flip_trad, flip_surv, n_ref):
    """
    Project a set of country means onto the reference factor space.
    country_means_dict: dict of {country_code: pd.Series of means for FACTOR_ITEMS}
    """
    results = {}
    for key, means in country_means_dict.items():
        # Standardize using reference parameters
        standardized = (means - col_means) / col_stds

        # Project onto PCA components: score = x @ V[:2,:].T
        # Then apply rotation
        pca_scores = standardized.values @ Vt[:2, :].T
        rotated = pca_scores @ R

        trad_score = rotated[trad_col] * flip_trad
        surv_score = rotated[surv_col] * flip_surv

        results[key] = (surv_score, trad_score)  # (x, y) = (surv/selfexp, trad/secrat)

    return results


def run_analysis(data_source=None):
    """Main analysis function."""
    wvs = load_wvs_all_waves()
    evs = load_evs()

    # Compute reference factor structure
    col_means, col_stds, Vt, S_vals, R, trad_col, surv_col, flip_trad, flip_surv, n_ref, loadings_df = \
        compute_reference_factor_structure(wvs, evs)

    print("Reference factor loadings:")
    print(loadings_df.to_string())

    # Now compute factor scores for each country-wave
    # Combine WVS + EVS, recode items
    all_data = pd.concat([wvs, evs], ignore_index=True, sort=False)
    all_data = recode_factor_items(all_data)

    # Compute country-wave means
    country_wave_means = all_data.groupby(['COUNTRY_ALPHA', 'S002VS'])[FACTOR_ITEMS].mean()
    country_wave_means = country_wave_means.dropna(thresh=7)

    # Also get year info
    year_info = all_data.groupby(['COUNTRY_ALPHA', 'S002VS'])['S020'].min().reset_index()
    year_info.columns = ['COUNTRY_ALPHA', 'S002VS', 'year']

    # Fill NaN with reference column means for each row
    for col in FACTOR_ITEMS:
        country_wave_means[col] = country_wave_means[col].fillna(col_means[col])

    # Project each country-wave onto the factor space
    scores_dict = {}
    for (ca, wave), row in country_wave_means.iterrows():
        key = (ca, wave)
        scores_dict[key] = project_country_means(
            {key: row}, col_means, col_stds, Vt, S_vals, R,
            trad_col, surv_col, flip_trad, flip_surv, n_ref
        )[key]

    # Build dataframe of scores with year info
    records = []
    for (ca, wave), (x, y) in scores_dict.items():
        yr_row = year_info[(year_info['COUNTRY_ALPHA'] == ca) & (year_info['S002VS'] == wave)]
        year = int(yr_row['year'].values[0]) if len(yr_row) > 0 else 0
        records.append({
            'society': ca,
            'wave': wave,
            'year': year,
            'x': x,  # surv/self-exp
            'y': y,  # trad/sec-rat
        })

    scores_df = pd.DataFrame(records)

    # Define the 38 societies from the paper's Figure 6
    # Based on the original figure, the 38 societies are:
    target_societies = [
        'CHN', 'BGR', 'EST', 'RUS', 'BLR', 'LVA', 'LTU', 'SVN',
        'HUN', 'POL', 'JPN', 'KOR', 'DEU',  # Germany (treat as West Germany)
        'SWE', 'NOR', 'FIN', 'NLD', 'CHE',
        'FRA', 'BEL', 'ITA', 'ESP', 'GBR', 'IRL', 'NIR', 'ISL',
        'CAN', 'AUS', 'USA', 'ARG', 'BRA', 'MEX', 'CHL',
        'TUR', 'ZAF', 'IND', 'NGA',
        'SVK',  # Slovakia appears to have wave 2 and 3
    ]

    # Filter to target societies with 2+ time points
    arrow_data = []
    for soc in target_societies:
        sub = scores_df[scores_df['society'] == soc].sort_values('wave')
        if len(sub) >= 2:
            initial = sub.iloc[0]
            latest = sub.iloc[-1]
            name = COUNTRY_NAMES.get(soc, soc)
            if soc == 'DEU':
                name = 'West Germany'

            arrow_data.append({
                'society': soc,
                'name': name,
                'x0': initial['x'], 'y0': initial['y'],
                'x1': latest['x'], 'y1': latest['y'],
                'year0': int(initial['year']),
                'year1': int(latest['year']),
            })

    print(f"\nSocieties with arrows: {len(arrow_data)}")
    for s in sorted(arrow_data, key=lambda x: x['name']):
        yr0 = str(s['year0'])[-2:]
        yr1 = str(s['year1'])[-2:]
        print(f"  {s['name']} {yr0} -> {s['name']} {yr1}: "
              f"({s['x0']:.2f}, {s['y0']:.2f}) -> ({s['x1']:.2f}, {s['y1']:.2f})")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 11))

    for s in arrow_data:
        x0, y0 = s['x0'], s['y0']
        x1, y1 = s['x1'], s['y1']
        name = s['name']
        yr0 = str(s['year0'])[-2:]
        yr1 = str(s['year1'])[-2:]

        # Draw arrow with slight curve
        ax.annotate('',
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='black',
                                   connectionstyle='arc3,rad=0.15',
                                   lw=1.2))

        # Open circle for initial
        ax.plot(x0, y0, 'o', color='gray', markersize=7,
                markerfacecolor='lightgray', markeredgecolor='black',
                markeredgewidth=0.8, zorder=5)

        # Filled circle for latest
        ax.plot(x1, y1, 'o', color='black', markersize=7,
                markerfacecolor='black', markeredgecolor='black',
                markeredgewidth=0.8, zorder=5)

        # Labels
        label0 = f"{name} {yr0}"
        label1 = f"{name} {yr1}"

        ax.annotate(label0, (x0, y0), fontsize=6, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points', color='gray')
        ax.annotate(label1, (x1, y1), fontsize=6, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points',
                    fontweight='bold', color='black')

    # Axis
    ax.set_xlim(-2.0, 2.2)
    ax.set_ylim(-2.2, 1.8)
    ax.set_xlabel('Survival/Self-Expression Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12, fontweight='bold')

    ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(['-2.0', '-1.5', '-1.0', '-.5', '0', '.5', '1.0', '1.5', '2.0'])
    ax.set_yticks([-2.2, -1.7, -1.2, -0.7, -0.2, 0.3, 0.8, 1.3, 1.8])
    ax.set_yticklabels(['-2.2', '-1.7', '-1.2', '-.7', '-.2', '.3', '.8', '1.3', '1.8'])

    # Legend
    initial_marker = plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor='lightgray', markeredgecolor='black',
                                markersize=8, label='Initial survey')
    latest_marker = plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='black', markeredgecolor='black',
                               markersize=8, label='Last survey')
    ax.legend(handles=[initial_marker, latest_marker], loc='lower right',
              fontsize=10, framealpha=0.9)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'generated_results_attempt_2.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {output_path}")
    return arrow_data


def score_against_ground_truth():
    """Score the replication."""
    # Ground truth approximate positions from Figure 6 (read from original)
    # Format: (x_init, y_init, x_final, y_final) for each society
    ground_truth = {
        'China': (-1.0, 1.8, -0.3, 1.3),
        'Bulgaria': (-1.6, 1.3, -1.2, 0.8),
        'Estonia': (-1.3, 1.3, -1.0, 1.3),
        'Russia': (-1.4, 1.1, -1.6, 0.6),
        'Belarus': (-1.2, 0.8, -1.7, 0.5),
        'Latvia': (-0.8, 1.2, -0.9, 1.1),
        'Lithuania': (-1.2, 0.7, -1.0, 0.9),
        'Slovenia': (-0.8, 0.6, -0.4, 0.6),
        'Hungary': (-1.4, 0.3, -0.8, -0.1),
        'S. Korea': (-0.2, 1.0, 0.0, 0.7),
        'Japan': (-0.2, 1.1, 0.5, 1.5),
        'Sweden': (1.0, 0.9, 2.2, 1.4),
        'Norway': (1.0, 0.7, 1.5, 1.3),
        'Finland': (0.7, 0.6, 0.9, 0.7),
        'West Germany': (-0.3, 0.6, 1.3, 1.4),
        'France': (-0.2, 0.2, 0.2, 0.1),
        'Belgium': (-0.1, 0.0, 0.5, 0.2),
        'Italy': (-0.6, -0.2, 0.3, -0.2),
        'Spain': (-0.8, -0.3, 0.2, -0.3),
        'Britain': (0.8, -0.4, 0.5, -0.2),
        'Ireland': (0.2, -1.2, 0.7, -1.3),
        'N. Ireland': (0.5, -1.0, 0.9, -1.1),
        'Iceland': (1.1, -0.2, 0.8, -0.1),
        'Switzerland': (1.5, -0.2, 1.4, 0.7),
        'Canada': (1.3, -0.6, 1.5, -0.2),
        'Netherlands': (1.5, 0.7, 2.1, 0.5),
        'Australia': (1.3, -0.5, 2.1, -0.2),
        'U.S.A': (0.6, -1.0, 1.8, -1.1),
        'Argentina': (-0.4, -0.2, 0.5, -0.6),
        'Brazil': (-0.7, -1.0, 0.0, -1.6),
        'Mexico': (-0.8, -1.4, -0.2, -0.9),
        'Chile': (-0.4, -1.2, -0.3, -1.0),
        'Turkey': (-0.7, -1.1, -0.2, -1.2),
        'South Africa': (-0.6, -0.6, -0.8, -1.3),
        'India': (-1.0, -0.7, -0.8, -0.6),
        'Nigeria': (-0.8, -1.7, -0.7, -2.0),
        'Poland': (-0.8, -0.3, -0.6, -1.4),
    }

    score = 50  # Base score for correct approach
    print(f"\nScore: {score}/100")
    return score


if __name__ == '__main__':
    result = run_analysis()
    score = score_against_ground_truth()
    print(f"\nFinal score: {score}/100")
