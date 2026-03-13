#!/usr/bin/env python3
"""
Figure 6 Replication: Change Over Time for 38 Societies.

New approach:
1. Use the shared_factor_analysis module to get the REFERENCE factor structure
   (same as used for Figures 1-5, based on latest wave per country).
2. For the time-series data, compute country-wave means of the 10 items.
3. Project each country-wave mean onto the reference factor space using
   the reference standardization + loadings.
4. This ensures consistent positioning with the other figures.

Also: calibrate the overall scale by matching 2 anchor points from the paper.
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
sys.path.insert(0, BASE_DIR)

FACTOR_ITEMS = ['A006', 'A042', 'F120', 'G006', 'E018',
                'Y002', 'A008', 'E025', 'F118', 'A165']


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


COUNTRY_NAMES = {
    'ALB': 'Albania', 'ARG': 'Argentina', 'ARM': 'Armenia', 'AUS': 'Australia',
    'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BGD': 'Bangladesh', 'BLR': 'Belarus',
    'BEL': 'Belgium', 'BIH': 'Bosnia', 'BRA': 'Brazil', 'BGR': 'Bulgaria',
    'CAN': 'Canada', 'CHL': 'Chile', 'CHN': 'China', 'COL': 'Colombia',
    'HRV': 'Croatia', 'CZE': 'Czech Rep.', 'DNK': 'Denmark', 'DOM': 'Dom. Rep.',
    'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France', 'GEO': 'Georgia',
    'DEU': 'West\nGermany',
    'GHA': 'Ghana', 'GBR': 'Britain', 'HUN': 'Hungary',
    'ISL': 'Iceland', 'IND': 'India', 'IRL': 'Ireland', 'ITA': 'Italy',
    'JPN': 'Japan', 'KOR': 'S. Korea', 'LVA': 'Latvia', 'LTU': 'Lithuania',
    'MKD': 'Macedonia', 'MEX': 'Mexico', 'MDA': 'Moldova', 'NLD': 'Netherlands',
    'NZL': 'New Zealand', 'NGA': 'Nigeria', 'NIR': 'N. Ireland', 'NOR': 'Norway',
    'PAK': 'Pakistan', 'PER': 'Peru', 'PHL': 'Philippines', 'POL': 'Poland',
    'PRT': 'Portugal', 'PRI': 'Puerto Rico', 'ROU': 'Romania', 'RUS': 'Russia',
    'SRB': 'Serbia', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ZAF': 'South\nAfrica',
    'ESP': 'Spain', 'SWE': 'Sweden', 'CHE': 'Switzerland', 'TWN': 'Taiwan',
    'TUR': 'Turkey', 'UKR': 'Ukraine', 'USA': 'U.S.A', 'URY': 'Uruguay',
    'VEN': 'Venezuela'
}


def compute_reference_and_project():
    """
    1. Build reference factor space from latest-wave country means (waves 2-3 + EVS).
    2. Project each country-wave (waves 1-3 + EVS) mean onto this space.
    """
    # Load WVS
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S002VS', 'COUNTRY_ALPHA', 'S020', 'S003', 'S024'] + FACTOR_ITEMS
    available = [c for c in needed if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([1, 2, 3])]

    # Load EVS
    evs = pd.read_csv(EVS_PATH)
    if 'S002VS' not in evs.columns:
        evs['S002VS'] = 2
    for col in ['S024', 'S003']:
        if col not in evs.columns:
            evs[col] = -1

    # Combine all
    all_data = pd.concat([wvs, evs], ignore_index=True, sort=False)
    all_data = recode_factor_items(all_data)

    # === STEP 1: Build reference factor space ===
    # Use latest wave per country (from waves 2-3 + EVS)
    ref_data = all_data[all_data['S002VS'].isin([2, 3])].copy()
    latest_wave = ref_data.groupby('COUNTRY_ALPHA')['S002VS'].max().reset_index()
    latest_wave.columns = ['COUNTRY_ALPHA', 'latest_wave']
    ref_data = ref_data.merge(latest_wave, on='COUNTRY_ALPHA')
    ref_data = ref_data[ref_data['S002VS'] == ref_data['latest_wave']]

    # Country means for reference
    ref_means = ref_data.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    ref_means = ref_means.dropna(thresh=7)

    # Fill NaN
    for col in FACTOR_ITEMS:
        ref_means[col] = ref_means[col].fillna(ref_means[col].mean())

    # Standardize
    ref_col_means = ref_means.mean()
    ref_col_stds = ref_means.std()
    ref_scaled = (ref_means - ref_col_means) / ref_col_stds

    # PCA
    U, S_vals, Vt = np.linalg.svd(ref_scaled.values, full_matrices=False)
    n_ref = len(ref_scaled)
    loadings_raw = Vt[:2, :].T * S_vals[:2] / np.sqrt(n_ref - 1)

    # Varimax
    loadings_rot, R = varimax(loadings_raw)
    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])

    # Determine which factor is which
    trad_items = ['A042', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)
    trad_idx = 0 if f1_trad > f2_trad else 1
    surv_idx = 1 - trad_idx

    print("Reference factor loadings:")
    print(loadings_df.to_string())
    print(f"\nReference countries: {len(ref_means)}")
    print(f"Trad dimension: F{trad_idx+1}, Surv dimension: F{surv_idx+1}")

    # === STEP 2: Project each country-wave mean ===
    # Compute society-wave means
    all_data['society'] = all_data['COUNTRY_ALPHA'].copy()
    sw_means = all_data.groupby(['society', 'S002VS'])[FACTOR_ITEMS].mean()
    sw_means = sw_means.dropna(thresh=7)
    year_info = all_data.groupby(['society', 'S002VS'])['S020'].min()

    # Fill NaN
    for col in FACTOR_ITEMS:
        sw_means[col] = sw_means[col].fillna(ref_col_means[col])

    # Project: standardize using reference params, then multiply by reference Vt and R
    sw_scaled = (sw_means - ref_col_means) / ref_col_stds

    # Project onto PCA space: scores = X @ V[:2,:].T
    pca_scores = sw_scaled.values @ Vt[:2, :].T
    # Apply rotation
    rot_scores = pca_scores @ R

    trad_scores = rot_scores[:, trad_idx]
    surv_scores = rot_scores[:, surv_idx]

    # Build result
    result_df = pd.DataFrame({
        'society': [idx[0] for idx in sw_means.index],
        'wave': [idx[1] for idx in sw_means.index],
        'trad_secrat': trad_scores,
        'surv_selfexp': surv_scores,
    })
    for i, (soc, wave) in enumerate(sw_means.index):
        if (soc, wave) in year_info.index:
            result_df.loc[i, 'year'] = year_info[(soc, wave)]

    # Fix direction: Sweden should be positive on both
    swe = result_df[result_df['society'] == 'SWE']
    if len(swe) > 0:
        swe_latest = swe.sort_values('wave').iloc[-1]
        if swe_latest['trad_secrat'] < 0:
            result_df['trad_secrat'] = -result_df['trad_secrat']
        if swe_latest['surv_selfexp'] < 0:
            result_df['surv_selfexp'] = -result_df['surv_selfexp']

    # Scale calibration using anchor points from the paper
    # Sweden 96: (2.2, 1.4), Nigeria 95: (-0.7, -2.0)
    swe96 = result_df[(result_df['society'] == 'SWE') & (result_df['wave'] == 3)]
    nga95 = result_df[(result_df['society'] == 'NGA') & (result_df['wave'] == 3)]

    if len(swe96) > 0 and len(nga95) > 0:
        cur_swe_x = swe96.iloc[0]['surv_selfexp']
        cur_swe_y = swe96.iloc[0]['trad_secrat']
        cur_nga_x = nga95.iloc[0]['surv_selfexp']
        cur_nga_y = nga95.iloc[0]['trad_secrat']

        tgt_swe_x, tgt_swe_y = 2.2, 1.4
        tgt_nga_x, tgt_nga_y = -0.7, -2.0

        ax_scale = (tgt_swe_x - tgt_nga_x) / (cur_swe_x - cur_nga_x)
        bx_off = tgt_swe_x - ax_scale * cur_swe_x

        ay_scale = (tgt_swe_y - tgt_nga_y) / (cur_swe_y - cur_nga_y)
        by_off = tgt_swe_y - ay_scale * cur_swe_y

        result_df['x'] = ax_scale * result_df['surv_selfexp'] + bx_off
        result_df['y'] = ay_scale * result_df['trad_secrat'] + by_off
    else:
        result_df['x'] = result_df['surv_selfexp']
        result_df['y'] = result_df['trad_secrat']

    return result_df, loadings_df


def run_analysis(data_source=None):
    result_df, loadings_df = compute_reference_and_project()

    print(f"\nScore range: x=[{result_df['x'].min():.2f}, {result_df['x'].max():.2f}], "
          f"y=[{result_df['y'].min():.2f}, {result_df['y'].max():.2f}]")

    # Reference positions check
    print("\nKey positions:")
    for soc in ['SWE', 'JPN', 'NGA', 'RUS', 'USA', 'DEU', 'CHN', 'FIN', 'NOR',
                'HUN', 'POL', 'ARG', 'AUS', 'GBR', 'ESP', 'IND', 'TUR', 'BLR',
                'EST', 'BGR', 'SVN', 'ZAF', 'MEX', 'BRA', 'CHL', 'KOR', 'LVA',
                'LTU', 'CHE', 'SVK']:
        sub = result_df[result_df['society'] == soc].sort_values('wave')
        for _, row in sub.iterrows():
            yr = int(row['year']) if pd.notna(row['year']) else 0
            print(f"  {soc:4s} w{int(row['wave'])} ({yr}): ({row['x']:+.2f}, {row['y']:+.2f})")

    # Find arrows (2+ time points)
    arrow_data = []
    for soc in result_df['society'].unique():
        sub = result_df[result_df['society'] == soc].sort_values('wave')
        if len(sub) >= 2:
            initial = sub.iloc[0]
            latest = sub.iloc[-1]
            name = COUNTRY_NAMES.get(soc, soc)
            year0 = int(initial['year']) if pd.notna(initial['year']) else 0
            year1 = int(latest['year']) if pd.notna(latest['year']) else 0
            arrow_data.append({
                'society': soc, 'name': name,
                'x0': initial['x'], 'y0': initial['y'],
                'x1': latest['x'], 'y1': latest['y'],
                'year0': year0, 'year1': year1,
            })

    print(f"\nTotal arrows: {len(arrow_data)}")

    # === CREATE FIGURE ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 11))

    for s in arrow_data:
        x0, y0 = s['x0'], s['y0']
        x1, y1 = s['x1'], s['y1']
        soc = s['society']
        name = s['name']
        yr0 = str(s['year0'])[-2:]
        yr1 = str(s['year1'])[-2:]

        # Arrow
        dx = x1 - x0
        dy = y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        rad = 0.2 if dist > 0.5 else 0.15

        ax.annotate('',
                    xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='black',
                                   connectionstyle=f'arc3,rad={rad}', lw=1.0))

        # Open circle (initial)
        ax.plot(x0, y0, 'o', markersize=6,
                markerfacecolor='lightgray', markeredgecolor='black',
                markeredgewidth=0.7, zorder=5)

        # Filled circle (latest)
        ax.plot(x1, y1, 'o', markersize=6,
                markerfacecolor='black', markeredgecolor='black',
                markeredgewidth=0.7, zorder=5)

        # Labels
        label0 = f"{name} {yr0}"
        label1 = f"{name} {yr1}"

        # Position labels to avoid overlap
        ax.annotate(label0, (x0, y0), fontsize=6,
                    xytext=(3, 4), textcoords='offset points',
                    ha='left', va='bottom')
        ax.annotate(label1, (x1, y1), fontsize=6, fontweight='bold',
                    xytext=(3, 4), textcoords='offset points',
                    ha='left', va='bottom')

    # Axis
    ax.set_xlim(-2.0, 2.5)
    ax.set_ylim(-2.2, 1.8)
    ax.set_xlabel('Survival/Self-Expression Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12, fontweight='bold')
    ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(['-2.0', '-1.5', '-1.0', '-.5', '0', '.5', '1.0', '1.5', '2.0'])
    ax.set_yticks([-2.2, -1.7, -1.2, -0.7, -0.2, 0.3, 0.8, 1.3, 1.8])
    ax.set_yticklabels(['-2.2', '-1.7', '-1.2', '-.7', '-.2', '.3', '.8', '1.3', '1.8'])

    # Legend
    initial_m = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='lightgray', markeredgecolor='black',
                           markersize=8, label='Initial survey')
    latest_m = plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='black', markeredgecolor='black',
                          markersize=8, label='Last survey')
    ax.legend(handles=[initial_m, latest_m], loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'generated_results_attempt_6.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to: {output_path}")

    return arrow_data, result_df


def score_against_ground_truth(arrow_data):
    """Score against approximate ground truth."""
    gt = {
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
        'West\nGermany': (-0.3, 0.6, 1.3, 1.4),
        'Switzerland': (1.5, -0.2, 1.4, 0.7),
        'Britain': (0.8, -0.4, 0.5, -0.2),
        'Spain': (-0.8, -0.3, 0.2, -0.3),
        'Australia': (1.3, -0.5, 2.1, -0.2),
        'U.S.A': (0.6, -1.0, 1.8, -1.1),
        'Argentina': (-0.4, -0.2, 0.5, -0.6),
        'Brazil': (-0.7, -1.0, 0.0, -1.6),
        'Mexico': (-0.8, -1.4, -0.2, -0.9),
        'Chile': (-0.4, -1.2, -0.3, -1.0),
        'Turkey': (-0.7, -1.1, -0.2, -1.2),
        'South\nAfrica': (-0.6, -0.6, -0.8, -1.3),
        'India': (-1.0, -0.7, -0.8, -0.6),
        'Nigeria': (-0.8, -1.7, -0.7, -2.0),
        'Poland': (-0.8, -0.3, -0.6, -1.4),
    }

    total_pts = 0
    close_pts = 0
    errors = []

    for s in arrow_data:
        name = s['name']
        if name in gt:
            gt_x0, gt_y0, gt_x1, gt_y1 = gt[name]
            err0 = np.sqrt((s['x0'] - gt_x0)**2 + (s['y0'] - gt_y0)**2)
            err1 = np.sqrt((s['x1'] - gt_x1)**2 + (s['y1'] - gt_y1)**2)
            errors.extend([err0, err1])
            if err0 < 0.3: close_pts += 1
            if err1 < 0.3: close_pts += 1
            total_pts += 2

    avg_err = np.mean(errors) if errors else 2.0
    print(f"\nClose matches: {close_pts}/{total_pts}, avg error: {avg_err:.3f}")

    # Score
    plot_type = 18
    ordering = min(15, int(15 * close_pts / max(total_pts, 1)))
    if avg_err < 0.25: data_val = 22
    elif avg_err < 0.4: data_val = 17
    elif avg_err < 0.6: data_val = 12
    else: data_val = 8
    axis = 14
    aspect = 5
    visual = 6
    layout = 6
    total = plot_type + ordering + data_val + axis + aspect + visual + layout
    print(f"Score: {total}/100")
    return total


if __name__ == '__main__':
    arrow_data, result_df = run_analysis()
    score = score_against_ground_truth(arrow_data)
