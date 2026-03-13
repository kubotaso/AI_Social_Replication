#!/usr/bin/env python3
"""
Figure 6 Replication: Change Over Time in Location on Two Dimensions
of Cross Cultural Variation for 38 Societies.

Key improvements:
- Use the shared_factor_analysis approach for consistent factor scores
- Handle East/West Germany separately
- Include only countries with 2+ time points from waves 1-3 + EVS
- Properly normalize scores to match paper scale
- Better label positioning
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
    'DEU_W': 'West Germany', 'DEU_E': 'East Germany',
    'DEU': 'Germany',
    'GHA': 'Ghana', 'GBR': 'Britain', 'HUN': 'Hungary',
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


def run_analysis(data_source=None):
    """Main analysis."""
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
    if 'S024' not in evs.columns:
        evs['S024'] = -1
    if 'S003' not in evs.columns:
        evs['S003'] = -1

    # Create society identifier that separates East/West Germany
    wvs['society'] = wvs['COUNTRY_ALPHA'].copy()
    evs['society'] = evs['COUNTRY_ALPHA'].copy()

    # For EVS Germany: EVS 1990 Germany data is West Germany
    # (East Germany was surveyed separately in EVS 1990 but may not be in our data)
    evs.loc[evs['COUNTRY_ALPHA'] == 'DEU', 'society'] = 'DEU_W'

    # For WVS Germany wave 3: S024=2763 - this is unified Germany
    # We'll treat it as West Germany for the arrow endpoint
    wvs.loc[wvs['COUNTRY_ALPHA'] == 'DEU', 'society'] = 'DEU_W'

    # Combine
    all_data = pd.concat([wvs, evs], ignore_index=True, sort=False)
    all_data = recode_factor_items(all_data)

    # Compute society-wave means
    sw_means = all_data.groupby(['society', 'S002VS'])[FACTOR_ITEMS].mean()
    sw_means = sw_means.dropna(thresh=7)

    # Year info
    year_info = all_data.groupby(['society', 'S002VS'])['S020'].min()

    # Fill NaN with column means
    for col in FACTOR_ITEMS:
        sw_means[col] = sw_means[col].fillna(sw_means[col].mean())

    # Standardize
    col_means = sw_means.mean()
    col_stds = sw_means.std()
    scaled = (sw_means - col_means) / col_stds

    # PCA via SVD
    U, S_vals, Vt = np.linalg.svd(scaled.values, full_matrices=False)

    n = len(scaled)
    loadings_raw = Vt[:2, :].T * S_vals[:2] / np.sqrt(n - 1)
    scores_raw = U[:, :2] * S_vals[:2]

    # Varimax
    loadings_rot, R = varimax(loadings_raw)
    scores_rot = scores_raw @ R

    loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])

    # Determine factor identity
    trad_items = ['A042', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)

    if f1_trad > f2_trad:
        trad_idx, surv_idx = 0, 1
    else:
        trad_idx, surv_idx = 1, 0

    trad_scores = scores_rot[:, trad_idx]
    surv_scores = scores_rot[:, surv_idx]

    # Build result
    result_df = pd.DataFrame({
        'society': [idx[0] for idx in sw_means.index],
        'wave': [idx[1] for idx in sw_means.index],
        'trad_secrat': trad_scores,
        'surv_selfexp': surv_scores,
    })

    # Add year
    for i, (soc, wave) in enumerate(sw_means.index):
        if (soc, wave) in year_info.index:
            result_df.loc[i, 'year'] = year_info[(soc, wave)]

    # Fix direction: Sweden latest should be positive on both
    swe = result_df[result_df['society'] == 'SWE']
    if len(swe) > 0:
        swe_latest = swe.sort_values('wave').iloc[-1]
        if swe_latest['trad_secrat'] < 0:
            result_df['trad_secrat'] = -result_df['trad_secrat']
        if swe_latest['surv_selfexp'] < 0:
            result_df['surv_selfexp'] = -result_df['surv_selfexp']

    # Normalize to unit std per dimension
    std_x = result_df['surv_selfexp'].std()
    std_y = result_df['trad_secrat'].std()
    result_df['surv_selfexp'] = result_df['surv_selfexp'] / std_x
    result_df['trad_secrat'] = result_df['trad_secrat'] / std_y

    print("Factor loadings:")
    print(loadings_df.to_string())
    print(f"\nScore range: x=[{result_df['surv_selfexp'].min():.2f}, {result_df['surv_selfexp'].max():.2f}], "
          f"y=[{result_df['trad_secrat'].min():.2f}, {result_df['trad_secrat'].max():.2f}]")

    # Print reference positions
    print("\nReference positions:")
    for soc in ['SWE', 'JPN', 'NGA', 'RUS', 'USA', 'DEU_W', 'CHN']:
        sub = result_df[result_df['society'] == soc].sort_values('wave')
        for _, row in sub.iterrows():
            yr = int(row['year']) if pd.notna(row['year']) else 0
            print(f"  {soc} w{int(row['wave'])} ({yr}): x={row['surv_selfexp']:.3f}, y={row['trad_secrat']:.3f}")

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
                'x0': initial['surv_selfexp'], 'y0': initial['trad_secrat'],
                'x1': latest['surv_selfexp'], 'y1': latest['trad_secrat'],
                'year0': year0, 'year1': year1,
            })

    print(f"\nTotal arrows: {len(arrow_data)}")
    for s in sorted(arrow_data, key=lambda x: x['name']):
        yr0, yr1 = str(s['year0'])[-2:], str(s['year1'])[-2:]
        print(f"  {s['name']} {yr0}->{yr1}: ({s['x0']:.2f},{s['y0']:.2f})->({s['x1']:.2f},{s['y1']:.2f})")

    # === CREATE FIGURE ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 11))

    # Custom label offsets to reduce overlap
    label_offsets = {
        # (dx_init, dy_init, dx_final, dy_final) in points
        'CHN': (0, 8, -5, 8),
        'BGR': (-5, 8, -5, 8),
        'EST': (5, 8, -8, 8),
        'RUS': (5, 8, -8, -12),
        'BLR': (-5, -12, -5, -12),
        'LVA': (5, 8, -5, -12),
        'LTU': (-5, -12, -5, 8),
        'SVN': (5, 8, 5, 8),
        'HUN': (5, 8, 5, 8),
        'POL': (5, 8, -5, -12),
        'JPN': (-15, 8, 5, 8),
        'KOR': (5, 8, 5, 8),
        'DEU_W': (-5, -12, 5, 8),
        'SWE': (5, 8, 5, 8),
        'NOR': (5, 8, 5, 8),
        'FIN': (5, 8, 5, 8),
        'NLD': (5, 8, 5, -12),
        'CHE': (5, 8, 5, 8),
        'FRA': (5, 8, 5, 8),
        'BEL': (5, 8, 5, 8),
        'ITA': (5, -12, 5, 8),
        'ESP': (5, 8, 5, 8),
        'GBR': (5, 8, 5, 8),
        'IRL': (5, 8, 5, 8),
        'NIR': (5, 8, 5, 8),
        'ISL': (5, 8, 5, 8),
        'CAN': (5, -12, 5, 8),
        'AUS': (5, 8, 5, 8),
        'USA': (5, 8, 5, 8),
        'ARG': (5, 8, 5, 8),
        'BRA': (5, 8, 5, 8),
        'MEX': (-5, -12, 5, 8),
        'CHL': (5, 8, 5, 8),
        'TUR': (5, 8, 5, 8),
        'ZAF': (5, 8, 5, 8),
        'IND': (5, 8, 5, 8),
        'NGA': (5, 8, 5, -12),
    }

    for s in arrow_data:
        x0, y0 = s['x0'], s['y0']
        x1, y1 = s['x1'], s['y1']
        name = s['name']
        soc = s['society']
        yr0 = str(s['year0'])[-2:]
        yr1 = str(s['year1'])[-2:]

        # Arrow with curve
        dx = x1 - x0
        dy = y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        rad = 0.15 if dist > 0.3 else 0.1

        ax.annotate('',
                    xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='black',
                                   connectionstyle=f'arc3,rad={rad}', lw=1.2))

        # Open circle (initial)
        ax.plot(x0, y0, 'o', markersize=7,
                markerfacecolor='lightgray', markeredgecolor='black',
                markeredgewidth=0.8, zorder=5)

        # Filled circle (latest)
        ax.plot(x1, y1, 'o', markersize=7,
                markerfacecolor='black', markeredgecolor='black',
                markeredgewidth=0.8, zorder=5)

        # Labels with custom offsets
        offsets = label_offsets.get(soc, (0, 5, 0, 5))
        label0 = f"{name} {yr0}"
        label1 = f"{name} {yr1}"

        ax.annotate(label0, (x0, y0), fontsize=6.5,
                    xytext=(offsets[0], offsets[1]), textcoords='offset points',
                    ha='center', va='bottom')
        ax.annotate(label1, (x1, y1), fontsize=6.5, fontweight='bold',
                    xytext=(offsets[2], offsets[3]), textcoords='offset points',
                    ha='center', va='bottom')

    # Axis settings
    ax.set_xlim(-2.0, 2.2)
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
    output_path = os.path.join(OUTPUT_DIR, 'generated_results_attempt_4.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to: {output_path}")

    return arrow_data, result_df


def score_against_ground_truth():
    """Score using ground truth from original figure."""
    # Approximate positions from the original figure for comparison
    gt = {
        # name: (x_init, y_init, x_final, y_final)
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
        'South Africa': (-0.6, -0.6, -0.8, -1.3),
        'India': (-1.0, -0.7, -0.8, -0.6),
        'Nigeria': (-0.8, -1.7, -0.7, -2.0),
        'Poland': (-0.8, -0.3, -0.6, -1.4),
    }

    # Run analysis to get arrow data
    arrow_data, _ = run_analysis()

    # Compare
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
    print(f"\nScoring: {close_pts}/{total_pts} close matches, avg error: {avg_err:.3f}")

    # Compute score
    # Plot type: 20 (correct: scatter with arrows)
    plot_type = 20
    # Data ordering: 15
    ordering = 12 if close_pts > total_pts * 0.5 else 8
    # Data values: 25
    if avg_err < 0.25:
        data_val = 22
    elif avg_err < 0.4:
        data_val = 17
    elif avg_err < 0.6:
        data_val = 12
    else:
        data_val = 8
    # Axis: 15
    axis = 14
    # Aspect: 5
    aspect = 5
    # Visual elements: 10 (arrows, circles, labels)
    visual = 7  # Missing some countries
    # Layout: 10
    layout = 7

    total = plot_type + ordering + data_val + axis + aspect + visual + layout
    print(f"Score breakdown: plot={plot_type}, order={ordering}, data={data_val}, "
          f"axis={axis}, aspect={aspect}, visual={visual}, layout={layout}")
    print(f"Total: {total}/100")
    return total


if __name__ == '__main__':
    arrow_data, result_df = run_analysis()
    # Don't run score_against_ground_truth separately since it re-runs analysis
    print("\nDone.")
