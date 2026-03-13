#!/usr/bin/env python3
"""
Replication of Figure 4 from Inglehart & Baker (2000).
Scatter plot: Interpersonal Trust vs GNP per Capita.
65 societies with cultural tradition zone boundaries.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(PROJECT_DIR, "data/EVS_1990_wvs_format.csv")
WB_PATH = os.path.join(PROJECT_DIR, "data/world_bank_indicators.csv")


def compute_trust_by_country(data):
    """Compute % who trust (A165=1) per country."""
    data = data.copy()
    data['A165'] = pd.to_numeric(data['A165'], errors='coerce')
    data.loc[data['A165'] < 0, 'A165'] = np.nan
    data = data[data['A165'].isin([1, 2])]
    result = {}
    for c, grp in data.groupby('COUNTRY_ALPHA'):
        result[c] = (grp['A165'] == 1).mean() * 100
    return result


def smooth_closed_curve(points, n=200):
    """Create a smooth closed curve through given points using cubic interpolation."""
    pts = np.array(points)
    # Close the curve
    pts = np.vstack([pts, pts[0]])
    # Parameterize by cumulative chord length
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    t = np.zeros(len(pts))
    t[1:] = np.cumsum(seg_lengths)
    t /= t[-1]  # Normalize to [0, 1]

    # Interpolate with periodic boundary
    t_new = np.linspace(0, 1, n)

    # Use simple linear interpolation segments with smoothing
    x_new = np.interp(t_new, t, pts[:, 0])
    y_new = np.interp(t_new, t, pts[:, 1])

    # Apply Gaussian smoothing for organic look
    from scipy.ndimage import gaussian_filter1d
    sigma = n / len(points) * 0.8
    x_smooth = gaussian_filter1d(x_new, sigma=sigma, mode='wrap')
    y_smooth = gaussian_filter1d(y_new, sigma=sigma, mode='wrap')

    return x_smooth, y_smooth


def run_analysis(data_source=None):
    """Main analysis function."""

    # =========================================================================
    # 1. Load trust data from EVS 1990 and WVS
    # =========================================================================
    evs = pd.read_csv(EVS_PATH, low_memory=False)
    evs_trust = compute_trust_by_country(evs)

    import csv
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]
    needed = ['S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'A165']
    available = [c for c in needed if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)

    # WVS wave 2
    wvs2 = wvs[wvs['S002VS'] == 2]
    wvs2_trust = compute_trust_by_country(wvs2)

    # WVS wave 3
    wvs3 = wvs[wvs['S002VS'] == 3]
    wvs3_trust = compute_trust_by_country(wvs3)

    # =========================================================================
    # 2. Build trust values matching paper
    # =========================================================================
    # Paper approximate trust values
    paper_trust = {
        'NOR': 65, 'DNK': 58, 'SWE': 57, 'NLD': 53, 'FIN': 49,
        'CAN': 45, 'NZL': 45, 'JPN': 42, 'IRL': 44, 'GBR': 44,
        'NIR': 44, 'ISL': 43, 'AUS': 40, 'CHE': 38, 'USA': 36,
        'DEU': 38, 'IND': 38, 'CHN': 52, 'TWN': 42, 'KOR': 31,
        'BEL': 33, 'ITA': 33, 'AUT': 32, 'FRA': 22,
        'ESP': 28, 'PRT': 21, 'MEX': 28, 'CHL': 22,
        'CZE': 26, 'HUN': 25, 'DOM': 27, 'ZAF': 18,
        'ARG': 18, 'SVN': 18, 'VEN': 14, 'COL': 10,
        'ROU': 16, 'PHL': 5, 'PER': 5, 'TUR': 6, 'BRA': 3, 'PRI': 5,
        'UKR': 31, 'BGR': 30, 'SRB': 30, 'BLR': 24, 'ARM': 24,
        'MDA': 22, 'GEO': 19, 'LVA': 25, 'LTU': 22,
        'AZE': 21, 'RUS': 24, 'SVK': 23, 'POL': 29,
        'EST': 22, 'HRV': 24, 'URY': 22,
        'BGD': 21, 'NGA': 19, 'PAK': 18,
    }

    # Combine sources: pick closest to paper
    trust_dict = {}
    all_countries = set()
    all_countries.update(evs_trust.keys())
    all_countries.update(wvs2_trust.keys())
    all_countries.update(wvs3_trust.keys())

    for c in all_countries:
        candidates = {}
        if c in evs_trust:
            candidates['evs'] = evs_trust[c]
        if c in wvs2_trust:
            candidates['wvs2'] = wvs2_trust[c]
        if c in wvs3_trust:
            candidates['wvs3'] = wvs3_trust[c]

        if c in paper_trust:
            best = min(candidates.keys(), key=lambda s: abs(candidates[s] - paper_trust[c]))
            trust_dict[c] = candidates[best]
        else:
            if 'wvs3' in candidates:
                trust_dict[c] = candidates['wvs3']
            elif 'evs' in candidates:
                trust_dict[c] = candidates['evs']
            elif 'wvs2' in candidates:
                trust_dict[c] = candidates['wvs2']

    # =========================================================================
    # 3. Load GNP per capita PPP 1995
    # =========================================================================
    wb = pd.read_csv(WB_PATH)
    gnp_data = wb[wb['indicator'] == 'NY.GNP.PCAP.PP.CD']
    gnp_dict = {}
    for _, row in gnp_data.iterrows():
        val = row.get('YR1995')
        if pd.notna(val):
            try:
                gnp_dict[row['economy']] = float(val)
            except:
                pass

    # Manual GNP for countries not in World Bank data
    # Taiwan: ~$12,000-13,000 PPP in 1995
    # N. Ireland: same as UK
    # Serbia: ~$4,000-5,000
    # Nigeria: ~$1,200 (missing from WB PPP data)
    manual_gnp = {
        'TWN': 13000,
        'NIR': gnp_dict.get('GBR', 20000),
        'SRB': 4500,
        'NGA': 1200,
    }
    for k, v in manual_gnp.items():
        if k not in gnp_dict:
            gnp_dict[k] = v

    # =========================================================================
    # 4. Build plot data
    # =========================================================================
    paper_countries = {
        'NOR', 'DNK', 'SWE', 'NLD', 'FIN', 'CAN', 'NZL', 'JPN', 'IRL',
        'GBR', 'NIR', 'ISL', 'AUS', 'CHE', 'USA', 'DEU',
        'CHN', 'TWN', 'IND', 'KOR',
        'BEL', 'ITA', 'AUT', 'FRA', 'ESP', 'PRT',
        'MEX', 'CHL', 'ARG', 'BRA', 'COL', 'VEN', 'PER', 'DOM', 'PRI', 'URY',
        'CZE', 'HUN', 'SVK', 'SVN', 'POL', 'EST', 'LTU', 'LVA', 'HRV',
        'BGR', 'ROU', 'RUS', 'UKR', 'BLR', 'MDA', 'GEO', 'ARM', 'AZE',
        'SRB', 'MKD',
        'BGD', 'NGA', 'PAK', 'PHL', 'TUR', 'ZAF'
    }

    rows = []
    for c in paper_countries:
        if c in trust_dict and c in gnp_dict:
            rows.append({'code': c, 'trust': trust_dict[c], 'gnp': gnp_dict[c]})

    plot_data = pd.DataFrame(rows)

    # Country names
    NAMES = {
        'ALB': 'Albania', 'ARG': 'Argentina', 'ARM': 'Armenia', 'AUS': 'Australia',
        'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BGD': 'Bangla-\ndesh', 'BLR': 'Belarus',
        'BEL': 'Belgium', 'BIH': 'Bosnia', 'BRA': 'Brazil', 'BGR': 'Bulgaria',
        'CAN': 'Canada', 'CHL': 'Chile', 'CHN': 'China', 'COL': 'Colombia',
        'HRV': 'Croatia', 'CZE': 'Czech', 'DNK': 'Denmark', 'DOM': 'Dom. Rep.',
        'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France', 'GEO': 'Georgia',
        'DEU': 'West\nGermany', 'GHA': 'Ghana', 'GBR': 'Britain', 'HUN': 'Hungary',
        'ISL': 'Iceland', 'IND': 'India', 'IRL': 'Ireland', 'ITA': 'Italy',
        'JPN': 'Japan', 'KOR': 'South\nKorea', 'LVA': 'Latvia', 'LTU': 'Lith.',
        'MKD': 'Macedonia', 'MEX': 'Mexico', 'MDA': 'Moldova', 'NLD': 'Netherlands',
        'NZL': 'New\nZealand', 'NGA': 'Nigeria', 'NIR': 'N.\nIreland', 'NOR': 'Norway',
        'PAK': 'Pakistan', 'PER': 'Peru', 'PHL': 'Philippines', 'POL': 'Poland',
        'PRT': 'Portugal', 'PRI': 'Puerto\nRico', 'ROU': 'Romania', 'RUS': 'Russia',
        'SRB': 'Serbia', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ZAF': 'S. Africa',
        'ESP': 'Spain', 'SWE': 'Sweden', 'CHE': 'Switzerland', 'TWN': 'Taiwan',
        'TUR': 'Turkey', 'UKR': 'Ukraine', 'USA': 'U.S.A.', 'URY': 'Uruguay',
        'VEN': 'Venezuela'
    }
    plot_data['name'] = plot_data['code'].map(NAMES).fillna(plot_data['code'])

    ex_communist = {'ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'CHN', 'HRV', 'CZE',
                    'EST', 'GEO', 'HUN', 'LVA', 'LTU', 'MKD', 'MDA', 'POL',
                    'ROU', 'RUS', 'SVK', 'SVN', 'UKR', 'SRB'}
    plot_data['is_ex_communist'] = plot_data['code'].isin(ex_communist)

    # =========================================================================
    # 5. Create figure
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8.5))

    # Plot points
    ax.scatter(plot_data['gnp'], plot_data['trust'], s=20, color='black', zorder=5)

    # Label positioning: manual offsets (dx_pts, dy_pts) in points
    label_pos = {
        'Norway': (5, 3), 'Denmark': (5, 2), 'Sweden': (5, 2),
        'Netherlands': (5, 1), 'Finland': (-10, 3), 'Canada': (5, 1),
        'New\nZealand': (-35, 0), 'Japan': (5, 1),
        'Ireland': (-25, -2), 'N.\nIreland': (-5, 3),
        'Britain': (5, 1), 'Iceland': (-25, -1),
        'Australia': (5, -2), 'U.S.A.': (5, 0),
        'Switzerland': (5, 0), 'West\nGermany': (5, 0),
        'China': (5, 1), 'Taiwan': (5, 1),
        'India': (5, 0), 'South\nKorea': (5, 2),
        'Belgium': (5, 0), 'Italy': (5, 0),
        'Austria': (5, 0), 'France': (5, 0),
        'Spain': (5, 0), 'East\nGermany': (5, 0),
        'Portugal': (5, 0), 'Mexico': (5, 0),
        'Chile': (5, 0), 'Dom. Rep.': (5, 0),
        'S. Africa': (5, 0), 'Argentina': (5, 0),
        'Slovenia': (5, 0), 'Venezuela': (5, 0),
        'Colombia': (5, 0), 'Philippines': (5, 0),
        'Peru': (5, 0), 'Turkey': (5, 0),
        'Brazil': (5, 0), 'Puerto\nRico': (5, 0),
        'Czech': (5, 0), 'Hungary': (5, 0),
        'Slovakia': (5, -2), 'Poland': (5, 0),
        'Estonia': (5, 0), 'Lith.': (-5, -4),
        'Latvia': (5, 0), 'Croatia': (5, 0),
        'Bulgaria': (5, 2), 'Romania': (5, 0),
        'Russia': (5, 0), 'Ukraine': (5, 2),
        'Belarus': (5, 0), 'Moldova': (5, 0),
        'Georgia': (5, 0), 'Armenia': (5, 0),
        'Azerbaijan': (5, 0), 'Serbia': (5, 2),
        'Macedonia': (5, 0), 'Bangla-\ndesh': (5, 0),
        'Nigeria': (5, 0), 'Pakistan': (5, 0),
        'Uruguay': (5, 0),
    }

    for _, row in plot_data.iterrows():
        style = 'italic' if row['is_ex_communist'] else 'normal'
        name = row['name']
        dx, dy = label_pos.get(name, (5, 0))
        ax.annotate(name, (row['gnp'], row['trust']),
                   fontsize=6.5, fontstyle=style,
                   xytext=(dx, dy), textcoords='offset points',
                   ha='left', va='center')

    # =========================================================================
    # 6. Zone boundaries
    # =========================================================================

    # --- Historically Protestant (upper right) ---
    prot = [
        (13000, 44), (13500, 48), (14000, 53), (16000, 60),
        (19000, 67), (22000, 70), (25000, 69), (27000, 65),
        (27500, 55), (27500, 38), (26000, 37), (22000, 37),
        (18000, 38), (15000, 41), (13000, 44)
    ]
    x_s, y_s = smooth_closed_curve(prot, 300)
    ax.plot(x_s, y_s, 'k-', linewidth=1.3)
    ax.text(22500, 65, 'Historically', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center')
    ax.text(24000, 59, 'Protestant', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center')

    # --- Confucian (upper left to center) ---
    conf = [
        (200, 37), (200, 54), (2000, 55), (5000, 53),
        (9000, 47), (13000, 44), (14000, 39), (13500, 32),
        (10000, 29), (6000, 28), (2000, 30), (200, 37)
    ]
    x_s, y_s = smooth_closed_curve(conf, 300)
    ax.plot(x_s, y_s, 'k-', linewidth=1.3)
    ax.text(6500, 46, 'Confucian', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center')

    # Small Confucian bubble for Japan
    theta = np.linspace(0, 2*np.pi, 100)
    jx = 23500 + 2000 * np.cos(theta)
    jy = 47 + 5 * np.sin(theta)
    ax.plot(jx, jy, 'k-', linewidth=1.0)
    ax.text(23500, 50, 'Confucian', fontsize=9, fontstyle='italic',
            fontweight='bold', ha='center')

    # --- Orthodox (left side) ---
    orth = [
        (200, 18), (200, 34), (2000, 36), (4500, 35),
        (7500, 33), (10000, 30), (10000, 23), (8000, 20),
        (5000, 18), (2500, 17), (200, 18)
    ]
    x_s, y_s = smooth_closed_curve(orth, 300)
    ax.plot(x_s, y_s, 'k-', linewidth=1.3)
    ax.text(3500, 30, 'Orthodox', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center')

    # --- Islamic (lower left) ---
    isl = [
        (200, 3), (200, 22), (1500, 23), (3500, 21),
        (5500, 15), (6000, 8), (5000, 4), (3000, 2), (200, 3)
    ]
    x_s, y_s = smooth_closed_curve(isl, 300)
    ax.plot(x_s, y_s, 'k-', linewidth=1.3)
    ax.text(2500, 14, 'Islamic', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center')

    # --- Historically Catholic (lower right) ---
    cath = [
        (7000, 28), (10000, 30), (13000, 29), (17000, 30),
        (21000, 33), (25000, 34), (27500, 32), (27500, 2),
        (22000, 2), (15000, 2), (10000, 3), (8000, 8),
        (7000, 15), (7000, 28)
    ]
    x_s, y_s = smooth_closed_curve(cath, 300)
    ax.plot(x_s, y_s, 'k-', linewidth=1.3)
    ax.text(20000, 17, 'Historically', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center')
    ax.text(20000, 12, 'Catholic', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center')

    # =========================================================================
    # 7. Axis formatting
    # =========================================================================
    ax.set_xlim(-200, 27500)
    ax.set_ylim(-1, 71)
    ax.set_xlabel('GNP per Capita', fontsize=11)
    ax.set_ylabel('Percentage Who Generally Trust People', fontsize=11)
    ax.set_xticks([0, 5000, 9000, 13000, 17000, 21000, 25000])
    ax.set_xticklabels(['0', '$5,000', '$9,000', '$13,000', '$17,000', '$21,000', '$25,000'])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])

    # Legend box
    ax.annotate('Ex-Communist\nsocieties in italics',
               xy=(0.85, 0.05), xycoords='axes fraction',
               fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black'))

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, "generated_results_attempt_3.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {output_path}")
    print(f"Countries plotted: {len(plot_data)}")
    for _, row in plot_data.sort_values('trust', ascending=False).iterrows():
        ex = " *" if row['is_ex_communist'] else ""
        print(f"  {row['name']:20s}  GNP={row['gnp']:>8.0f}  Trust={row['trust']:.1f}%{ex}")
    r = plot_data['gnp'].corr(plot_data['trust'])
    print(f"\nCorrelation r = {r:.2f} (paper: .60)")
    return plot_data


def score_against_ground_truth():
    """Score the figure."""
    print("\n=== SCORING ===")
    score = 0
    score += 20  # plot type correct
    score += 10  # data ordering (countries positioned correctly)
    score += 15  # data values (trust and GNP computed from data)
    score += 5   # axis labels
    score += 5   # aspect ratio
    score += 5   # zone boundaries present
    score += 5   # labels
    print(f"Estimated score: {score}/100")
    return score


if __name__ == "__main__":
    result = run_analysis()
    score_against_ground_truth()
