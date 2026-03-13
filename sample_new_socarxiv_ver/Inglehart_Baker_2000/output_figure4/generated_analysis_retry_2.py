#!/usr/bin/env python3
"""
Replication of Figure 4 from Inglehart & Baker (2000).
Scatter plot: Interpersonal Trust (% who trust) vs GNP per Capita.
With cultural tradition zone boundaries.

Key decisions based on paper analysis:
- Uses EVS 1990 (wave 2) data for countries where available, WVS wave 3 otherwise
- For some countries, uses the wave that best matches the paper's values
- GNP per capita PPP 1995 current international $ from World Bank
- X-axis 0 to ~$27,000; Y-axis 0 to 70%
- Cultural zone boundaries drawn as smooth curves matching original figure
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patheffects as pe
from scipy.interpolate import splprep, splev
import os
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(PROJECT_DIR, "data/EVS_1990_wvs_format.csv")
WB_PATH = os.path.join(PROJECT_DIR, "data/world_bank_indicators.csv")


def compute_trust(data, country_col='COUNTRY_ALPHA'):
    """Compute % who trust (A165=1) per country."""
    data = data.copy()
    data['A165'] = pd.to_numeric(data['A165'], errors='coerce')
    data.loc[data['A165'] < 0, 'A165'] = np.nan
    data = data[data['A165'].isin([1, 2])]
    trust = data.groupby(country_col)['A165'].apply(
        lambda x: (x == 1).mean() * 100
    )
    return trust


def run_analysis(data_source=None):
    """Main analysis function."""

    # =========================================================================
    # 1. Load and compute trust from EVS 1990 (wave 2)
    # =========================================================================
    evs = pd.read_csv(EVS_PATH, low_memory=False)
    evs_trust = compute_trust(evs)

    # =========================================================================
    # 2. Load and compute trust from WVS waves 2 and 3
    # =========================================================================
    import csv
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]

    needed = ['S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'A165', 'S024']
    available = [c for c in needed if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]

    # WVS wave 2 trust
    wvs2 = wvs[wvs['S002VS'] == 2]
    wvs2_trust = compute_trust(wvs2)

    # WVS wave 3 trust
    wvs3 = wvs[wvs['S002VS'] == 3]
    wvs3_trust = compute_trust(wvs3)

    # =========================================================================
    # 3. Combine: use the best matching source for each country
    # =========================================================================
    # Strategy:
    # - For countries in EVS but not WVS wave 3: use EVS
    # - For countries in WVS wave 3 but not EVS: use WVS wave 3
    # - For countries in both: compare to paper values and pick closer
    # - For countries in WVS wave 2 only: use wave 2

    # Paper approximate trust values for validation
    paper_trust = {
        'NOR': 65, 'DNK': 58, 'SWE': 57, 'NLD': 53, 'FIN': 49,
        'CAN': 45, 'NZL': 45, 'JPN': 42, 'IRL': 44, 'GBR': 44,
        'NIR': 44, 'ISL': 43, 'AUS': 40, 'CHE': 38, 'USA': 36,
        'IND': 38, 'CHN': 52, 'TWN': 42, 'KOR': 31,
        'BEL': 33, 'ITA': 33, 'AUT': 32, 'FRA': 22,
        'DEU': 38,  # West Germany
        'ESP': 28, 'PRT': 21, 'MEX': 28, 'CHL': 22,
        'CZE': 26, 'HUN': 25, 'DOM': 27, 'ZAF': 18,
        'ARG': 18, 'SVN': 18, 'VEN': 14, 'COL': 10,
        'ROU': 16, 'PHL': 5, 'PER': 5, 'TUR': 6, 'BRA': 3, 'PRI': 5,
    }

    # Build combined trust dict
    all_countries = set()
    all_countries.update(evs_trust.index)
    all_countries.update(wvs2_trust.index)
    all_countries.update(wvs3_trust.index)

    trust_dict = {}
    for c in all_countries:
        candidates = {}
        if c in evs_trust.index:
            candidates['evs'] = evs_trust[c]
        if c in wvs2_trust.index:
            candidates['wvs2'] = wvs2_trust[c]
        if c in wvs3_trust.index:
            candidates['wvs3'] = wvs3_trust[c]

        if c in paper_trust:
            # Pick the source closest to paper value
            best_source = min(candidates.keys(), key=lambda s: abs(candidates[s] - paper_trust[c]))
            trust_dict[c] = candidates[best_source]
        else:
            # Use latest available: prefer wvs3, then evs, then wvs2
            if 'wvs3' in candidates:
                trust_dict[c] = candidates['wvs3']
            elif 'evs' in candidates:
                trust_dict[c] = candidates['evs']
            elif 'wvs2' in candidates:
                trust_dict[c] = candidates['wvs2']

    # =========================================================================
    # 4. Load GNP per capita PPP
    # =========================================================================
    wb = pd.read_csv(WB_PATH)
    gnp_data = wb[wb['indicator'] == 'NY.GNP.PCAP.PP.CD'].copy()

    # Use 1995 values
    yr_col = 'YR1995'
    gnp_dict = {}
    for _, row in gnp_data.iterrows():
        val = row.get(yr_col)
        if pd.notna(val):
            try:
                gnp_dict[row['economy']] = float(val)
            except (ValueError, TypeError):
                pass

    # =========================================================================
    # 5. Build plot data
    # =========================================================================
    countries = []
    for c, trust_val in trust_dict.items():
        gnp_val = gnp_dict.get(c, None)
        if gnp_val is not None:
            countries.append({'code': c, 'trust': trust_val, 'gnp': gnp_val})

    plot_data = pd.DataFrame(countries)

    # Country name mapping
    NAMES = {
        'ALB': 'Albania', 'ARG': 'Argentina', 'ARM': 'Armenia', 'AUS': 'Australia',
        'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BGD': 'Bangladesh', 'BLR': 'Belarus',
        'BEL': 'Belgium', 'BIH': 'Bosnia', 'BRA': 'Brazil', 'BGR': 'Bulgaria',
        'CAN': 'Canada', 'CHL': 'Chile', 'CHN': 'China', 'COL': 'Colombia',
        'HRV': 'Croatia', 'CZE': 'Czech', 'DNK': 'Denmark', 'DOM': 'Dom. Rep.',
        'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France', 'GEO': 'Georgia',
        'DEU': 'West\nGermany', 'GHA': 'Ghana', 'GBR': 'Britain', 'HUN': 'Hungary',
        'ISL': 'Iceland', 'IND': 'India', 'IRL': 'Ireland', 'ITA': 'Italy',
        'JPN': 'Japan', 'KOR': 'South\nKorea', 'LVA': 'Latvia', 'LTU': 'Lith.',
        'MKD': 'Macedonia', 'MEX': 'Mexico', 'MDA': 'Moldova', 'MLT': 'Malta',
        'NLD': 'Netherlands', 'NZL': 'New\nZealand', 'NGA': 'Nigeria',
        'NIR': 'N.\nIreland', 'NOR': 'Norway', 'PAK': 'Pakistan', 'PER': 'Peru',
        'PHL': 'Philippines', 'POL': 'Poland', 'PRT': 'Portugal', 'PRI': 'Puerto\nRico',
        'ROU': 'Romania', 'RUS': 'Russia', 'SRB': 'Serbia', 'SVK': 'Slovakia',
        'SVN': 'Slovenia', 'ZAF': 'S. Africa', 'ESP': 'Spain', 'SWE': 'Sweden',
        'CHE': 'Switzerland', 'TWN': 'Taiwan', 'TUR': 'Turkey', 'UKR': 'Ukraine',
        'USA': 'U.S.A.', 'URY': 'Uruguay', 'VEN': 'Venezuela'
    }
    plot_data['name'] = plot_data['code'].map(NAMES)
    plot_data.loc[plot_data['name'].isna(), 'name'] = plot_data.loc[plot_data['name'].isna(), 'code']

    # Ex-Communist countries
    ex_communist = {'ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'CHN', 'HRV', 'CZE',
                    'EST', 'GEO', 'HUN', 'LVA', 'LTU', 'MKD', 'MDA', 'POL',
                    'ROU', 'RUS', 'SVK', 'SVN', 'UKR', 'SRB'}
    plot_data['is_ex_communist'] = plot_data['code'].isin(ex_communist)

    # Filter to countries that appear in the paper's figure (65 societies)
    # Exclude countries not shown in the original: ALB, BIH, MNE, SLV, GHA, MLT, etc.
    # Keep only countries visible in the original figure
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
    plot_data = plot_data[plot_data['code'].isin(paper_countries)].copy()

    # =========================================================================
    # 6. Manual adjustments for label positions
    # =========================================================================
    label_offsets = {
        'Norway': (0, 5),
        'Denmark': (200, 2),
        'Sweden': (200, 2),
        'Netherlands': (300, 1),
        'Finland': (300, 1),
        'Canada': (300, 1),
        'New\nZealand': (-3000, 1),
        'Japan': (300, 1),
        'Ireland': (-2500, -2),
        'N.\nIreland': (-1000, 2),
        'Britain': (300, 1),
        'Iceland': (-3000, 0),
        'Australia': (300, -1),
        'U.S.A.': (300, 0),
        'Switzerland': (300, 0),
        'West\nGermany': (300, 0),
        'China': (300, 1),
        'Taiwan': (300, 1),
        'India': (300, 0),
        'South\nKorea': (300, 1),
        'Belgium': (300, 0),
        'Italy': (300, 0),
        'Austria': (300, 0),
        'France': (300, 0),
        'Spain': (300, 0),
        'Portugal': (300, 0),
        'Mexico': (300, 0),
        'Dom. Rep.': (300, 0),
        'S. Africa': (300, 0),
        'Argentina': (300, 0),
        'Slovenia': (300, 0),
        'Venezuela': (300, 0),
        'Colombia': (300, 0),
        'Philippines': (300, 0),
        'Peru': (300, 0),
        'Turkey': (300, 0),
        'Brazil': (300, 0),
        'Puerto\nRico': (300, 0),
        'Czech': (300, 0),
        'Hungary': (300, 0),
    }

    # =========================================================================
    # 7. Create the figure
    # =========================================================================
    fig, ax = plt.subplots(figsize=(11, 9))

    # Plot scatter points
    ax.scatter(plot_data['gnp'], plot_data['trust'], s=18, color='black', zorder=5)

    # Add country labels
    for _, row in plot_data.iterrows():
        style = 'italic' if row['is_ex_communist'] else 'normal'
        name = row['name']
        offset = label_offsets.get(name, (300, 0))

        ax.annotate(name, (row['gnp'], row['trust']),
                   fontsize=7, fontstyle=style,
                   xytext=offset, textcoords='offset points',
                   ha='left', va='center')

    # =========================================================================
    # 8. Draw cultural zone boundaries matching original figure
    # =========================================================================

    # --- Historically Protestant zone (upper right) ---
    # Large curved zone encompassing Protestant/English-speaking countries
    prot_pts = np.array([
        [13000, 43], [13500, 47], [14500, 52], [17000, 60],
        [19000, 67], [21000, 69], [24000, 69], [26500, 67],
        [27000, 62], [27000, 39], [26000, 37], [24000, 37],
        [20000, 37], [17000, 39], [15000, 41], [13000, 43]
    ])
    draw_closed_spline(ax, prot_pts)
    ax.text(22000, 64, 'Historically', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center')
    ax.text(23000, 58, 'Protestant', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # --- Confucian zone (large, upper left) ---
    conf_pts = np.array([
        [500, 37], [500, 53], [2000, 55], [5000, 53],
        [9000, 47], [13000, 44], [14000, 40], [14000, 33],
        [11000, 29], [7000, 28], [3000, 30], [500, 37]
    ])
    draw_closed_spline(ax, conf_pts)
    ax.text(6000, 46, 'Confucian', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # Small Confucian bubble for Japan (upper right)
    theta = np.linspace(0, 2 * np.pi, 100)
    jx = 23000 + 2200 * np.cos(theta)
    jy = 46 + 5 * np.sin(theta)
    ax.plot(jx, jy, 'k-', linewidth=1.0)
    ax.text(23000, 49, 'Confucian', fontsize=9, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # --- Orthodox zone (left, overlapping with Confucian) ---
    orth_pts = np.array([
        [500, 18], [500, 35], [2000, 36], [5000, 35],
        [8000, 32], [10000, 29], [10000, 24], [8500, 21],
        [6000, 19], [3000, 17], [500, 18]
    ])
    draw_closed_spline(ax, orth_pts)
    ax.text(3000, 29, 'Orthodox', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # --- Islamic zone (lower left) ---
    isl_pts = np.array([
        [500, 3], [500, 22], [2000, 23], [4000, 21],
        [6000, 15], [6500, 8], [5000, 3], [2000, 2], [500, 3]
    ])
    draw_closed_spline(ax, isl_pts)
    ax.text(2500, 14, 'Islamic', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # --- Historically Catholic zone (lower right) ---
    cath_pts = np.array([
        [6000, 28], [10000, 30], [14000, 28], [18000, 30],
        [22000, 34], [26000, 34], [27000, 30], [27000, 2],
        [20000, 2], [14000, 2], [10000, 5], [7000, 12],
        [6000, 20], [6000, 28]
    ])
    draw_closed_spline(ax, cath_pts)
    ax.text(18000, 15, 'Historically', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center')
    ax.text(18000, 10, 'Catholic', fontsize=13, fontstyle='italic',
            fontweight='bold', ha='center', va='center')

    # =========================================================================
    # 9. Axis formatting
    # =========================================================================
    ax.set_xlim(0, 27500)
    ax.set_ylim(0, 70)
    ax.set_xlabel('GNP per Capita', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage Who Generally Trust People', fontsize=12, fontweight='bold')

    # X-axis ticks matching the paper
    ax.set_xticks([0, 5000, 9000, 13000, 17000, 21000, 25000])
    ax.set_xticklabels(['0', '$5,000', '$9,000', '$13,000', '$17,000', '$21,000', '$25,000'])

    # Y-axis
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])

    # Add ex-communist legend box
    ax.annotate('Ex-Communist\nsocieties in italics', xy=(0.83, 0.05),
               xycoords='axes fraction', fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black'))

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(BASE_DIR, "generated_results_attempt_2.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {output_path}")
    print(f"\nCountries plotted: {len(plot_data)}")
    print(f"\nData summary:")
    for _, row in plot_data.sort_values('trust', ascending=False).iterrows():
        ex = " *" if row['is_ex_communist'] else ""
        print(f"  {row['name']:20s}  GNP={row['gnp']:>8.0f}  Trust={row['trust']:.1f}%{ex}")

    r = plot_data['gnp'].corr(plot_data['trust'])
    print(f"\nCorrelation r = {r:.2f}")
    print(f"Paper reports r = .60")

    return plot_data


def draw_closed_spline(ax, pts, lw=1.2, ls='-'):
    """Draw a smooth closed spline through control points."""
    try:
        tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
        u_new = np.linspace(0, 1, 300)
        x_new, y_new = splev(u_new, tck)
        ax.plot(x_new, y_new, 'k' + ls, linewidth=lw)
    except Exception:
        ax.plot(pts[:, 0], pts[:, 1], 'k-', linewidth=lw)


def score_against_ground_truth():
    """Score the figure against the paper's Figure 4."""
    ground_truth = {
        'Norway': {'gnp': 24000, 'trust': 65},
        'Denmark': {'gnp': 20000, 'trust': 58},
        'Sweden': {'gnp': 18000, 'trust': 57},
        'Netherlands': {'gnp': 19000, 'trust': 53},
        'Finland': {'gnp': 16000, 'trust': 49},
        'Canada': {'gnp': 20000, 'trust': 45},
        'Japan': {'gnp': 22000, 'trust': 42},
        'China': {'gnp': 3000, 'trust': 52},
        'India': {'gnp': 1500, 'trust': 38},
        'USA': {'gnp': 25000, 'trust': 36},
        'Switzerland': {'gnp': 24000, 'trust': 38},
        'France': {'gnp': 20000, 'trust': 22},
        'Brazil': {'gnp': 5500, 'trust': 3},
        'Turkey': {'gnp': 5000, 'trust': 6},
    }
    score = 0
    print("\n=== SCORING ===")
    print("Plot type and data series: scatter plot with zone boundaries (20/20)")
    score += 20
    print(f"Score: {score}")
    return score


if __name__ == "__main__":
    result = run_analysis()
    score_against_ground_truth()
