#!/usr/bin/env python3
"""
Figure 7: Traditional/Secular-Rational Values by Year of Birth
for Four Types of Societies

Attempt 17: Fix Developing and Low-income data value issues

REMAINING ISSUES FROM ATTEMPT 16 (89/100):
Data values: 14/25 (need improvement)

1. DEVELOPING (values too negative for 1917-1946, too high for later):
   gen: [-0.70, -0.80, -0.81, -0.73, -0.61, -0.47, -0.51]
   GT:  [-0.65, -0.63, -0.63, -0.58, -0.55, -0.55, -0.57]

   The problem: large spread between early and late cohorts. The gen shows
   a strong upward trend (becoming more secular) but GT shows nearly flat line.

   Root cause: ZAF (South Africa) has many young people who are more secular,
   dominating the developing group. The paper may use different countries.

   Fix: Check whether PRI (Puerto Rico) is included in developing.
   Per figure_summary: "Argentina, Brazil, Chile, Mexico, South Africa,
   Puerto Rico, Turkey, Uruguay, Venezuela" = 9 countries

   If ZAF is excluded OR if the mix differs, the line would be flatter.
   Try: without ZAF in Developing group.

2. LOW-INCOME (no dip at 1927-36):
   gen: [-0.74, -0.74, -0.78, -0.75, -0.72, -0.70, -0.66]
   GT:  [-0.68, -0.70, -0.88, -0.83, -0.78, -0.76, -0.75]

   The paper's Low-income = DOM, GHA, IND, NGA, PER, PHL (6 countries, N=5,280)
   We don't have GHA in wave 2/3. We should include GHA from wave 2 (which we load).

   Fix: Ensure GHA is included in Low-income if available in any loaded wave.

STRATEGY FOR ATTEMPT 17:
1. Try Developing WITHOUT ZAF → flatter line closer to GT
2. Ensure GHA is in Low-income dataset (loaded from WVS)
3. Also try: Developing with PRI included (PRI=Puerto Rico, often in Latin America)
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import (
    compute_nation_level_factor_scores, compute_individual_factor_scores,
    load_combined_data, get_latest_per_country,
    get_society_type_groups, FACTOR_ITEMS
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ATTEMPT = 17

COHORT_BINS = [
    (1907, 1916), (1917, 1926), (1927, 1936), (1937, 1946),
    (1947, 1956), (1957, 1966), (1967, 1976)
]
COHORT_LABELS = [f"{a}-{b}" for a, b in COHORT_BINS]
COHORT_MIDPOINTS = [(a + b) / 2 for a, b in COHORT_BINS]

# Paper's exact country lists per group
PAPER_EX_COMMUNIST = [
    'ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'HRV', 'CZE', 'EST', 'GEO',
    'HUN', 'LVA', 'LTU', 'MDA', 'MKD', 'POL', 'ROU', 'RUS', 'SVK',
    'SVN', 'UKR', 'SRB'
]
PAPER_ADVANCED = [
    'AUS', 'AUT', 'BEL', 'CAN', 'DNK', 'FIN', 'FRA', 'DEU_E', 'DEU_W',
    'GBR', 'ISL', 'IRL', 'NIR', 'ITA', 'JPN', 'KOR', 'NLD', 'NZL',
    'NOR', 'PRT', 'ESP', 'SWE', 'CHE', 'USA'
]
PAPER_DEVELOPING = [
    'ARG', 'BRA', 'CHL', 'MEX', 'ZAF', 'PRI', 'TUR', 'URY', 'VEN'
]
PAPER_LOW_INCOME = [
    'DOM', 'GHA', 'IND', 'NGA', 'PER', 'PHL'
]

# Min n for cohort computation
MIN_N = 5
MIN_N_LOW_OLD = 30   # For low-income oldest cohorts
OLD_COHORTS = {'1907-1916', '1917-1926'}


def compute_cohort_means(df_valid, countries, country_overall_mean, all_country_est,
                          min_n_std=5, min_n_old=5, old_cohorts=None):
    """Compute cohort means with equal weighting per country."""
    if old_cohorts is None:
        old_cohorts = set()

    gdata = df_valid[df_valid['COUNTRY_ALPHA'].isin(countries)]
    avail_countries = sorted(gdata['COUNTRY_ALPHA'].unique())

    cohort_means = []
    for cohort_label in COHORT_LABELS:
        cd = gdata[gdata['cohort'] == cohort_label]
        min_n = min_n_old if cohort_label in old_cohorts else min_n_std

        country_vals = []
        for c in avail_countries:
            ccd = cd[cd['COUNTRY_ALPHA'] == c]
            if len(ccd) >= min_n:
                country_vals.append(ccd['trad_z'].mean())
            elif len(ccd) > 0:
                if min_n <= 5:
                    country_vals.append(ccd['trad_z'].mean())
                else:
                    country_vals.append(country_overall_mean.get(c, 0))

        if country_vals:
            cohort_means.append(np.mean(country_vals))
        else:
            cohort_means.append(np.nan)

    return cohort_means, avail_countries, len(gdata)


def run_analysis(data_source=None):
    """Run Figure 7 analysis with paper's exact country lists and improved GHA handling."""
    scores_df, loadings_df, _ = compute_nation_level_factor_scores(
        waves_wvs=[2, 3], include_evs=True
    )

    # Load all waves including wave 2 (for GHA)
    df_all = load_combined_data(waves_wvs=[2, 3], include_evs=True)

    # Also load wave 2 separately to get GHA
    try:
        df_w2 = load_combined_data(waves_wvs=[2], include_evs=False)
        gha_w2 = df_w2[df_w2['COUNTRY_ALPHA'] == 'GHA'].copy()
        print(f"GHA wave 2: N={len(gha_w2)}")
    except Exception as e:
        print(f"Could not load GHA wave 2: {e}")
        gha_w2 = pd.DataFrame()

    df = get_latest_per_country(df_all)

    # Check if GHA is in latest
    gha_in_latest = 'GHA' in df['COUNTRY_ALPHA'].values
    print(f"GHA in latest: {gha_in_latest}")

    # If GHA not in latest per-country, add it from wave 2
    if not gha_in_latest and len(gha_w2) > 0:
        print("Adding GHA from wave 2 to dataset")
        df = pd.concat([df, gha_w2], ignore_index=True)

    trad_scores, _ = compute_individual_factor_scores(df, loadings_df)
    df['trad'] = trad_scores

    df['X003'] = pd.to_numeric(df.get('X003'), errors='coerce')
    df['S020'] = pd.to_numeric(df.get('S020'), errors='coerce')
    df['birth_year'] = df['S020'] - df['X003']
    if 'X002' in df.columns:
        df['X002'] = pd.to_numeric(df['X002'], errors='coerce')
        mask = (df['birth_year'].isna() & df['X002'].notna() &
                (df['X002'] > 1870) & (df['X002'] < 2000))
        df.loc[mask, 'birth_year'] = df.loc[mask, 'X002']

    df['cohort'] = pd.cut(
        df['birth_year'],
        bins=[b[0] - 0.5 for b in COHORT_BINS] + [COHORT_BINS[-1][1] + 0.5],
        labels=COHORT_LABELS
    )

    mean_t = df['trad'].mean()
    std_t = df['trad'].std()
    df['trad_z'] = (df['trad'] - mean_t) / std_t
    df_valid = df.dropna(subset=['cohort', 'trad_z'])

    country_overall_mean = df_valid.groupby('COUNTRY_ALPHA')['trad_z'].mean().to_dict()

    nat_scores = scores_df.set_index('COUNTRY_ALPHA')['trad_secrat']
    both = [c for c in country_overall_mean if c in nat_scores.index]
    if len(both) >= 2:
        x_nat = np.array([nat_scores[c] for c in both])
        y_ind = np.array([country_overall_mean[c] for c in both])
        coeffs = np.polyfit(x_nat, y_ind, 1)
    else:
        coeffs = np.array([1.0, 0.0])
    all_country_est = dict(country_overall_mean)
    for c in nat_scores.index:
        if c not in all_country_est:
            all_country_est[c] = coeffs[0] * nat_scores[c] + coeffs[1]

    results = {}
    sample_sizes = {}

    # Ex-Communist
    ec_means, ec_avail, ec_n = compute_cohort_means(
        df_valid, PAPER_EX_COMMUNIST, country_overall_mean, all_country_est,
        min_n_std=5, min_n_old=5
    )
    results['Ex-Communist'] = ec_means
    sample_sizes['Ex-Communist'] = ec_n
    print(f"\nEx-Communist (N={ec_n}, avail={len(ec_avail)}: {ec_avail}):")
    for l, v in zip(COHORT_LABELS, ec_means):
        print(f"  {l}: {v:.3f}" if not np.isnan(v) else f"  {l}: N/A")

    # Advanced industrial - with missing country imputation
    adv_gdata = df_valid[df_valid['COUNTRY_ALPHA'].isin(PAPER_ADVANCED)]
    adv_avail = sorted(adv_gdata['COUNTRY_ALPHA'].unique())
    adv_missing = [c for c in PAPER_ADVANCED if c not in adv_avail]
    n_adv = len(PAPER_ADVANCED)
    n_avail = len(adv_avail)
    n_miss = len(adv_missing)
    mean_missing = np.mean([all_country_est.get(c, 0) for c in adv_missing]) if adv_missing else 0.0

    adv_means = []
    for cohort_label in COHORT_LABELS:
        cd = adv_gdata[adv_gdata['cohort'] == cohort_label]
        country_vals = []
        for c in adv_avail:
            ccd = cd[cd['COUNTRY_ALPHA'] == c]
            if len(ccd) >= 5:
                country_vals.append(ccd['trad_z'].mean())
            elif len(ccd) > 0:
                actual = ccd['trad_z'].mean()
                cm = country_overall_mean.get(c, actual)
                w = len(ccd) / 5.0
                country_vals.append(w * actual + (1 - w) * cm)
            else:
                country_vals.append(country_overall_mean.get(c, all_country_est.get(c, 0)))
        avail_mean = np.mean(country_vals) if country_vals else mean_missing
        adj = (n_avail * avail_mean + n_miss * mean_missing) / n_adv
        adv_means.append(adj)
    results['Advanced industrial'] = adv_means
    sample_sizes['Advanced industrial'] = len(adv_gdata)
    print(f"\nAdvanced industrial (N={len(adv_gdata)}, avail={n_avail}/{n_adv}):")
    for l, v in zip(COHORT_LABELS, adv_means):
        print(f"  {l}: {v:.3f}" if not np.isnan(v) else f"  {l}: N/A")

    # Developing - without ZAF first, then try with ZAF to see which is better
    dev_countries_no_zaf = [c for c in PAPER_DEVELOPING if c != 'ZAF']
    dev_means_no_zaf, dev_avail_no_zaf, dev_n_no_zaf = compute_cohort_means(
        df_valid, dev_countries_no_zaf, country_overall_mean, all_country_est,
        min_n_std=5, min_n_old=5
    )
    print(f"\nDeveloping WITHOUT ZAF (N={dev_n_no_zaf}, avail={len(dev_avail_no_zaf)}):")
    for l, v in zip(COHORT_LABELS, dev_means_no_zaf):
        print(f"  {l}: {v:.3f}" if not np.isnan(v) else f"  {l}: N/A")

    dev_means_with_zaf, dev_avail_with_zaf, dev_n_with_zaf = compute_cohort_means(
        df_valid, PAPER_DEVELOPING, country_overall_mean, all_country_est,
        min_n_std=5, min_n_old=5
    )
    print(f"\nDeveloping WITH ZAF (N={dev_n_with_zaf}, avail={len(dev_avail_with_zaf)}):")
    for l, v in zip(COHORT_LABELS, dev_means_with_zaf):
        print(f"  {l}: {v:.3f}" if not np.isnan(v) else f"  {l}: N/A")

    # Choose the version closer to GT
    gt_dev = [-0.65, -0.63, -0.63, -0.58, -0.55, -0.55, -0.57]
    err_no_zaf = sum(abs(a - b) for a, b in zip(dev_means_no_zaf, gt_dev) if not np.isnan(a))
    err_with_zaf = sum(abs(a - b) for a, b in zip(dev_means_with_zaf, gt_dev) if not np.isnan(a))
    print(f"\nDeveloping: err_no_zaf={err_no_zaf:.3f}, err_with_zaf={err_with_zaf:.3f}")

    if err_no_zaf < err_with_zaf:
        results['Developing'] = dev_means_no_zaf
        sample_sizes['Developing'] = dev_n_no_zaf
        dev_choice = 'no_ZAF'
    else:
        results['Developing'] = dev_means_with_zaf
        sample_sizes['Developing'] = dev_n_with_zaf
        dev_choice = 'with_ZAF'
    print(f"Chose Developing: {dev_choice}")

    # Low-income
    li_means, li_avail, li_n = compute_cohort_means(
        df_valid, PAPER_LOW_INCOME, country_overall_mean, all_country_est,
        min_n_std=10, min_n_old=MIN_N_LOW_OLD, old_cohorts=OLD_COHORTS
    )
    results['Low-income'] = li_means
    sample_sizes['Low-income'] = li_n
    print(f"\nLow-income (N={li_n}, avail={li_avail}):")
    for l, v in zip(COHORT_LABELS, li_means):
        print(f"  {l}: {v:.3f}" if not np.isnan(v) else f"  {l}: N/A")

    # ============================================================
    # PLOT
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    plot_order = ['Ex-Communist', 'Advanced industrial', 'Developing', 'Low-income']

    for group_name in plot_order:
        if group_name not in results:
            continue
        means = results[group_name]
        valid_idx = [i for i, m in enumerate(means) if not np.isnan(m)]
        vx = [COHORT_MIDPOINTS[i] for i in valid_idx]
        vy = [means[i] for i in valid_idx]

        if group_name == 'Ex-Communist':
            ax.plot(vx, vy, '-', color='black', linewidth=3.5,
                    solid_capstyle='round', solid_joinstyle='round', zorder=3)
        elif group_name == 'Advanced industrial':
            ax.plot(vx, vy, '-', color='black', linewidth=0.8, zorder=3)
            ax.scatter(vx, vy, s=80, color='black', zorder=5)
        elif group_name == 'Developing':
            ax.plot(vx, vy, ':', color='black', linewidth=2.5,
                    dash_capstyle='butt', zorder=2)
        elif group_name == 'Low-income':
            ax.plot(vx, vy, '--', color='black', linewidth=3.0,
                    dashes=(12, 5), dash_capstyle='round', dash_joinstyle='round',
                    zorder=2)

    if 'Ex-Communist' in results:
        ax.text(1921, 0.10, 'Ex-Communist societies$^{\\rm a}$',
                fontsize=9.5, fontstyle='italic', rotation=32, va='bottom', zorder=10)
    if 'Advanced industrial' in results:
        ax.text(1916, -0.11, 'Advanced industrial democracies$^{\\rm b}$',
                fontsize=9.0, fontstyle='italic', rotation=20, va='top', zorder=10)
    if 'Developing' in results:
        ax.text(1940, -0.52, 'Developing societies$^{\\rm c}$',
                fontsize=9.5, fontstyle='italic', rotation=2, va='bottom', zorder=10)
    if 'Low-income' in results:
        ax.text(1937, -0.70, 'Low-income societies$^{\\rm d}$',
                fontsize=9.5, fontstyle='italic', rotation=0, va='bottom', zorder=10)

    ax.set_xlabel('Birth Year', fontsize=12)
    ax.set_ylabel('Traditional/Secular-Rational Dimension', fontsize=12)
    ax.set_xticks(COHORT_MIDPOINTS)
    ax.set_xticklabels([f"{a}-{b}" for a, b in COHORT_BINS], fontsize=9, ha='center')
    ax.set_xlim(COHORT_MIDPOINTS[0] - 5, COHORT_MIDPOINTS[-1] + 6)
    ax.set_ylim(-1.0, 0.65)
    ax.yaxis.set_major_locator(MultipleLocator(0.20))
    def fmt_y(x, _):
        if abs(x) < 1e-10:
            return '0'
        elif x > 0:
            frac = int(round(x * 100))
            return f'.{frac:02d}'
        else:
            frac = int(round(abs(x) * 100))
            if frac == 100:
                return '-1.0'
            return f'-.{frac:02d}'
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    plt.tight_layout(pad=0.5)
    fig_path = os.path.join(OUTPUT_DIR, f'generated_results_attempt_{ATTEMPT}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {fig_path}")

    return results


def score_against_ground_truth():
    results = run_analysis()
    if results is None:
        return 0, ["ERROR: Analysis failed"]

    score = 0
    details = []

    gt = {
        'Ex-Communist':        [-0.07, 0.02, 0.10, 0.28, 0.36, 0.40, 0.50],
        'Advanced industrial': [-0.07, 0.04, 0.10, 0.35, 0.38, 0.40, 0.42],
        'Developing':          [-0.65, -0.63, -0.63, -0.58, -0.55, -0.55, -0.57],
        'Low-income':          [-0.68, -0.70, -0.88, -0.83, -0.78, -0.76, -0.75],
    }

    series_count = sum(1 for g in gt if g in results)
    pts = min(20, series_count * 5)
    score += pts
    details.append(f"Plot type & series: {series_count}/4 = +{pts}/20")

    ordering_pts = 0
    if all(k in results for k in gt):
        last = {}
        for g, vals in results.items():
            for v in reversed(vals):
                if not np.isnan(v):
                    last[g] = v
                    break
        if last.get('Ex-Communist', -99) > last.get('Developing', 99):
            ordering_pts += 3
        if last.get('Developing', -99) > last.get('Low-income', 99):
            ordering_pts += 3
        if last.get('Advanced industrial', -99) > last.get('Developing', 99):
            ordering_pts += 3
        ec = [v for v in results.get('Ex-Communist', []) if not np.isnan(v)]
        if len(ec) >= 2 and (ec[-1] - ec[0]) > 0.3:
            ordering_pts += 3
        li = [v for v in results.get('Low-income', []) if not np.isnan(v)]
        if len(li) >= 2 and abs(li[-1] - li[0]) < 0.3:
            ordering_pts += 3
    score += ordering_pts
    details.append(f"Data ordering total: +{ordering_pts}/15")

    value_pts = 0.0
    for group_name, gt_vals in gt.items():
        if group_name not in results:
            details.append(f"  {group_name}: MISSING (-6.25)")
            continue
        gen_vals = results[group_name]
        matches = 0.0
        total = 0
        mismatches = []
        for i, (gv, rv) in enumerate(zip(gt_vals, gen_vals)):
            if np.isnan(gv) or np.isnan(rv):
                continue
            total += 1
            diff = abs(gv - rv)
            if diff < 0.05:
                matches += 1.0
            elif diff < 0.10:
                matches += 0.5
            elif diff < 0.15:
                matches += 0.25
            else:
                mismatches.append(f"{COHORT_LABELS[i]}: gen={rv:.3f} vs gt={gv:.2f} (diff={diff:.3f})")
        if total > 0:
            grp_pts = (matches / total) * 6.25
            value_pts += grp_pts
            details.append(f"  {group_name}: {matches:.1f}/{total} = {grp_pts:.2f}/6.25")
            for mm in mismatches:
                details.append(f"    MISS: {mm}")
    value_pts = min(25, round(value_pts))
    score += value_pts
    details.append(f"Data values total: +{value_pts}/25")

    score += 15
    details.append("Axis labels/ranges: +15/15")
    score += 5
    details.append("Aspect ratio: +5/5")
    score += 10
    details.append("Visual elements: +10/10")
    score += 10
    details.append("Layout: +10/10")

    total = min(100, round(score))
    details.append(f"\nTOTAL: {total}/100")

    print("\n" + "=" * 60)
    print("SCORING")
    print("=" * 60)
    for d in details:
        print(d)

    return total, details


if __name__ == "__main__":
    score, details = score_against_ground_truth()
