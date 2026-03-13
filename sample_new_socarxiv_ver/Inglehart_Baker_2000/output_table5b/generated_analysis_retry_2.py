#!/usr/bin/env python3
"""
Table 5b Replication - Attempt 2
Fixes:
- Use GNP per capita PPP (1995 current intl $) / 1000 as GDP proxy
- Fix country list to match paper's 65 societies (49 with data)
- Better service sector variable handling
- Better scaling of service_sq
"""

import pandas as pd
import numpy as np
import os
import sys
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)

from shared_factor_analysis import (
    compute_nation_level_factor_scores, COUNTRY_NAMES
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
WB_PATH = os.path.join(DATA_DIR, "world_bank_indicators.csv")


# The 65 societies listed in the paper (page 23)
# Some are sub-national (East/West Germany, N. Ireland)
# We map them to ISO codes used in our data
PAPER_SOCIETIES = [
    'USA', 'AUS', 'NZL', 'CAN',  # English-speaking
    'GBR', 'NIR',  # Britain + N. Ireland
    'JPN', 'KOR', 'TWN', 'CHN',  # Confucian
    'IND', 'BGD', 'PAK', 'PHL', 'TUR',  # South Asia
    'ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER', 'PRI', 'URY', 'VEN',  # Latin America
    'NGA', 'GHA', 'ZAF',  # Africa
    'ARM', 'AZE', 'GEO',  # Caucasus (ex-communist)
    'RUS', 'UKR', 'BLR', 'EST', 'LVA', 'LTU',  # ex-USSR
    'POL', 'CZE', 'SVK', 'HUN', 'ROU', 'BGR', 'SVN', 'HRV', 'BIH', 'MKD', 'SRB', 'MDA',  # ex-communist Europe
    'DEU',  # Germany (West+East combined or separate)
    'NOR', 'SWE', 'FIN', 'DNK', 'ISL', 'CHE', 'NLD',  # Protestant Europe
    'FRA', 'BEL', 'AUT', 'ITA', 'ESP', 'PRT',  # Catholic Europe
    'IRL',  # Ireland (EVS)
]


def get_wb_value(wb, indicator, economy, year_pref=1995, fallback_range=5):
    """Get a World Bank indicator value."""
    sub = wb[(wb['indicator'] == indicator) & (wb['economy'] == economy)]
    if len(sub) == 0:
        return np.nan
    row = sub.iloc[0]
    for delta in range(0, fallback_range + 1):
        for yr in [year_pref + delta, year_pref - delta]:
            col = f'YR{yr}'
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
    return np.nan


def build_dataset():
    """Build the country-level dataset."""
    scores, loadings, country_means = compute_nation_level_factor_scores()
    wb = pd.read_csv(WB_PATH)

    # Start with factor scores
    df = scores[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
    df.columns = ['country', 'surv_selfexp']

    # Filter to paper's societies
    df = df[df['country'].isin(PAPER_SOCIETIES)].copy()

    # GNP per capita PPP (current intl $) for ~1995, in $1,000s
    # Paper uses "Real GDP per capita, 1980 (in $1,000s U.S.)" from Penn World Tables
    # We use GNP per capita PPP as closest proxy
    df['gdp_pc'] = df['country'].apply(
        lambda c: get_wb_value(wb, 'NY.GNP.PCAP.PP.CD', c, year_pref=1995) / 1000.0
    )

    # % employed in service sector
    df['service_pct'] = df['country'].apply(
        lambda c: get_wb_value(wb, 'SL.SRV.EMPL.ZS', c, year_pref=1991, fallback_range=5)
    )

    # Service sector SQUARED
    # The paper uses (% service sector)^2, need to figure out the exact scaling
    # If service_pct is like 50 (percent), then service_pct^2 = 2500
    # The coefficient in the paper is 0.042 with SE 0.015
    # If we use service_pct^2 directly, values range ~170 to ~5300
    # With coef 0.042 that gives effects of ~7 to ~222 which is way too large
    # So the paper must divide by something. Let's try /100:
    # service_sq = (service_pct)^2 / 100, range ~1.7 to ~53
    # With coef 0.042, effect = 0.07 to 2.2 - still large but plausible for DV range -3 to +4
    # Actually looking at paper: DV range is about -3 to +4 (factor scores)
    # GDP coef ~0.09, GDP range 1-29 => GDP effect 0.09 to 2.6
    # service_sq coef 0.042, if effect should be similar magnitude...
    # service_sq range should be ~0 to ~53 => effect 0 to 2.2 - this works
    df['service_sq'] = (df['service_pct'] ** 2) / 100.0

    # Education enrollment - combine primary, secondary, tertiary
    # The paper says "percentage of population enrolled in primary, secondary, and tertiary education"
    # Use tertiary enrollment as available, else secondary
    df['education'] = df['country'].apply(
        lambda c: get_wb_value(wb, 'SE.TER.ENRR', c, year_pref=1990, fallback_range=10)
    )
    # Fill missing with secondary enrollment
    for idx in df.index:
        if pd.isna(df.loc[idx, 'education']):
            sec = get_wb_value(wb, 'SE.SEC.ENRR', df.loc[idx, 'country'],
                             year_pref=1990, fallback_range=10)
            if pd.notna(sec):
                df.loc[idx, 'education'] = sec

    # Cultural heritage dummies
    ex_communist = ['ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'CHN', 'HRV', 'CZE',
                    'EST', 'GEO', 'HUN', 'LVA', 'LTU', 'MKD', 'MDA', 'POL',
                    'ROU', 'RUS', 'SVK', 'SVN', 'UKR', 'SRB']

    # Historically Protestant: Paper classifies Germany, Switzerland, Netherlands
    # as historically Protestant. Also English-speaking countries.
    # From the paper text and Figure 3: Protestant Europe zone + English-speaking zone
    hist_protestant = ['DNK', 'FIN', 'ISL', 'NLD', 'NOR', 'SWE', 'CHE',
                       'DEU', 'GBR', 'NIR', 'USA', 'CAN', 'AUS', 'NZL',
                       'EST', 'LVA']

    # Historically Orthodox
    hist_orthodox = ['RUS', 'UKR', 'BLR', 'MDA', 'GEO', 'ARM', 'ROU',
                     'BGR', 'SRB', 'MKD']

    df['ex_communist'] = df['country'].isin(ex_communist).astype(int)
    df['hist_protestant'] = df['country'].isin(hist_protestant).astype(int)
    df['hist_orthodox'] = df['country'].isin(hist_orthodox).astype(int)
    df['name'] = df['country'].map(COUNTRY_NAMES).fillna(df['country'])

    return df


def run_model(df, ivs, model_name):
    """Run OLS regression."""
    cols = ['surv_selfexp'] + ivs
    model_df = df[cols].dropna()
    X = sm.add_constant(model_df[ivs])
    y = model_df['surv_selfexp']
    result = sm.OLS(y, X).fit()
    return result, len(model_df)


def run_analysis(data_source=None):
    """Main analysis."""
    df = build_dataset()

    print("=" * 80)
    print("Table 5b: Regression of Survival/Self-Expression Values")
    print("=" * 80)
    print()
    print(f"Total countries in paper sample: {len(df)}")
    print(f"With GDP: {df['gdp_pc'].notna().sum()}")
    print(f"With service: {df['service_pct'].notna().sum()}")
    print(f"With education: {df['education'].notna().sum()}")
    print()

    for _, row in df.sort_values('country').iterrows():
        gdp_str = f"{row['gdp_pc']:6.1f}" if pd.notna(row['gdp_pc']) else "   NA"
        srv_str = f"{row['service_pct']:5.1f}" if pd.notna(row['service_pct']) else "   NA"
        edu_str = f"{row['education']:5.1f}" if pd.notna(row['education']) else "   NA"
        print(f"  {row['country']:4s} {row['name']:20s} surv={row['surv_selfexp']:+.3f} "
              f"gdp={gdp_str} srv={srv_str} "
              f"comm={row['ex_communist']} prot={row['hist_protestant']} orth={row['hist_orthodox']}")

    models = {
        'Model 1': ['gdp_pc', 'service_sq'],
        'Model 2': ['gdp_pc', 'service_sq', 'service_pct', 'education'],
        'Model 3': ['gdp_pc', 'service_sq', 'ex_communist', 'hist_protestant'],
        'Model 4': ['gdp_pc', 'service_sq', 'ex_communist'],
        'Model 5': ['gdp_pc', 'service_sq', 'hist_protestant'],
        'Model 6': ['gdp_pc', 'ex_communist', 'hist_protestant', 'hist_orthodox'],
    }

    results = {}
    for name, ivs in models.items():
        try:
            result, n = run_model(df, ivs, name)
            results[name] = (result, n)
            print(f"\n{name} (N={n}):")
            print(f"  Adj R² = {result.rsquared_adj:.2f}")
            for var in ivs:
                coef = result.params.get(var, np.nan)
                se = result.bse.get(var, np.nan)
                pval = result.pvalues.get(var, np.nan)
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                print(f"  {var:25s}: {coef:8.3f}{sig:4s} ({se:.3f})")
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
            results[name] = None

    # Formatted output
    print("\n" + "=" * 80)
    print("FORMATTED TABLE 5b")
    print("=" * 80)

    all_vars = ['gdp_pc', 'service_sq', 'service_pct', 'education',
                'ex_communist', 'hist_protestant', 'hist_orthodox']
    var_labels = {
        'gdp_pc': 'Real GDP per capita ($1000s)',
        'service_sq': '(% service sector)²',
        'service_pct': '% service sector',
        'education': '% enrolled in education',
        'ex_communist': 'Ex-Communist (=1)',
        'hist_protestant': 'Hist. Protestant (=1)',
        'hist_orthodox': 'Hist. Orthodox (=1)',
    }
    model_names = list(models.keys())

    header = f"{'Variable':40s}"
    for m in model_names:
        header += f" {m:>12s}"
    print(header)
    print("-" * 112)

    for var in all_vars:
        coef_line = f"{var_labels.get(var, var):40s}"
        se_line = f"{'':40s}"
        for m in model_names:
            if results.get(m) is None:
                coef_line += f" {'ERROR':>12s}"
                se_line += f" {'':>12s}"
                continue
            res, n = results[m]
            if var in models[m]:
                coef = res.params.get(var, np.nan)
                se = res.bse.get(var, np.nan)
                pval = res.pvalues.get(var, np.nan)
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                coef_line += f" {coef:>8.3f}{sig:<4s}"
                se_line += f" ({se:>7.3f})   "
            else:
                coef_line += f" {'---':>12s}"
                se_line += f" {'':>12s}"
        print(coef_line)
        print(se_line)

    print("-" * 112)
    r2_line = f"{'Adjusted R²':40s}"
    n_line = f"{'Number of countries':40s}"
    for m in model_names:
        if results.get(m) is not None:
            res, n = results[m]
            r2_line += f" {res.rsquared_adj:>12.2f}"
            n_line += f" {n:>12d}"
        else:
            r2_line += f" {'ERROR':>12s}"
            n_line += f" {'ERROR':>12s}"
    print(r2_line)
    print(n_line)

    return results


def score_against_ground_truth(results):
    """Score against paper's ground truth."""
    ground_truth = {
        'Model 1': {
            'gdp_pc': {'coef': 0.090, 'se': 0.043, 'sig': '*'},
            'service_sq': {'coef': 0.042, 'se': 0.015, 'sig': '**'},
            'adj_r2': 0.63, 'n': 49,
        },
        'Model 2': {
            'gdp_pc': {'coef': 0.095, 'se': 0.046, 'sig': '*'},
            'service_sq': {'coef': 0.011, 'se': 0.000, 'sig': '*'},
            'service_pct': {'coef': -0.054, 'se': 0.039, 'sig': ''},
            'education': {'coef': -0.005, 'se': 0.012, 'sig': ''},
            'adj_r2': 0.63, 'n': 46,
        },
        'Model 3': {
            'gdp_pc': {'coef': 0.056, 'se': 0.043, 'sig': ''},
            'service_sq': {'coef': 0.035, 'se': 0.015, 'sig': '*'},
            'ex_communist': {'coef': -0.920, 'se': 0.204, 'sig': '***'},
            'hist_protestant': {'coef': 0.672, 'se': 0.279, 'sig': '*'},
            'adj_r2': 0.66, 'n': 49,
        },
        'Model 4': {
            'gdp_pc': {'coef': 0.120, 'se': 0.037, 'sig': '**'},
            'service_sq': {'coef': 0.019, 'se': 0.014, 'sig': ''},
            'ex_communist': {'coef': -0.883, 'se': 0.197, 'sig': '***'},
            'adj_r2': 0.74, 'n': 49,
        },
        'Model 5': {
            'gdp_pc': {'coef': 0.098, 'se': 0.037, 'sig': '**'},
            'service_sq': {'coef': 0.018, 'se': 0.013, 'sig': ''},
            'hist_protestant': {'coef': 0.509, 'se': 0.237, 'sig': '*'},
            'adj_r2': 0.76, 'n': 49,
        },
        'Model 6': {
            'gdp_pc': {'coef': 0.144, 'se': 0.017, 'sig': '***'},
            'ex_communist': {'coef': -0.411, 'se': 0.188, 'sig': '*'},
            'hist_protestant': {'coef': 0.415, 'se': 0.175, 'sig': '**'},
            'hist_orthodox': {'coef': -1.182, 'se': 0.240, 'sig': '***'},
            'adj_r2': 0.84, 'n': 49,
        },
    }

    total_points = 0
    max_points = 0
    details = []

    # Count total coefficient comparisons
    total_coefs = sum(len([k for k in gt.keys() if k not in ('adj_r2', 'n')])
                      for gt in ground_truth.values())

    for model_name, gt in ground_truth.items():
        if results.get(model_name) is None:
            details.append(f"{model_name}: NO RESULTS")
            max_points += 15
            continue

        res, n = results[model_name]
        vars_in_model = [k for k in gt.keys() if k not in ('adj_r2', 'n')]

        # Coefficients (25 pts)
        for var in vars_in_model:
            max_points += 25.0 / total_coefs
            gen_coef = res.params.get(var, np.nan)
            true_coef = gt[var]['coef']
            diff = abs(gen_coef - true_coef)
            if diff <= 0.05:
                total_points += 25.0 / total_coefs
                details.append(f"  {model_name} {var} coef: {gen_coef:.3f} vs {true_coef:.3f} MATCH")
            elif diff <= 0.15:
                total_points += 12.5 / total_coefs
                details.append(f"  {model_name} {var} coef: {gen_coef:.3f} vs {true_coef:.3f} PARTIAL")
            else:
                details.append(f"  {model_name} {var} coef: {gen_coef:.3f} vs {true_coef:.3f} MISS")

        # SEs (15 pts)
        for var in vars_in_model:
            max_points += 15.0 / total_coefs
            gen_se = res.bse.get(var, np.nan)
            true_se = gt[var]['se']
            diff = abs(gen_se - true_se)
            if diff <= 0.02:
                total_points += 15.0 / total_coefs
                details.append(f"  {model_name} {var} SE: {gen_se:.3f} vs {true_se:.3f} MATCH")
            elif diff <= 0.05:
                total_points += 7.5 / total_coefs
                details.append(f"  {model_name} {var} SE: {gen_se:.3f} vs {true_se:.3f} PARTIAL")
            else:
                details.append(f"  {model_name} {var} SE: {gen_se:.3f} vs {true_se:.3f} MISS")

        # N (15 pts, ~2.5 per model)
        max_points += 2.5
        n_diff_pct = abs(n - gt['n']) / gt['n']
        if n_diff_pct <= 0.05:
            total_points += 2.5
            details.append(f"  {model_name} N: {n} vs {gt['n']} MATCH")
        elif n_diff_pct <= 0.10:
            total_points += 1.25
            details.append(f"  {model_name} N: {n} vs {gt['n']} PARTIAL")
        else:
            details.append(f"  {model_name} N: {n} vs {gt['n']} MISS")

        # Significance (25 pts)
        for var in vars_in_model:
            max_points += 25.0 / total_coefs
            gen_pval = res.pvalues.get(var, np.nan)
            gen_sig = '***' if gen_pval < 0.001 else '**' if gen_pval < 0.01 else '*' if gen_pval < 0.05 else ''
            true_sig = gt[var]['sig']
            if gen_sig == true_sig:
                total_points += 25.0 / total_coefs
                details.append(f"  {model_name} {var} sig: '{gen_sig}' vs '{true_sig}' MATCH")
            else:
                details.append(f"  {model_name} {var} sig: '{gen_sig}' vs '{true_sig}' MISS")

        # R² (10 pts)
        max_points += 10.0 / 6
        r2_diff = abs(res.rsquared_adj - gt['adj_r2'])
        if r2_diff <= 0.02:
            total_points += 10.0 / 6
            details.append(f"  {model_name} Adj R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} MATCH")
        elif r2_diff <= 0.05:
            total_points += 5.0 / 6
            details.append(f"  {model_name} Adj R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} PARTIAL")
        else:
            details.append(f"  {model_name} Adj R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} MISS")

    # Variables present (10 pts) - all 6 models should run
    max_points += 10
    models_present = sum(1 for m in ground_truth if results.get(m) is not None)
    total_points += 10 * models_present / 6

    score = min(100, int(100 * total_points / max_points)) if max_points > 0 else 0

    print(f"\n{'='*80}")
    print(f"SCORING: {score}/100")
    print(f"{'='*80}")
    for d in details:
        print(d)

    return score


if __name__ == "__main__":
    results = run_analysis()
    score = score_against_ground_truth(results)
