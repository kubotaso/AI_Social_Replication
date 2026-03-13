#!/usr/bin/env python3
"""
Table 5b Replication: Unstandardized Coefficients from the Regression of
Survival/Self-Expression Values on Modernization and Cultural Heritage Variables.

DV: Survival/Self-Expression factor score (nation-level)
IVs: GDP per capita, service sector squared, cultural heritage dummies

6 models with different IV combinations.
"""

import pandas as pd
import numpy as np
import os
import sys
import statsmodels.api as sm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)

from shared_factor_analysis import (
    compute_nation_level_factor_scores, load_world_bank_data, COUNTRY_NAMES
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
WB_PATH = os.path.join(DATA_DIR, "world_bank_indicators.csv")


def get_wb_value(wb, indicator, economy, year_pref=1990, fallback_range=5):
    """Get a World Bank indicator value for an economy, trying nearby years if needed."""
    sub = wb[(wb['indicator'] == indicator) & (wb['economy'] == economy)]
    if len(sub) == 0:
        return np.nan
    row = sub.iloc[0]
    # Try preferred year first, then nearby years
    for delta in range(0, fallback_range + 1):
        for yr in [year_pref + delta, year_pref - delta]:
            col = f'YR{yr}'
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
    return np.nan


def build_dataset():
    """Build the country-level dataset with factor scores and IVs."""
    # Compute factor scores
    scores, loadings, country_means = compute_nation_level_factor_scores()

    # Load World Bank data
    wb = pd.read_csv(WB_PATH)

    # Build country-level dataframe
    df = scores[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
    df.columns = ['country', 'surv_selfexp']

    # GDP per capita PPP (constant 2017 intl $) - use 1990 as proxy for 1980
    # The paper uses "Real GDP per capita, 1980 (in $1,000s U.S.)"
    # We scale to $1,000s
    df['gdp_pc'] = df['country'].apply(
        lambda c: get_wb_value(wb, 'NY.GDP.PCAP.PP.KD', c, year_pref=1990) / 1000.0
    )

    # % employed in service sector - use 1991 as earliest available
    df['service_pct'] = df['country'].apply(
        lambda c: get_wb_value(wb, 'SL.SRV.EMPL.ZS', c, year_pref=1991)
    )

    # Service sector SQUARED (divided by 100 for scaling)
    df['service_sq'] = (df['service_pct'] ** 2) / 100.0

    # Education enrollment - combine secondary and tertiary
    # The paper uses "% enrolled in education" - likely tertiary or combined
    # Use tertiary enrollment as closest match
    df['education'] = df['country'].apply(
        lambda c: get_wb_value(wb, 'SE.TER.ENRR', c, year_pref=1990, fallback_range=10)
    )
    # If tertiary not available, try secondary
    for idx in df.index:
        if pd.isna(df.loc[idx, 'education']):
            df.loc[idx, 'education'] = get_wb_value(
                wb, 'SE.SEC.ENRR', df.loc[idx, 'country'], year_pref=1990, fallback_range=10
            )

    # Cultural heritage dummies
    ex_communist = ['ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'CHN', 'HRV', 'CZE',
                    'EST', 'GEO', 'HUN', 'LVA', 'LTU', 'MKD', 'MDA', 'POL',
                    'ROU', 'RUS', 'SVK', 'SVN', 'UKR', 'SRB']

    # Historically Protestant: includes Protestant Europe + English-speaking countries
    hist_protestant = ['DNK', 'FIN', 'ISL', 'NLD', 'NOR', 'SWE', 'CHE',
                       'DEU', 'GBR', 'NIR', 'USA', 'CAN', 'AUS', 'NZL',
                       'EST', 'LVA']

    # Historically Orthodox
    hist_orthodox = ['RUS', 'UKR', 'BLR', 'MDA', 'GEO', 'ARM', 'ROU',
                     'BGR', 'SRB', 'MKD', 'GRC']

    df['ex_communist'] = df['country'].isin(ex_communist).astype(int)
    df['hist_protestant'] = df['country'].isin(hist_protestant).astype(int)
    df['hist_orthodox'] = df['country'].isin(hist_orthodox).astype(int)

    # Add country names for display
    df['name'] = df['country'].map(COUNTRY_NAMES).fillna(df['country'])

    return df


def run_model(df, ivs, model_name):
    """Run OLS regression with given IVs."""
    cols = ['surv_selfexp'] + ivs
    model_df = df[cols].dropna()

    X = model_df[ivs]
    X = sm.add_constant(X)
    y = model_df['surv_selfexp']

    result = sm.OLS(y, X).fit()
    return result, len(model_df)


def run_analysis(data_source=None):
    """Main analysis function."""
    df = build_dataset()

    print("=" * 80)
    print("Table 5b: Regression of Survival/Self-Expression Values")
    print("=" * 80)
    print()

    # Print data summary
    print(f"Total countries with factor scores: {len(df)}")
    print(f"Countries with GDP data: {df['gdp_pc'].notna().sum()}")
    print(f"Countries with service data: {df['service_pct'].notna().sum()}")
    print(f"Countries with education data: {df['education'].notna().sum()}")
    print()

    # Print country data for verification
    print("Country data:")
    for _, row in df.sort_values('country').iterrows():
        print(f"  {row['country']:4s} {row['name']:20s} surv={row['surv_selfexp']:+.3f} "
              f"gdp={row['gdp_pc']:6.1f} srv={row['service_pct']:5.1f} "
              f"srv2={row['service_sq']:6.1f} "
              f"comm={row['ex_communist']} prot={row['hist_protestant']} orth={row['hist_orthodox']}")
    print()

    # Define models
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
                sig = ''
                if pval < 0.001:
                    sig = '***'
                elif pval < 0.01:
                    sig = '**'
                elif pval < 0.05:
                    sig = '*'
                print(f"  {var:25s}: {coef:8.3f}{sig:4s} ({se:.3f})")
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")
            results[name] = None

    # Print formatted table
    print("\n" + "=" * 80)
    print("FORMATTED TABLE 5b")
    print("=" * 80)

    all_vars = ['gdp_pc', 'service_sq', 'service_pct', 'education',
                'ex_communist', 'hist_protestant', 'hist_orthodox']
    var_labels = {
        'gdp_pc': 'Real GDP per capita, 1980',
        'service_sq': '(% service sector, 1980)²',
        'service_pct': '% service sector, 1980',
        'education': '% enrolled in education',
        'ex_communist': 'Ex-Communist (=1)',
        'hist_protestant': 'Historically Protestant (=1)',
        'hist_orthodox': 'Historically Orthodox (=1)',
    }

    model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']

    # Header
    header = f"{'Variable':40s}"
    for m in model_names:
        header += f" {m:>10s}"
    print(header)
    print("-" * 100)

    for var in all_vars:
        coef_line = f"{var_labels.get(var, var):40s}"
        se_line = f"{'':40s}"

        for m in model_names:
            if results.get(m) is None:
                coef_line += f" {'ERROR':>10s}"
                se_line += f" {'':>10s}"
                continue

            res, n = results[m]
            if var in models[m]:
                coef = res.params.get(var, np.nan)
                se = res.bse.get(var, np.nan)
                pval = res.pvalues.get(var, np.nan)
                sig = ''
                if pval < 0.001:
                    sig = '***'
                elif pval < 0.01:
                    sig = '**'
                elif pval < 0.05:
                    sig = '*'
                coef_line += f" {coef:>7.3f}{sig:<3s}"
                se_line += f" ({se:>6.3f})  "
            else:
                coef_line += f" {'---':>10s}"
                se_line += f" {'':>10s}"

        print(coef_line)
        print(se_line)

    print("-" * 100)

    # Adjusted R²
    r2_line = f"{'Adjusted R²':40s}"
    n_line = f"{'Number of countries':40s}"
    for m in model_names:
        if results.get(m) is None:
            r2_line += f" {'ERROR':>10s}"
            n_line += f" {'ERROR':>10s}"
        else:
            res, n = results[m]
            r2_line += f" {res.rsquared_adj:>10.2f}"
            n_line += f" {n:>10d}"
    print(r2_line)
    print(n_line)

    return results


def score_against_ground_truth(results):
    """Score the results against the paper's ground truth values."""
    # Ground truth from Table 5b of the paper
    ground_truth = {
        'Model 1': {
            'gdp_pc': {'coef': 0.090, 'se': 0.043, 'sig': '*'},
            'service_sq': {'coef': 0.042, 'se': 0.015, 'sig': '**'},
            'adj_r2': 0.63,
            'n': 49,
        },
        'Model 2': {
            'gdp_pc': {'coef': 0.095, 'se': 0.046, 'sig': '*'},
            'service_sq': {'coef': 0.011, 'se': 0.000, 'sig': '*'},
            'service_pct': {'coef': -0.054, 'se': 0.039, 'sig': ''},
            'education': {'coef': -0.005, 'se': 0.012, 'sig': ''},
            'adj_r2': 0.63,
            'n': 46,
        },
        'Model 3': {
            'gdp_pc': {'coef': 0.056, 'se': 0.043, 'sig': ''},
            'service_sq': {'coef': 0.035, 'se': 0.015, 'sig': '*'},
            'ex_communist': {'coef': -0.920, 'se': 0.204, 'sig': '***'},
            'hist_protestant': {'coef': 0.672, 'se': 0.279, 'sig': '*'},
            'adj_r2': 0.66,
            'n': 49,
        },
        'Model 4': {
            'gdp_pc': {'coef': 0.120, 'se': 0.037, 'sig': '**'},
            'service_sq': {'coef': 0.019, 'se': 0.014, 'sig': ''},
            'ex_communist': {'coef': -0.883, 'se': 0.197, 'sig': '***'},
            'adj_r2': 0.74,
            'n': 49,
        },
        'Model 5': {
            'gdp_pc': {'coef': 0.098, 'se': 0.037, 'sig': '**'},
            'service_sq': {'coef': 0.018, 'se': 0.013, 'sig': ''},
            'hist_protestant': {'coef': 0.509, 'se': 0.237, 'sig': '*'},
            'adj_r2': 0.76,
            'n': 49,
        },
        'Model 6': {
            'gdp_pc': {'coef': 0.144, 'se': 0.017, 'sig': '***'},
            'ex_communist': {'coef': -0.411, 'se': 0.188, 'sig': '*'},
            'hist_protestant': {'coef': 0.415, 'se': 0.175, 'sig': '**'},
            'hist_orthodox': {'coef': -1.182, 'se': 0.240, 'sig': '***'},
            'adj_r2': 0.84,
            'n': 49,
        },
    }

    total_points = 0
    max_points = 0
    details = []

    for model_name, gt in ground_truth.items():
        if results.get(model_name) is None:
            details.append(f"{model_name}: ERROR - no results")
            max_points += 20  # rough estimate of points for this model
            continue

        res, n = results[model_name]

        # Score coefficients (25 pts total across all models)
        vars_in_model = [k for k in gt.keys() if k not in ('adj_r2', 'n')]
        for var in vars_in_model:
            max_points += 25.0 / 20  # ~20 coefficients total
            gen_coef = res.params.get(var, np.nan)
            true_coef = gt[var]['coef']
            if abs(gen_coef - true_coef) <= 0.05:
                total_points += 25.0 / 20
                details.append(f"  {model_name} {var} coef: {gen_coef:.3f} vs {true_coef:.3f} MATCH")
            elif abs(gen_coef - true_coef) <= 0.15:
                total_points += 25.0 / 40
                details.append(f"  {model_name} {var} coef: {gen_coef:.3f} vs {true_coef:.3f} PARTIAL")
            else:
                details.append(f"  {model_name} {var} coef: {gen_coef:.3f} vs {true_coef:.3f} MISS")

        # Score SEs (15 pts total)
        for var in vars_in_model:
            max_points += 15.0 / 20
            gen_se = res.bse.get(var, np.nan)
            true_se = gt[var]['se']
            if abs(gen_se - true_se) <= 0.02:
                total_points += 15.0 / 20
                details.append(f"  {model_name} {var} SE: {gen_se:.3f} vs {true_se:.3f} MATCH")
            elif abs(gen_se - true_se) <= 0.05:
                total_points += 15.0 / 40
                details.append(f"  {model_name} {var} SE: {gen_se:.3f} vs {true_se:.3f} PARTIAL")
            else:
                details.append(f"  {model_name} {var} SE: {gen_se:.3f} vs {true_se:.3f} MISS")

        # Score N (15 pts total, ~2.5 per model)
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

        # Score significance (25 pts total)
        for var in vars_in_model:
            max_points += 25.0 / 20
            gen_pval = res.pvalues.get(var, np.nan)
            gen_sig = ''
            if gen_pval < 0.001:
                gen_sig = '***'
            elif gen_pval < 0.01:
                gen_sig = '**'
            elif gen_pval < 0.05:
                gen_sig = '*'
            true_sig = gt[var]['sig']
            if gen_sig == true_sig:
                total_points += 25.0 / 20
                details.append(f"  {model_name} {var} sig: {gen_sig} vs {true_sig} MATCH")
            else:
                details.append(f"  {model_name} {var} sig: {gen_sig} vs {true_sig} MISS")

        # Score R² (10 pts total, ~1.67 per model)
        max_points += 1.67
        r2_diff = abs(res.rsquared_adj - gt['adj_r2'])
        if r2_diff <= 0.02:
            total_points += 1.67
            details.append(f"  {model_name} Adj R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} MATCH")
        elif r2_diff <= 0.05:
            total_points += 0.83
            details.append(f"  {model_name} Adj R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} PARTIAL")
        else:
            details.append(f"  {model_name} Adj R²: {res.rsquared_adj:.2f} vs {gt['adj_r2']:.2f} MISS")

    # Variable presence (10 pts)
    max_points += 10
    all_present = True
    for model_name, gt in ground_truth.items():
        if results.get(model_name) is not None:
            total_points += 10 / 6
        else:
            all_present = False

    score = min(100, int(100 * total_points / max_points)) if max_points > 0 else 0

    print("\n" + "=" * 80)
    print(f"SCORING: {score}/100")
    print("=" * 80)
    for d in details:
        print(d)

    return score


if __name__ == "__main__":
    results = run_analysis()
    score = score_against_ground_truth(results)
