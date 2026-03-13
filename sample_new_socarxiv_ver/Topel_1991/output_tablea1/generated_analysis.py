#!/usr/bin/env python3
"""
Table A1 Replication: Variable Definitions and Summary Statistics
Topel (1991) - "Specific Capital, Mobility, and Wages"

This is a descriptive table with two panels:
  Top: Means and SDs for 8 variables
  Bottom: Sample distribution by year + CPS wage index
"""

import numpy as np
import pandas as pd
import os

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')

# Education category -> years mapping (for years with categorical coding)
EDUC_CAT_TO_YEARS = {
    0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17
}

# GNP Price Deflator for Consumption Expenditure (base 1982=100)
GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

# CPS Wage Index (from Murphy and Welch 1987)
CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

# Ground truth values from the paper
GROUND_TRUTH_TOP = {
    'Real wage':   {'mean': 1.131, 'sd': 0.497},
    'Experience':  {'mean': 20.021, 'sd': 11.045},
    'Tenure':      {'mean': 9.978, 'sd': 8.944},
    'Education':   {'mean': 12.645, 'sd': 2.809},
    'Married':     {'mean': 0.925, 'sd': 0.263},
    'Union':       {'mean': 0.344, 'sd': 0.473},
    'SMSA':        {'mean': 0.644, 'sd': 0.478},
    'Disabled':    {'mean': 0.074, 'sd': 0.262},
}

GROUND_TRUTH_BOTTOM = {
    1968: {'pct': 0.052, 'cps_index': 1.000},
    1969: {'pct': 0.050, 'cps_index': 1.032},
    1970: {'pct': 0.051, 'cps_index': 1.091},
    1971: {'pct': 0.053, 'cps_index': 1.115},
    1972: {'pct': 0.057, 'cps_index': 1.113},
    1973: {'pct': 0.058, 'cps_index': 1.151},
    1974: {'pct': 0.060, 'cps_index': 1.167},
    1975: {'pct': 0.061, 'cps_index': 1.188},
    1976: {'pct': 0.065, 'cps_index': 1.117},
    1977: {'pct': 0.065, 'cps_index': 1.121},
    1978: {'pct': 0.069, 'cps_index': 1.133},
    1979: {'pct': 0.071, 'cps_index': 1.128},
    1980: {'pct': 0.073, 'cps_index': 1.128},
    1981: {'pct': 0.072, 'cps_index': 1.109},
    1982: {'pct': 0.071, 'cps_index': 1.103},
    1983: {'pct': 0.068, 'cps_index': 1.089},
}

TOTAL_N_PAPER = 13128


def run_analysis(data_source=DATA_FILE):
    """Run Table A1 replication."""

    # =========================================================================
    # Load data
    # =========================================================================
    print("Loading PSID panel data...")
    df = pd.read_csv(data_source)
    print(f"  Raw panel: {len(df)} observations, {df['person_id'].nunique()} persons")
    print(f"  Year range: {df['year'].min()} - {df['year'].max()}")

    # =========================================================================
    # Recode education
    # =========================================================================
    # Years 1975-1976: education_clean is already in years (0-17)
    # All other years: education_clean is categorical (0-8), needs mapping
    df['education_years'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # =========================================================================
    # Recompute experience with correct education
    # =========================================================================
    df['experience_correct'] = df['age'] - df['education_years'] - 6
    df['experience_correct'] = df['experience_correct'].clip(lower=0)

    # =========================================================================
    # Construct real wage
    # =========================================================================
    # PSID interview year Y reports income for year Y-1
    # So for interview year Y, use GNP deflator for year Y-1
    df['income_year'] = df['year'] - 1
    df['gnp_deflator'] = df['income_year'].map(GNP_DEFLATOR)
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)

    # log_real_wage = ln(hourly_wage) - ln(GNP_deflator/100) - ln(CPS_index)
    df['log_real_wage'] = (np.log(df['hourly_wage'])
                           - np.log(df['gnp_deflator'] / 100.0)
                           - np.log(df['cps_index']))

    # =========================================================================
    # Sample restrictions
    # =========================================================================
    print("\nApplying sample restrictions...")
    print(f"  Before restrictions: {len(df)}")

    # Age 18-60 (already filtered in panel construction)
    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    print(f"  After age 18-60: {len(df)}")

    # Exclude government workers
    df = df[df['govt_worker'] != 1].copy()
    print(f"  After excluding govt workers: {len(df)}")

    # Exclude self-employed
    df = df[df['self_employed'] != 1].copy()
    print(f"  After excluding self-employed: {len(df)}")

    # Exclude agriculture
    df = df[df['agriculture'] != 1].copy()
    print(f"  After excluding agriculture: {len(df)}")

    # Tenure >= 1 year
    df = df[df['tenure_topel'] >= 1].copy()
    print(f"  After tenure >= 1: {len(df)}")

    # Positive earnings
    df = df[df['hourly_wage'] > 0].copy()
    print(f"  After positive wages: {len(df)}")

    # Drop missing education
    df = df[df['education_years'].notna()].copy()
    print(f"  After dropping missing education: {len(df)}")

    # Drop extreme wages (outliers)
    # Top-code at reasonable threshold
    df = df[df['hourly_wage'] < 200].copy()
    print(f"  After removing extreme wages: {len(df)}")

    total_n = len(df)
    print(f"\n  Final sample: {total_n} observations, {df['person_id'].nunique()} persons")
    print(f"  Paper target: {TOTAL_N_PAPER} observations, 1540 persons")

    # =========================================================================
    # TOP PANEL: Compute means and standard deviations
    # =========================================================================
    results_top = {}

    # 1. Real wage (log)
    real_wage = df['log_real_wage'].dropna()
    results_top['Real wage'] = {'mean': real_wage.mean(), 'sd': real_wage.std()}

    # 2. Experience
    exp = df['experience_correct'].dropna()
    results_top['Experience'] = {'mean': exp.mean(), 'sd': exp.std()}

    # 3. Tenure
    ten = df['tenure_topel'].dropna()
    results_top['Tenure'] = {'mean': ten.mean(), 'sd': ten.std()}

    # 4. Education
    edu = df['education_years'].dropna()
    results_top['Education'] = {'mean': edu.mean(), 'sd': edu.std()}

    # 5. Married
    mar = df['married'].dropna()
    results_top['Married'] = {'mean': mar.mean(), 'sd': mar.std()}

    # 6. Union
    uni = df['union_member'].dropna()
    results_top['Union'] = {'mean': uni.mean(), 'sd': uni.std()}

    # 7. SMSA (known broken - all zeros in panel)
    # Since the variable is broken, we report what we have but note the issue
    smsa = df['lives_in_smsa'].dropna()
    results_top['SMSA'] = {'mean': smsa.mean(), 'sd': smsa.std()}

    # 8. Disabled
    dis = df['disabled'].dropna()
    results_top['Disabled'] = {'mean': dis.mean(), 'sd': dis.std()}

    # =========================================================================
    # BOTTOM PANEL: Sample distribution by year
    # =========================================================================
    year_counts = df['year'].value_counts().sort_index()
    results_bottom = {}
    for yr in range(1968, 1984):
        n_yr = year_counts.get(yr, 0)
        pct = n_yr / total_n if total_n > 0 else 0
        cps_idx = CPS_WAGE_INDEX[yr]
        results_bottom[yr] = {'n': n_yr, 'pct': pct, 'cps_index': cps_idx}

    # =========================================================================
    # Format and print results
    # =========================================================================
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TABLE A1: Variable Definitions and Summary Statistics")
    output_lines.append("PSID White Males, 1968-83")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("TOP PANEL: Variable Means and Standard Deviations")
    output_lines.append("-" * 80)
    output_lines.append(f"{'Variable':<15s} {'Mean':>10s} {'Std Dev':>10s}   {'Paper Mean':>10s} {'Paper SD':>10s}")
    output_lines.append("-" * 80)

    var_order = ['Real wage', 'Experience', 'Tenure', 'Education',
                 'Married', 'Union', 'SMSA', 'Disabled']

    for var in var_order:
        gen = results_top[var]
        true = GROUND_TRUTH_TOP[var]
        note = " (BROKEN)" if var == 'SMSA' and gen['mean'] == 0.0 else ""
        output_lines.append(
            f"{var:<15s} {gen['mean']:>10.3f} {gen['sd']:>10.3f}   "
            f"{true['mean']:>10.3f} {true['sd']:>10.3f}{note}"
        )

    output_lines.append("")
    output_lines.append(f"Total N: {total_n}  (Paper: {TOTAL_N_PAPER})")
    output_lines.append(f"Unique persons: {df['person_id'].nunique()}  (Paper: 1540)")
    output_lines.append("")
    output_lines.append("BOTTOM PANEL: Sample Distribution by Survey Year")
    output_lines.append("-" * 80)
    output_lines.append(f"{'Year':>6s} {'N':>8s} {'Pct':>8s} {'CPS Index':>10s}   "
                        f"{'Paper Pct':>10s} {'Paper CPS':>10s}")
    output_lines.append("-" * 80)

    pct_sum = 0
    for yr in range(1968, 1984):
        gen = results_bottom[yr]
        true = GROUND_TRUTH_BOTTOM[yr]
        pct_sum += gen['pct']
        output_lines.append(
            f"{yr:>6d} {gen['n']:>8d} {gen['pct']:>8.3f} {gen['cps_index']:>10.3f}   "
            f"{true['pct']:>10.3f} {true['cps_index']:>10.3f}"
        )

    output_lines.append(f"{'Total':>6s} {total_n:>8d} {pct_sum:>8.3f}")
    output_lines.append("")

    output = "\n".join(output_lines)
    print(output)
    return output, results_top, results_bottom, total_n


def score_against_ground_truth():
    """Score the results against ground truth from Table A1."""
    output, results_top, results_bottom, total_n = run_analysis()

    print("\n\n" + "=" * 70)
    print("SCORING BREAKDOWN")
    print("=" * 70)

    total_score = 0

    # -------------------------------------------------------------------------
    # 1. Categories present (20 pts): All 8 variables + 16 years
    # -------------------------------------------------------------------------
    cat_max = 20
    cat_score = 0
    n_vars_present = len(results_top)  # should be 8
    n_years_present = sum(1 for yr in range(1968, 1984) if results_bottom[yr]['n'] > 0)

    # 8 variables: 10 pts, 16 years: 10 pts
    cat_score += (min(n_vars_present, 8) / 8) * 10
    cat_score += (min(n_years_present, 16) / 16) * 10
    print(f"\n1. Categories present: {cat_score:.1f}/{cat_max}")
    print(f"   Variables present: {n_vars_present}/8")
    print(f"   Years with data: {n_years_present}/16")
    total_score += cat_score

    # -------------------------------------------------------------------------
    # 2. Count/percentage values (40 pts): within 2% of true values
    # -------------------------------------------------------------------------
    val_max = 40
    val_score = 0
    n_values = 0
    n_match = 0

    # Top panel: 8 means + 8 SDs = 16 values
    # Bottom panel: 16 pcts + 16 CPS indices = 32 values
    # Total: 48 values
    total_values = 48

    # Check means
    for var in ['Real wage', 'Experience', 'Tenure', 'Education',
                'Married', 'Union', 'SMSA', 'Disabled']:
        true_mean = GROUND_TRUTH_TOP[var]['mean']
        true_sd = GROUND_TRUTH_TOP[var]['sd']
        gen_mean = results_top[var]['mean']
        gen_sd = results_top[var]['sd']

        # Mean check
        n_values += 1
        if true_mean != 0:
            pct_err_mean = abs(gen_mean - true_mean) / abs(true_mean)
        else:
            pct_err_mean = abs(gen_mean - true_mean)

        if pct_err_mean <= 0.02:
            n_match += 1
            print(f"   {var} mean: {gen_mean:.3f} vs {true_mean:.3f} - MATCH ({pct_err_mean:.3f})")
        else:
            print(f"   {var} mean: {gen_mean:.3f} vs {true_mean:.3f} - MISS ({pct_err_mean:.3f})")

        # SD check
        n_values += 1
        if true_sd != 0:
            pct_err_sd = abs(gen_sd - true_sd) / abs(true_sd)
        else:
            pct_err_sd = abs(gen_sd - true_sd)

        if pct_err_sd <= 0.02:
            n_match += 1
            print(f"   {var} SD: {gen_sd:.3f} vs {true_sd:.3f} - MATCH ({pct_err_sd:.3f})")
        else:
            print(f"   {var} SD: {gen_sd:.3f} vs {true_sd:.3f} - MISS ({pct_err_sd:.3f})")

    # Bottom panel percentages
    for yr in range(1968, 1984):
        true_pct = GROUND_TRUTH_BOTTOM[yr]['pct']
        gen_pct = results_bottom[yr]['pct']

        n_values += 1
        if true_pct != 0:
            pct_err = abs(gen_pct - true_pct) / abs(true_pct)
        else:
            pct_err = abs(gen_pct - true_pct)

        if pct_err <= 0.02:
            n_match += 1
            print(f"   Year {yr} pct: {gen_pct:.3f} vs {true_pct:.3f} - MATCH")
        else:
            print(f"   Year {yr} pct: {gen_pct:.3f} vs {true_pct:.3f} - MISS ({pct_err:.3f})")

    # CPS index (hardcoded, should always match)
    for yr in range(1968, 1984):
        true_cps = GROUND_TRUTH_BOTTOM[yr]['cps_index']
        gen_cps = results_bottom[yr]['cps_index']

        n_values += 1
        if abs(gen_cps - true_cps) < 0.001:
            n_match += 1
        else:
            print(f"   Year {yr} CPS: {gen_cps:.3f} vs {true_cps:.3f} - MISS")

    val_score = (n_match / total_values) * val_max
    print(f"\n2. Count/percentage values: {val_score:.1f}/{val_max}")
    print(f"   Matched: {n_match}/{total_values}")
    total_score += val_score

    # -------------------------------------------------------------------------
    # 3. Ordering (10 pts)
    # -------------------------------------------------------------------------
    ord_max = 10
    ord_score = 10  # Variables and years are in correct order by construction
    print(f"\n3. Ordering: {ord_score:.1f}/{ord_max}")
    total_score += ord_score

    # -------------------------------------------------------------------------
    # 4. Sample size N (20 pts): within 5%
    # -------------------------------------------------------------------------
    n_max = 20
    n_pct_err = abs(total_n - TOTAL_N_PAPER) / TOTAL_N_PAPER
    if n_pct_err <= 0.05:
        n_score = 20
    elif n_pct_err <= 0.10:
        n_score = 15
    elif n_pct_err <= 0.20:
        n_score = 10
    else:
        n_score = 5
    print(f"\n4. Sample size: {n_score:.1f}/{n_max}")
    print(f"   Generated N: {total_n}, Paper N: {TOTAL_N_PAPER}, Error: {n_pct_err:.3f}")
    total_score += n_score

    # -------------------------------------------------------------------------
    # 5. Column structure (10 pts)
    # -------------------------------------------------------------------------
    col_max = 10
    col_score = 10  # We have Variable, Mean, SD for top; Year, Pct, CPS for bottom
    print(f"\n5. Column structure: {col_score:.1f}/{col_max}")
    total_score += col_score

    print(f"\n{'='*70}")
    print(f"TOTAL SCORE: {total_score:.0f}/100")
    print(f"{'='*70}")

    return total_score


if __name__ == '__main__':
    score = score_against_ground_truth()
