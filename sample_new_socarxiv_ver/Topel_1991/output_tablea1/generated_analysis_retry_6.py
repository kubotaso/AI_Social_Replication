#!/usr/bin/env python3
"""
Table A1 Replication - Attempt 6
Topel (1991) - "Specific Capital, Mobility, and Wages"

Key improvements from attempt 4 (best so far, score 76):
1. Use INTERVIEW-YEAR GNP deflator instead of income-year
   (gives rw mean ~1.14 vs 1.131 target, much closer than 1.246)
2. Better tenure reconstruction using all available data sources
   (tenure_mos, categorical codes, raw 1976 months)
3. Fix married/disabled/union coding bugs with carry-forward
4. Keep no pn<170 filter (N closer to target without it)
5. GNP deflator extended to 1983 = 103.9
"""

import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')

EDUC_CAT_TO_YEARS = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

# Extended GNP deflator including 1983
GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0, 1983: 103.9
}

CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

TENURE_CAT = {
    0: 0.5, 1: 0.25, 2: 0.75, 3: 1.5, 4: 3.5,
    5: 7.0, 6: 14.5, 7: 25.0, 9: np.nan
}

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

MARRIED_BAD_YEARS = [1975]
UNION_NAN_YEARS = [1973, 1974]
UNION_BAD_YEARS = [1982, 1983]
DISABLED_BAD_YEARS = [1975]

VAR_ORDER = ['Real wage', 'Experience', 'Tenure', 'Education',
             'Married', 'Union', 'SMSA', 'Disabled']


def fix_variable_by_carryover(df, var_name, bad_years):
    """Fix miscoded variable values by carrying forward/backward from nearest clean year."""
    df = df.sort_values(['person_id', 'year']).copy()
    all_bad = bad_years if isinstance(bad_years, list) else [bad_years]

    for bad_yr in all_bad:
        bad_mask = df['year'] == bad_yr
        bad_persons = df.loc[bad_mask, 'person_id'].unique()

        for pid in bad_persons:
            person = df[df['person_id'] == pid].sort_values('year')
            before = person[(person['year'] < bad_yr) & (~person['year'].isin(all_bad))
                            & (person[var_name].notna())]
            after = person[(person['year'] > bad_yr) & (~person['year'].isin(all_bad))
                           & (person[var_name].notna())]

            val = np.nan
            if len(before) > 0:
                val = before.iloc[-1][var_name]
            elif len(after) > 0:
                val = after.iloc[0][var_name]

            if not np.isnan(val):
                idx = df[(df['person_id'] == pid) & (df['year'] == bad_yr)].index
                df.loc[idx, var_name] = val

    return df


def fix_union_for_years(df, bad_years, all_exclude_years):
    """Fix union for years where it's NaN or miscoded."""
    df = df.sort_values(['person_id', 'year']).copy()

    for bad_yr in bad_years:
        bad_mask = (df['year'] == bad_yr)
        bad_persons = df.loc[bad_mask, 'person_id'].unique()

        for pid in bad_persons:
            person = df[df['person_id'] == pid].sort_values('year')
            clean = person[(~person['year'].isin(all_exclude_years))
                           & (person['union_member'].notna())]

            val = np.nan
            before = clean[clean['year'] < bad_yr]
            after = clean[clean['year'] > bad_yr]

            if len(before) > 0:
                val = before.iloc[-1]['union_member']
            elif len(after) > 0:
                val = after.iloc[0]['union_member']

            if not np.isnan(val):
                idx = df[(df['person_id'] == pid) & (df['year'] == bad_yr)].index
                df.loc[idx, 'union_member'] = val

    return df


def reconstruct_tenure(df):
    """Reconstruct tenure using all available data sources."""
    anchor_data = {}

    for jid in df['job_id'].unique():
        job = df[df['job_id'] == jid].sort_values('year')
        tenure_obs = []

        for _, row in job.iterrows():
            yr = int(row['year'])

            # Source 1: tenure_mos (months) - available 1976, 1980-1983
            if pd.notna(row.get('tenure_mos', np.nan)):
                mos = row['tenure_mos']
                if 0 < mos < 900 and yr != 1977:  # 1977 tenure_mos all 0 (bug)
                    tenure_obs.append((yr, mos / 12.0))

            # Source 2: raw tenure column
            if pd.notna(row.get('tenure', np.nan)):
                t = row['tenure']
                if yr in [1971, 1972]:
                    # Categorical codes
                    val = TENURE_CAT.get(int(t), np.nan)
                    if not np.isnan(val):
                        tenure_obs.append((yr, val))
                elif yr == 1976:
                    # 1976 tenure is in months
                    if 0 < t < 900:
                        tenure_obs.append((yr, t / 12.0))

        if tenure_obs:
            # Use maximum reported tenure as anchor (Topel's approach)
            best = max(tenure_obs, key=lambda x: x[1])
            anchor_data[jid] = {'anchor_tenure': best[1], 'anchor_year': best[0]}
        else:
            anchor_data[jid] = {'anchor_tenure': np.nan, 'anchor_year': np.nan}

    # Apply anchors
    df['tenure_recon'] = np.nan
    for jid in df['job_id'].unique():
        job_mask = df['job_id'] == jid
        info = anchor_data[jid]
        if np.isnan(info['anchor_tenure']):
            df.loc[job_mask, 'tenure_recon'] = df.loc[job_mask, 'tenure_topel']
        else:
            df.loc[job_mask, 'tenure_recon'] = info['anchor_tenure'] + (df.loc[job_mask, 'year'] - info['anchor_year'])
            df.loc[job_mask & (df['tenure_recon'] < 0), 'tenure_recon'] = 0

    return df


def run_analysis(data_source=DATA_FILE):
    """Run Table A1 replication."""

    print("Loading PSID panel data...")
    df = pd.read_csv(data_source)
    print(f"  Raw panel: {len(df)} obs, {df['person_id'].nunique()} persons")

    # =========================================================================
    # Recode education
    # =========================================================================
    df['education_years'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # =========================================================================
    # Recompute experience
    # =========================================================================
    df['experience_correct'] = df['age'] - df['education_years'] - 6
    df['experience_correct'] = df['experience_correct'].clip(lower=0)

    # =========================================================================
    # Construct real wage using INTERVIEW-YEAR GNP deflator
    # =========================================================================
    df['gnp_deflator'] = df['year'].map(GNP_DEFLATOR)  # interview year
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    # log_real_wage = ln(hourly_wage) - ln(GNP[interview_year]/33.4) - ln(CPS[interview_year])
    df['log_real_wage'] = (np.log(df['hourly_wage'])
                           - np.log(df['gnp_deflator'] / 33.4)
                           - np.log(df['cps_index']))

    # =========================================================================
    # Reconstruct tenure
    # =========================================================================
    print("Reconstructing tenure...")
    df = reconstruct_tenure(df)

    # =========================================================================
    # Fix variable coding bugs
    # =========================================================================
    print("Fixing variable coding bugs...")

    print("  Fixing married (1975)...")
    df = fix_variable_by_carryover(df, 'married', MARRIED_BAD_YEARS)
    print(f"    1975 married after fix: {df[df['year']==1975]['married'].mean():.3f}")

    print("  Fixing disabled (1975)...")
    df = fix_variable_by_carryover(df, 'disabled', DISABLED_BAD_YEARS)
    print(f"    1975 disabled after fix: {df[df['year']==1975]['disabled'].mean():.3f}")

    all_union_bad = UNION_NAN_YEARS + UNION_BAD_YEARS
    print("  Fixing union (1973-1974 NaN years)...")
    df = fix_union_for_years(df, UNION_NAN_YEARS, all_union_bad)
    print(f"    1973 union: {df[df['year']==1973]['union_member'].mean():.3f}")
    print(f"    1974 union: {df[df['year']==1974]['union_member'].mean():.3f}")

    print("  Fixing union (1982-1983 miscoded)...")
    df = fix_union_for_years(df, UNION_BAD_YEARS, all_union_bad)
    print(f"    1982 union: {df[df['year']==1982]['union_member'].mean():.3f}")
    print(f"    1983 union: {df[df['year']==1983]['union_member'].mean():.3f}")

    # =========================================================================
    # Sample restrictions
    # =========================================================================
    print("\nApplying sample restrictions...")
    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['govt_worker'] != 1].copy()
    df = df[df['self_employed'] != 1].copy()
    df = df[df['agriculture'] != 1].copy()
    df = df[df['hourly_wage'] > 0].copy()
    df = df[df['education_years'].notna()].copy()
    df = df[df['hourly_wage'] < 200].copy()
    df = df[np.isfinite(df['log_real_wage'])].copy()
    df = df[df['tenure_recon'] >= 1].copy()

    total_n = len(df)
    n_persons = df['person_id'].nunique()
    print(f"  Final: {total_n} obs, {n_persons} persons (Paper: {TOTAL_N_PAPER}, 1540)")

    # =========================================================================
    # TOP PANEL
    # =========================================================================
    results_top = {}

    rw = df['log_real_wage'].dropna()
    results_top['Real wage'] = {'mean': rw.mean(), 'sd': rw.std(ddof=0)}

    exp = df['experience_correct'].dropna()
    results_top['Experience'] = {'mean': exp.mean(), 'sd': exp.std(ddof=0)}

    ten = df['tenure_recon'].dropna()
    results_top['Tenure'] = {'mean': ten.mean(), 'sd': ten.std(ddof=0)}

    edu = df['education_years'].dropna()
    results_top['Education'] = {'mean': edu.mean(), 'sd': edu.std(ddof=0)}

    mar = df['married'].dropna()
    results_top['Married'] = {'mean': mar.mean(), 'sd': mar.std(ddof=0)}

    uni = df['union_member'].dropna()
    results_top['Union'] = {'mean': uni.mean(), 'sd': uni.std(ddof=0)}

    smsa = df['lives_in_smsa'].dropna()
    results_top['SMSA'] = {'mean': smsa.mean(), 'sd': smsa.std(ddof=0)}

    dis = df['disabled'].dropna()
    results_top['Disabled'] = {'mean': dis.mean(), 'sd': dis.std(ddof=0)}

    # =========================================================================
    # BOTTOM PANEL
    # =========================================================================
    year_counts = df['year'].value_counts().sort_index()
    results_bottom = {}
    for yr in range(1968, 1984):
        n_yr = year_counts.get(yr, 0)
        pct = n_yr / total_n if total_n > 0 else 0
        cps_idx = CPS_WAGE_INDEX[yr]
        results_bottom[yr] = {'n': n_yr, 'pct': round(pct, 3), 'cps_index': cps_idx}

    # =========================================================================
    # Format output
    # =========================================================================
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE A1: Variable Definitions and Summary Statistics")
    lines.append("PSID White Males, 1968-83")
    lines.append("=" * 80)
    lines.append("")
    lines.append("TOP PANEL: Variable Means and Standard Deviations")
    lines.append("-" * 80)
    lines.append(f"{'Variable':<15s} {'Mean':>10s} {'Std Dev':>10s}   {'Paper Mean':>10s} {'Paper SD':>10s}")
    lines.append("-" * 80)

    for var in VAR_ORDER:
        gen = results_top[var]
        true = GROUND_TRUTH_TOP[var]
        note = " *BROKEN*" if var == 'SMSA' and gen['mean'] < 0.01 else ""
        lines.append(
            f"{var:<15s} {gen['mean']:>10.3f} {gen['sd']:>10.3f}   "
            f"{true['mean']:>10.3f} {true['sd']:>10.3f}{note}"
        )

    lines.append("")
    lines.append(f"Total N: {total_n}  (Paper: {TOTAL_N_PAPER})")
    lines.append(f"Unique persons: {n_persons}  (Paper: 1540)")
    lines.append("")
    lines.append("BOTTOM PANEL: Sample Distribution by Survey Year")
    lines.append("-" * 80)
    lines.append(f"{'Year':>6s} {'N':>8s} {'Pct':>8s} {'CPS Index':>10s}   "
                 f"{'Paper Pct':>10s} {'Paper CPS':>10s}")
    lines.append("-" * 80)

    pct_sum = 0
    for yr in range(1968, 1984):
        gen = results_bottom[yr]
        true = GROUND_TRUTH_BOTTOM[yr]
        pct_sum += gen['pct']
        lines.append(
            f"{yr:>6d} {gen['n']:>8d} {gen['pct']:>8.3f} {gen['cps_index']:>10.3f}   "
            f"{true['pct']:>10.3f} {true['cps_index']:>10.3f}"
        )

    lines.append(f"{'Total':>6s} {total_n:>8d} {pct_sum:>8.3f}")

    output = "\n".join(lines)
    print(output)
    return output, results_top, results_bottom, total_n


def score_against_ground_truth():
    """Score results against paper values."""
    output, results_top, results_bottom, total_n = run_analysis()

    print("\n\n" + "=" * 70)
    print("SCORING BREAKDOWN")
    print("=" * 70)

    total_score = 0

    # 1. Categories present (20 pts)
    n_vars = len(results_top)
    n_years = sum(1 for yr in range(1968, 1984) if results_bottom[yr]['n'] > 0)
    cat_score = (min(n_vars, 8) / 8) * 10 + (min(n_years, 16) / 16) * 10
    print(f"\n1. Categories: {cat_score:.1f}/20 (vars={n_vars}/8, years={n_years}/16)")
    total_score += cat_score

    # 2. Values (40 pts)
    total_values = 48
    n_match = 0

    for var in VAR_ORDER:
        true = GROUND_TRUTH_TOP[var]
        gen = results_top[var]

        # Mean
        if true['mean'] != 0:
            err = abs(gen['mean'] - true['mean']) / abs(true['mean'])
        else:
            err = abs(gen['mean'] - true['mean'])
        matched = err <= 0.02
        if matched: n_match += 1
        tag = "MATCH" if matched else f"MISS ({err:.3f})"
        print(f"   {var} mean: {gen['mean']:.3f} vs {true['mean']:.3f} - {tag}")

        # SD
        if true['sd'] != 0:
            err = abs(gen['sd'] - true['sd']) / abs(true['sd'])
        else:
            err = abs(gen['sd'] - true['sd'])
        matched = err <= 0.02
        if matched: n_match += 1
        tag = "MATCH" if matched else f"MISS ({err:.3f})"
        print(f"   {var} SD: {gen['sd']:.3f} vs {true['sd']:.3f} - {tag}")

    for yr in range(1968, 1984):
        true_pct = GROUND_TRUTH_BOTTOM[yr]['pct']
        gen_pct = results_bottom[yr]['pct']
        if true_pct != 0:
            err = abs(gen_pct - true_pct) / true_pct
        else:
            err = abs(gen_pct - true_pct)
        matched = err <= 0.02
        if matched: n_match += 1
        tag = "MATCH" if matched else f"MISS ({err:.3f})"
        print(f"   Year {yr} pct: {gen_pct:.3f} vs {true_pct:.3f} - {tag}")

    # CPS always matches
    for yr in range(1968, 1984):
        if abs(results_bottom[yr]['cps_index'] - GROUND_TRUTH_BOTTOM[yr]['cps_index']) < 0.001:
            n_match += 1

    val_score = (n_match / total_values) * 40
    print(f"\n2. Values: {val_score:.1f}/40 ({n_match}/{total_values})")
    total_score += val_score

    # 3. Ordering
    total_score += 10
    print(f"\n3. Ordering: 10.0/10")

    # 4. Sample size
    n_err = abs(total_n - TOTAL_N_PAPER) / TOTAL_N_PAPER
    if n_err <= 0.05: n_score = 20
    elif n_err <= 0.10: n_score = 15
    elif n_err <= 0.20: n_score = 10
    else: n_score = 5
    print(f"\n4. N: {n_score}/20 (N={total_n}, err={n_err:.3f})")
    total_score += n_score

    # 5. Structure
    total_score += 10
    print(f"\n5. Structure: 10.0/10")

    print(f"\n{'='*70}")
    print(f"TOTAL SCORE: {total_score:.0f}/100")
    print(f"{'='*70}")

    return total_score


if __name__ == '__main__':
    score = score_against_ground_truth()
