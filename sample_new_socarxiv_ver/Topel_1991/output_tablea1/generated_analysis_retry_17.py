#!/usr/bin/env python3
"""
Table A1 Replication - Attempt 17
Topel (1991) - "Specific Capital, Mobility, and Wages"

KEY FIX: Corrected TENURE_CAT mapping for 1971-1972 categorical tenure codes.
The PSID 1971-1972 "years on job" variable uses brackets:
  0 = less than one year
  3 = one or two years (1-2)     -> midpoint 1.5
  4 = three through nine years   -> midpoint 6.0  (was 3.5!)
  5 = ten through nineteen years -> midpoint 14.5 (was 7.0!)
  6 = twenty or more years       -> midpoint 25.0 (was 14.5!)
  9 = NA/DK

Our previous mapping severely underestimated tenure for codes 4, 5, 6,
causing the overall tenure mean to be ~9.7 instead of ~10.0.
"""

import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')

EDUC_CAT_TO_YEARS = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

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

# CORRECTED TENURE CATEGORY MAPPING
# PSID 1971-1972 "years on job" brackets:
#   0 = less than one year
#   3 = one or two years (1-2)     -> midpoint 1.5
#   4 = three through nine years (3-9) -> midpoint 6.0
#   5 = ten through nineteen years (10-19) -> midpoint 14.5
#   6 = twenty or more years (20+) -> midpoint 25.0 (assume ~20-30)
#   9 = NA/DK
TENURE_CAT = {
    0: 0.5,
    3: 1.5,
    4: 6.0,     # was 3.5 -- CORRECTED
    5: 14.5,    # was 7.0 -- CORRECTED
    6: 25.0,    # was 14.5 -- CORRECTED
    9: np.nan
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
    df = df.sort_values(['person_id', 'year']).copy()
    all_bad = bad_years if isinstance(bad_years, list) else [bad_years]
    for bad_yr in all_bad:
        for pid in df.loc[df['year'] == bad_yr, 'person_id'].unique():
            person = df[df['person_id'] == pid].sort_values('year')
            before = person[(person['year'] < bad_yr) & (~person['year'].isin(all_bad)) & person[var_name].notna()]
            after = person[(person['year'] > bad_yr) & (~person['year'].isin(all_bad)) & person[var_name].notna()]
            val = before.iloc[-1][var_name] if len(before) > 0 else (after.iloc[0][var_name] if len(after) > 0 else np.nan)
            if pd.notna(val):
                df.loc[(df['person_id'] == pid) & (df['year'] == bad_yr), var_name] = val
    return df


def fix_union_for_years(df, bad_years, all_exclude):
    df = df.sort_values(['person_id', 'year']).copy()
    for bad_yr in bad_years:
        for pid in df.loc[df['year'] == bad_yr, 'person_id'].unique():
            person = df[df['person_id'] == pid].sort_values('year')
            clean = person[(~person['year'].isin(all_exclude)) & person['union_member'].notna()]
            before = clean[clean['year'] < bad_yr]
            after = clean[clean['year'] > bad_yr]
            val = before.iloc[-1]['union_member'] if len(before) > 0 else (after.iloc[0]['union_member'] if len(after) > 0 else np.nan)
            if pd.notna(val):
                df.loc[(df['person_id'] == pid) & (df['year'] == bad_yr), 'union_member'] = val
    return df


def reconstruct_tenure(df):
    """Reconstruct tenure using anchor points with CORRECTED category mapping."""
    anchor_data = {}
    for jid in df['job_id'].unique():
        job = df[df['job_id'] == jid].sort_values('year')
        obs = []
        for _, row in job.iterrows():
            yr = int(row['year'])
            # tenure_mos: available 1976, 1980-1983 (months on current job)
            if pd.notna(row.get('tenure_mos', np.nan)):
                mos = row['tenure_mos']
                if 0 < mos < 900 and yr != 1977:
                    obs.append((yr, mos / 12.0))
            # tenure: categorical for 1971-1972, months for 1976
            if pd.notna(row.get('tenure', np.nan)):
                t = row['tenure']
                if yr in [1971, 1972]:
                    val = TENURE_CAT.get(int(t), np.nan)
                    if pd.notna(val):
                        obs.append((yr, val))
                elif yr == 1976:
                    if 0 < t < 900:
                        obs.append((yr, t / 12.0))
        if obs:
            # Paper: "starting tenure gauged from period of maximum reported tenure"
            best = max(obs, key=lambda x: x[1])
            anchor_data[jid] = (best[1], best[0])
        else:
            anchor_data[jid] = (np.nan, np.nan)

    df['tenure_recon'] = np.nan
    for jid in df['job_id'].unique():
        mask = df['job_id'] == jid
        at, ay = anchor_data[jid]
        if np.isnan(at):
            df.loc[mask, 'tenure_recon'] = df.loc[mask, 'tenure_topel']
        else:
            df.loc[mask, 'tenure_recon'] = at + (df.loc[mask, 'year'] - ay)
            df.loc[mask & (df['tenure_recon'] < 0), 'tenure_recon'] = 0
    return df


def compute_job_union(df):
    clean_mask = (~df['year'].isin(UNION_NAN_YEARS + UNION_BAD_YEARS)) & df['union_member'].notna()
    df_clean = df[clean_mask]
    uj = df_clean.groupby('job_id')['union_member'].mean()
    uj = (uj > 0.5).astype(float)
    df['union_job'] = df['job_id'].map(uj)
    return df


def run_analysis(data_source=DATA_FILE):
    print("Loading PSID panel data...")
    df = pd.read_csv(data_source)
    print(f"  Raw panel: {len(df)} obs, {df['person_id'].nunique()} persons")

    # Education
    df['education_years'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # Experience
    df['experience_correct'] = (df['age'] - df['education_years'] - 6).clip(lower=0)

    # Real wage
    df['gnp_deflator'] = df['year'].map(GNP_DEFLATOR)
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['hw'] = df['wages'] / df['hours']
    invalid = ~((df['hw'] > 0) & np.isfinite(df['hw']))
    df.loc[invalid, 'hw'] = df.loc[invalid, 'hourly_wage']
    df['log_real_wage'] = np.log(df['hw']) - np.log(df['gnp_deflator'] / 33.4) - np.log(df['cps_index'])

    # Tenure reconstruction with CORRECTED mapping
    print("Reconstructing tenure (corrected category mapping)...")
    df = reconstruct_tenure(df)

    # Fix variables
    print("Fixing variables...")
    df = fix_variable_by_carryover(df, 'married', MARRIED_BAD_YEARS)
    df = fix_variable_by_carryover(df, 'disabled', DISABLED_BAD_YEARS)
    all_ub = UNION_NAN_YEARS + UNION_BAD_YEARS
    df = fix_union_for_years(df, UNION_NAN_YEARS, all_ub)
    df = fix_union_for_years(df, UNION_BAD_YEARS, all_ub)
    df = compute_job_union(df)

    # Sample restrictions
    print("Applying restrictions...")
    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['govt_worker'] != 1].copy()
    df = df[df['self_employed'] != 1].copy()
    df = df[df['agriculture'] != 1].copy()
    df = df[df['hw'] > 0].copy()
    df = df[df['hw'] < 200].copy()
    df = df[df['education_years'].notna()].copy()
    df = df[np.isfinite(df['log_real_wage'])].copy()
    df = df[df['tenure_recon'] >= 1].copy()

    total_n = len(df)
    n_persons = df['person_id'].nunique()
    print(f"  Final: {total_n} obs, {n_persons} persons (Paper: {TOTAL_N_PAPER}, 1540)")
    for yr in sorted(df['year'].unique()):
        print(f"    {yr}: {len(df[df['year'] == yr])}")

    # Results
    results_top = {}
    results_top['Real wage'] = {'mean': df['log_real_wage'].mean(), 'sd': df['log_real_wage'].std(ddof=0)}
    results_top['Experience'] = {'mean': df['experience_correct'].mean(), 'sd': df['experience_correct'].std(ddof=0)}
    results_top['Tenure'] = {'mean': df['tenure_recon'].mean(), 'sd': df['tenure_recon'].std(ddof=0)}
    results_top['Education'] = {'mean': df['education_years'].mean(), 'sd': df['education_years'].std(ddof=0)}
    results_top['Married'] = {'mean': df['married'].dropna().mean(), 'sd': df['married'].dropna().std(ddof=0)}
    results_top['Union'] = {'mean': df['union_job'].dropna().mean(), 'sd': df['union_job'].dropna().std(ddof=0)}
    results_top['SMSA'] = {'mean': df['lives_in_smsa'].dropna().mean(), 'sd': df['lives_in_smsa'].dropna().std(ddof=0)}
    results_top['Disabled'] = {'mean': df['disabled'].dropna().mean(), 'sd': df['disabled'].dropna().std(ddof=0)}

    yc = df['year'].value_counts().sort_index()
    results_bottom = {}
    for yr in range(1968, 1984):
        n_yr = yc.get(yr, 0)
        results_bottom[yr] = {'n': n_yr, 'pct': round(n_yr / total_n, 3) if total_n > 0 else 0, 'cps_index': CPS_WAGE_INDEX[yr]}

    # Output
    lines = ["=" * 80, "TABLE A1: Variable Definitions and Summary Statistics",
             "PSID White Males, 1968-83", "=" * 80, "",
             "TOP PANEL: Variable Means and Standard Deviations", "-" * 80,
             f"{'Variable':<15s} {'Mean':>10s} {'Std Dev':>10s}   {'Paper Mean':>10s} {'Paper SD':>10s}", "-" * 80]
    for var in VAR_ORDER:
        g = results_top[var]; t = GROUND_TRUTH_TOP[var]
        note = " *BROKEN*" if var == 'SMSA' and g['mean'] < 0.01 else ""
        lines.append(f"{var:<15s} {g['mean']:>10.3f} {g['sd']:>10.3f}   {t['mean']:>10.3f} {t['sd']:>10.3f}{note}")
    lines.extend(["", f"Total N: {total_n}  (Paper: {TOTAL_N_PAPER})",
                   f"Unique persons: {n_persons}  (Paper: 1540)", "",
                   "BOTTOM PANEL: Sample Distribution by Survey Year", "-" * 80,
                   f"{'Year':>6s} {'N':>8s} {'Pct':>8s} {'CPS Index':>10s}   {'Paper Pct':>10s} {'Paper CPS':>10s}", "-" * 80])
    ps = 0
    for yr in range(1968, 1984):
        g = results_bottom[yr]; t = GROUND_TRUTH_BOTTOM[yr]; ps += g['pct']
        lines.append(f"{yr:>6d} {g['n']:>8d} {g['pct']:>8.3f} {g['cps_index']:>10.3f}   {t['pct']:>10.3f} {t['cps_index']:>10.3f}")
    lines.append(f"{'Total':>6s} {total_n:>8d} {ps:>8.3f}")
    output = "\n".join(lines)
    print(output)
    return output, results_top, results_bottom, total_n


def score_against_ground_truth():
    output, rt, rb, tn = run_analysis()
    print("\n\n" + "=" * 70 + "\nSCORING BREAKDOWN\n" + "=" * 70)
    ts = 0
    nv = len(rt); ny = sum(1 for yr in range(1968, 1984) if rb[yr]['n'] > 0)
    cs = (min(nv, 8) / 8) * 10 + (min(ny, 16) / 16) * 10
    print(f"\n1. Categories: {cs:.1f}/20 (vars={nv}/8, years={ny}/16)"); ts += cs
    tv = 48; nm = 0
    for var in VAR_ORDER:
        for stat in ['mean', 'sd']:
            true_val = GROUND_TRUTH_TOP[var][stat]; gen_val = rt[var][stat]
            err = abs(gen_val - true_val) / abs(true_val) if true_val != 0 else abs(gen_val - true_val)
            m = err <= 0.02; nm += int(m)
            print(f"   {var} {stat}: {gen_val:.3f} vs {true_val:.3f} - {'MATCH' if m else f'MISS ({err:.3f})'}")
    for yr in range(1968, 1984):
        tp = GROUND_TRUTH_BOTTOM[yr]['pct']; gp = rb[yr]['pct']
        err = abs(gp - tp) / tp if tp != 0 else abs(gp - tp)
        m = err <= 0.02; nm += int(m)
        print(f"   Year {yr} pct: {gp:.3f} vs {tp:.3f} - {'MATCH' if m else f'MISS ({err:.3f})'}")
    for yr in range(1968, 1984):
        if abs(rb[yr]['cps_index'] - GROUND_TRUTH_BOTTOM[yr]['cps_index']) < 0.001: nm += 1
    vs = (nm / tv) * 40
    print(f"\n2. Values: {vs:.1f}/40 ({nm}/{tv})"); ts += vs
    ts += 10; print(f"\n3. Ordering: 10.0/10")
    ne = abs(tn - TOTAL_N_PAPER) / TOTAL_N_PAPER
    ns = 20 if ne <= 0.05 else (15 if ne <= 0.10 else (10 if ne <= 0.20 else 5))
    print(f"\n4. N: {ns}/20 (N={tn}, err={ne:.3f})"); ts += ns
    ts += 10; print(f"\n5. Structure: 10.0/10")
    print(f"\n{'=' * 70}\nTOTAL SCORE: {ts:.0f}/100\n{'=' * 70}")
    return ts


if __name__ == '__main__':
    score = score_against_ground_truth()
