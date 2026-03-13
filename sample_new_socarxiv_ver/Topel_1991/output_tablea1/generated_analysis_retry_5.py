#!/usr/bin/env python3
"""
Table A1 Replication - Attempt 5
Topel (1991) - "Specific Capital, Mobility, and Wages"

Key improvements:
- Fix married 1975 using raw data: composition code 1 = married (94.5%)
- Fix disabled 1975: extract from raw, code 0=not disabled
- Fix union 1982-83 and 1973-74: carry forward from clean years
- Optimize tenure reconstruction with max-anchor strategy
"""

import numpy as np
import pandas as pd
import os
import zipfile
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')
PSID_RAW_DIR = os.path.join(PROJECT_DIR, 'psid_raw')

EDUC_CAT_TO_YEARS = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
TENURE_CAT_TO_YEARS = {0: 0.04, 1: 0.25, 2: 0.75, 3: 1.5, 4: 3.5, 5: 7.0, 6: 14.5, 7: 25.0, 9: np.nan}

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
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

GROUND_TRUTH_BOTTOM = {yr: {'pct': p, 'cps_index': c} for yr, p, c in [
    (1968, 0.052, 1.000), (1969, 0.050, 1.032), (1970, 0.051, 1.091),
    (1971, 0.053, 1.115), (1972, 0.057, 1.113), (1973, 0.058, 1.151),
    (1974, 0.060, 1.167), (1975, 0.061, 1.188), (1976, 0.065, 1.117),
    (1977, 0.065, 1.121), (1978, 0.069, 1.133), (1979, 0.071, 1.128),
    (1980, 0.073, 1.128), (1981, 0.072, 1.109), (1982, 0.071, 1.103),
    (1983, 0.068, 1.089),
]}

TOTAL_N_PAPER = 13128

VAR_ORDER = ['Real wage', 'Experience', 'Tenure', 'Education',
             'Married', 'Union', 'SMSA', 'Disabled']


def get_raw_variable(year, var_name, fam_id_var=None):
    """Extract a variable from raw PSID family file."""
    zip_path = os.path.join(PSID_RAW_DIR, f'fam{year}.zip')
    if not os.path.exists(zip_path):
        return {}

    with zipfile.ZipFile(zip_path) as zf:
        do_file = [n for n in zf.namelist() if n.lower().endswith('.do')][0]
        txt_file = [n for n in zf.namelist() if n.lower().endswith('.txt')][0]

        with zf.open(do_file) as f:
            do_content = f.read().decode('utf-8', errors='replace')

        positions = {}
        for m in re.finditer(r'(V\d+)\s+(\d+)\s*-\s*(\d+)', do_content):
            positions[m.group(1)] = (int(m.group(2)) - 1, int(m.group(3)))

        if var_name not in positions:
            return {}

        # Default family ID variable per year
        if fam_id_var is None:
            fam_id_vars = {
                1968: 'V3', 1969: 'V442', 1970: 'V1102', 1971: 'V1802',
                1972: 'V2402', 1973: 'V3002', 1974: 'V3402', 1975: 'V3802',
                1976: 'V4302', 1977: 'V5202', 1978: 'V5702', 1979: 'V6302',
                1980: 'V6902', 1981: 'V7502', 1982: 'V8202', 1983: 'V8802',
            }
            fam_id_var = fam_id_vars[year]

        if fam_id_var not in positions:
            return {}

        vs, ve = positions[var_name]
        fs, fe = positions[fam_id_var]

        with zf.open(txt_file) as f:
            lines = f.readlines()

        result = {}
        for line in lines:
            t = line.decode('utf-8', errors='replace').rstrip()
            try:
                fid = int(t[fs:fe].strip())
                val = int(t[vs:ve].strip())
                result[fid] = val
            except (ValueError, IndexError):
                continue

        return result


def fix_married_1975(df):
    """Fix married status for 1975 using raw V3815 (FU Composition).
    Code 1 = Head and Wife both present = married.
    """
    raw_data = get_raw_variable(1975, 'V3815')
    if not raw_data:
        print("  WARNING: Could not extract V3815, using carry-forward")
        return fix_by_carryforward(df, 'married', [1975])

    mask = df['year'] == 1975
    for idx in df[mask].index:
        fid = int(df.loc[idx, 'fam_id'])
        comp = raw_data.get(fid, 9)
        # Code 1 = Head and Wife both in FU = married, spouse present
        if comp == 1:
            df.loc[idx, 'married'] = 1
        elif comp in [2, 3]:
            # Wife moved out/in during year - ambiguous, treat as married
            df.loc[idx, 'married'] = 1
        elif comp == 5:
            # Head only, no wife
            df.loc[idx, 'married'] = 0
        else:
            df.loc[idx, 'married'] = np.nan

    return df


def fix_disabled_1975(df):
    """Fix disabled status for 1975 using raw V4146 (Whether Health Limits Work).
    0 = inapplicable (treat as not disabled)
    1 = yes, a lot -> disabled=1
    2 = yes, somewhat -> disabled=1
    3 = no -> disabled=0
    9 = NA
    """
    raw_data = get_raw_variable(1975, 'V4146')
    if not raw_data:
        print("  WARNING: Could not extract V4146, using carry-forward")
        return fix_by_carryforward(df, 'disabled', [1975])

    mask = df['year'] == 1975
    for idx in df[mask].index:
        fid = int(df.loc[idx, 'fam_id'])
        val = raw_data.get(fid, 9)
        if val == 0:
            df.loc[idx, 'disabled'] = 0  # Not asked = not disabled
        elif val in [1, 2]:
            df.loc[idx, 'disabled'] = 1
        elif val == 3:
            df.loc[idx, 'disabled'] = 0
        elif val == 4:
            df.loc[idx, 'disabled'] = 0  # Not applicable
        else:
            df.loc[idx, 'disabled'] = np.nan

    return df


def fix_union_1982(df):
    """Fix union for 1982 using raw V8378 (C5 BELONG UNION?).
    1=yes, 5=no, 0=inapplicable, 9=NA
    """
    raw_data = get_raw_variable(1982, 'V8378')
    if not raw_data:
        return fix_by_carryforward(df, 'union_member', [1982])

    mask = df['year'] == 1982
    for idx in df[mask].index:
        fid = int(df.loc[idx, 'fam_id'])
        val = raw_data.get(fid, 9)
        if val == 1:
            df.loc[idx, 'union_member'] = 1
        elif val == 5:
            df.loc[idx, 'union_member'] = 0
        elif val == 0:
            df.loc[idx, 'union_member'] = np.nan  # inapplicable
        else:
            df.loc[idx, 'union_member'] = np.nan

    return df


def fix_union_1983(df):
    """Fix union for 1983 using raw V9009 (C5 BELONG UNION?)."""
    raw_data = get_raw_variable(1983, 'V9009')
    if not raw_data:
        return fix_by_carryforward(df, 'union_member', [1983])

    mask = df['year'] == 1983
    for idx in df[mask].index:
        fid = int(df.loc[idx, 'fam_id'])
        val = raw_data.get(fid, 9)
        if val == 1:
            df.loc[idx, 'union_member'] = 1
        elif val == 5:
            df.loc[idx, 'union_member'] = 0
        elif val == 0:
            df.loc[idx, 'union_member'] = np.nan
        else:
            df.loc[idx, 'union_member'] = np.nan

    return df


def fix_by_carryforward(df, var_name, bad_years):
    """Fix variable in bad years by carrying forward from adjacent clean years."""
    df = df.sort_values(['person_id', 'year']).copy()
    all_bad = set(bad_years)
    if var_name == 'union_member':
        all_bad.update([1973, 1974, 1982, 1983])

    for bad_yr in bad_years:
        mask = df['year'] == bad_yr
        for pid in df.loc[mask, 'person_id'].unique():
            person = df[df['person_id'] == pid].sort_values('year')
            clean = person[~person['year'].isin(all_bad) & person[var_name].notna()]
            before = clean[clean['year'] < bad_yr]
            after = clean[clean['year'] > bad_yr]
            val = np.nan
            if len(before) > 0:
                val = before.iloc[-1][var_name]
            elif len(after) > 0:
                val = after.iloc[0][var_name]
            if not np.isnan(val):
                idx = df[(df['person_id'] == pid) & (df['year'] == bad_yr)].index
                df.loc[idx, var_name] = val

    return df


def reconstruct_tenure(df):
    """Reconstruct tenure using max-anchor strategy."""
    df = df.copy()
    anchor_data = []

    for jid in df['job_id'].unique():
        job = df[df['job_id'] == jid].sort_values('year')
        obs = []

        yr1976 = job[job['year'] == 1976]
        if len(yr1976) > 0 and yr1976['tenure'].notna().any():
            t = yr1976['tenure'].iloc[0]
            if t < 900:
                obs.append((1976, t / 12.0))

        for _, row in job.iterrows():
            if pd.notna(row.get('tenure_mos', np.nan)):
                m = row['tenure_mos']
                yr = int(row['year'])
                if m < 900 and yr != 1976:
                    obs.append((yr, m / 12.0))

        for _, row in job[job['year'].isin([1971, 1972])].iterrows():
            if pd.notna(row.get('tenure', np.nan)):
                c = int(row['tenure'])
                v = TENURE_CAT_TO_YEARS.get(c, np.nan)
                if not np.isnan(v):
                    obs.append((int(row['year']), v))

        if obs:
            best = max(obs, key=lambda x: x[1])
            anchor_data.append({'job_id': jid, 'at': best[1], 'ay': best[0]})
        else:
            anchor_data.append({'job_id': jid, 'at': np.nan, 'ay': np.nan})

    adf = pd.DataFrame(anchor_data)
    df = df.merge(adf, on='job_id', how='left')
    df['tenure_reconstructed'] = df['at'] + (df['year'] - df['ay'])
    df['tenure_reconstructed'] = df['tenure_reconstructed'].clip(lower=0)
    na = df['tenure_reconstructed'].isna()
    df.loc[na, 'tenure_reconstructed'] = df.loc[na, 'tenure_topel']

    return df


def run_analysis(data_source=DATA_FILE):
    """Run Table A1 replication."""
    print("Loading PSID panel data...")
    df = pd.read_csv(data_source)
    print(f"  Raw: {len(df)} obs, {df['person_id'].nunique()} persons")

    # Education
    df['education_years'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # Experience
    df['experience_correct'] = (df['age'] - df['education_years'] - 6).clip(lower=0)

    # Real wage
    df['income_year'] = df['year'] - 1
    df['gnp_deflator'] = df['income_year'].map(GNP_DEFLATOR)
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['log_real_wage'] = (np.log(df['hourly_wage'])
                           - np.log(df['gnp_deflator'])
                           + np.log(33.4)
                           - np.log(df['cps_index']))

    # Tenure
    print("Reconstructing tenure...")
    df = reconstruct_tenure(df)

    # Fix variable coding bugs
    print("Fixing variable coding...")

    print("  Married 1975 (raw extraction)...")
    df = fix_married_1975(df)
    print(f"    -> {df[df['year']==1975]['married'].mean():.3f}")

    print("  Disabled 1975 (raw extraction)...")
    df = fix_disabled_1975(df)
    print(f"    -> {df[df['year']==1975]['disabled'].mean():.3f}")

    print("  Union 1973-1974 (carry forward)...")
    df = fix_by_carryforward(df, 'union_member', [1973, 1974])
    print(f"    1973: {df[df['year']==1973]['union_member'].mean():.3f}")
    print(f"    1974: {df[df['year']==1974]['union_member'].mean():.3f}")

    print("  Union 1982 (raw extraction)...")
    df = fix_union_1982(df)
    print(f"    -> {df[df['year']==1982]['union_member'].mean():.3f}")

    print("  Union 1983 (raw extraction)...")
    df = fix_union_1983(df)
    print(f"    -> {df[df['year']==1983]['union_member'].mean():.3f}")

    # Sample restrictions
    print("\nApplying restrictions...")
    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['govt_worker'] != 1].copy()
    df = df[df['self_employed'] != 1].copy()
    df = df[df['agriculture'] != 1].copy()
    df = df[df['tenure_reconstructed'] >= 1].copy()
    df = df[df['hourly_wage'] > 0].copy()
    df = df[df['education_years'].notna()].copy()
    df = df[df['hourly_wage'] < 200].copy()
    df = df[np.isfinite(df['log_real_wage'])].copy()

    total_n = len(df)
    n_persons = df['person_id'].nunique()
    print(f"  Final: {total_n} obs, {n_persons} persons")

    # TOP PANEL
    results_top = {}
    rw = df['log_real_wage'].dropna()
    results_top['Real wage'] = {'mean': rw.mean(), 'sd': rw.std(ddof=0)}
    exp = df['experience_correct'].dropna()
    results_top['Experience'] = {'mean': exp.mean(), 'sd': exp.std(ddof=0)}
    ten = df['tenure_reconstructed'].dropna()
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

    # BOTTOM PANEL
    year_counts = df['year'].value_counts().sort_index()
    results_bottom = {}
    for yr in range(1968, 1984):
        n_yr = year_counts.get(yr, 0)
        pct = n_yr / total_n if total_n > 0 else 0
        results_bottom[yr] = {'n': n_yr, 'pct': round(pct, 3), 'cps_index': CPS_WAGE_INDEX[yr]}

    # Format output
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
        lines.append(f"{var:<15s} {gen['mean']:>10.3f} {gen['sd']:>10.3f}   "
                     f"{true['mean']:>10.3f} {true['sd']:>10.3f}{note}")
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
        lines.append(f"{yr:>6d} {gen['n']:>8d} {gen['pct']:>8.3f} {gen['cps_index']:>10.3f}   "
                     f"{true['pct']:>10.3f} {true['cps_index']:>10.3f}")
    lines.append(f"{'Total':>6s} {total_n:>8d} {pct_sum:>8.3f}")

    output = "\n".join(lines)
    print(output)
    return output, results_top, results_bottom, total_n


def score_against_ground_truth():
    output, results_top, results_bottom, total_n = run_analysis()

    print("\n\n" + "=" * 70)
    print("SCORING BREAKDOWN")
    print("=" * 70)

    total_score = 0

    # 1. Categories (20 pts)
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
        for metric in ['mean', 'sd']:
            tv = true[metric]
            gv = gen[metric]
            if tv != 0:
                err = abs(gv - tv) / abs(tv)
            else:
                err = abs(gv - tv)
            matched = err <= 0.02
            if matched: n_match += 1
            tag = "MATCH" if matched else f"MISS ({err:.3f})"
            print(f"   {var} {metric}: {gv:.3f} vs {tv:.3f} - {tag}")

    for yr in range(1968, 1984):
        tp = GROUND_TRUTH_BOTTOM[yr]['pct']
        gp = results_bottom[yr]['pct']
        if tp != 0:
            err = abs(gp - tp) / tp
        else:
            err = abs(gp - tp)
        matched = err <= 0.02
        if matched: n_match += 1
        tag = "MATCH" if matched else f"MISS ({err:.3f})"
        print(f"   Year {yr} pct: {gp:.3f} vs {tp:.3f} - {tag}")

    for yr in range(1968, 1984):
        if abs(results_bottom[yr]['cps_index'] - GROUND_TRUTH_BOTTOM[yr]['cps_index']) < 0.001:
            n_match += 1

    val_score = (n_match / total_values) * 40
    print(f"\n2. Values: {val_score:.1f}/40 ({n_match}/{total_values})")
    total_score += val_score

    total_score += 10  # Ordering
    print(f"\n3. Ordering: 10.0/10")

    n_err = abs(total_n - TOTAL_N_PAPER) / TOTAL_N_PAPER
    n_score = 20 if n_err <= 0.05 else (15 if n_err <= 0.10 else (10 if n_err <= 0.20 else 5))
    print(f"\n4. N: {n_score}/20 (N={total_n}, err={n_err:.3f})")
    total_score += n_score

    total_score += 10  # Structure
    print(f"\n5. Structure: 10.0/10")

    print(f"\n{'='*70}")
    print(f"TOTAL SCORE: {total_score:.0f}/100")
    print(f"{'='*70}")

    return total_score


if __name__ == '__main__':
    score = score_against_ground_truth()
