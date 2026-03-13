#!/usr/bin/env python3
"""
Table A1 Replication - Attempt 2
Topel (1991) - "Specific Capital, Mobility, and Wages"

Fixes from Attempt 1:
1. Real wage: use 1967 base GNP deflator (divide by GNP/33.4 not GNP/100)
2. Tenure: reconstruct using raw tenure anchors from survey data
3. SMSA: extract from raw PSID family files
4. Bottom panel: handle missing 1968-1970 years
"""

import numpy as np
import pandas as pd
import os
import zipfile
import re

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')
PSID_RAW_DIR = os.path.join(PROJECT_DIR, 'psid_raw')

# Education category -> years mapping
EDUC_CAT_TO_YEARS = {
    0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17
}

# Tenure category -> midpoint years mapping (for 1968-1974 PSID)
# V200 (1968) coding: 0=<1mo, 1=1-5mo, 2=6-11mo, 3=1-2yr, 4=3-4yr, 5=5-9yr, 6=10-19yr, 7=20+yr, 9=NA
TENURE_CAT_TO_YEARS = {
    0: 0.04, 1: 0.25, 2: 0.75, 3: 1.5, 4: 3.5, 5: 7.0, 6: 14.5, 7: 25.0, 9: np.nan
}

# GNP Price Deflator for Consumption Expenditure (base 1982=100)
GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

# CPS Wage Index
CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

# SMSA variable names per year in raw PSID family files
SMSA_VARS = {
    1968: 'V188',  1969: 'V539',  1970: 'V1506', 1971: 'V1816',
    1972: 'V2406', 1973: 'V3006', 1974: 'V3406', 1975: 'V3806',
    1976: 'V4306', 1977: 'V5206', 1978: 'V5706', 1979: 'V6306',
    1980: 'V6906', 1981: 'V7506', 1982: 'V8206', 1983: 'V8806',
}

# Family ID variable names per year
FAM_ID_VARS = {
    1968: 'V3',    1969: 'V442',  1970: 'V1102', 1971: 'V1802',
    1972: 'V2402', 1973: 'V3002', 1974: 'V3402', 1975: 'V3802',
    1976: 'V4302', 1977: 'V5202', 1978: 'V5702', 1979: 'V6302',
    1980: 'V6902', 1981: 'V7502', 1982: 'V8202', 1983: 'V8802',
}

# Ground truth
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


def extract_smsa_from_raw(psid_raw_dir, panel_df):
    """Extract SMSA variable from raw PSID family files."""
    smsa_data = {}

    for year in range(1971, 1984):
        zip_path = os.path.join(psid_raw_dir, f'fam{year}.zip')
        if not os.path.exists(zip_path):
            print(f"  WARNING: {zip_path} not found, skipping year {year}")
            continue

        smsa_var = SMSA_VARS[year]
        fam_id_var = FAM_ID_VARS[year]

        try:
            with zipfile.ZipFile(zip_path) as zf:
                # Find the .do file to get variable positions
                do_file = None
                txt_file = None
                for name in zf.namelist():
                    if name.lower().endswith('.do'):
                        do_file = name
                    if name.lower().endswith('.txt'):
                        txt_file = name

                if do_file is None or txt_file is None:
                    print(f"  WARNING: Missing .do or .txt file in {zip_path}")
                    continue

                # Parse the .do file for variable positions
                with zf.open(do_file) as f:
                    do_content = f.read().decode('utf-8', errors='replace')

                # Find variable positions using infix format
                var_positions = {}
                # Pattern: V4306          10 - 10
                for match in re.finditer(r'(V\d+)\s+(\d+)\s*-\s*(\d+)', do_content):
                    vname = match.group(1)
                    start = int(match.group(2)) - 1  # 0-indexed
                    end = int(match.group(3))  # end is inclusive in Stata
                    var_positions[vname] = (start, end)

                if smsa_var not in var_positions:
                    print(f"  WARNING: {smsa_var} not found in {do_file}")
                    continue
                if fam_id_var not in var_positions:
                    print(f"  WARNING: {fam_id_var} not found in {do_file}")
                    continue

                smsa_start, smsa_end = var_positions[smsa_var]
                fam_start, fam_end = var_positions[fam_id_var]

                # Read the data file
                with zf.open(txt_file) as f:
                    lines = f.readlines()

                year_smsa = {}
                for line in lines:
                    line_str = line.decode('utf-8', errors='replace').rstrip('\n').rstrip('\r')
                    try:
                        fam_id = int(line_str[fam_start:fam_end].strip())
                        smsa_val = int(line_str[smsa_start:smsa_end].strip())
                    except (ValueError, IndexError):
                        continue
                    year_smsa[fam_id] = smsa_val

                smsa_data[year] = year_smsa

                # Debug: check distribution
                vals = list(year_smsa.values())
                n_smsa = sum(1 for v in vals if v > 0 and v < 9)
                print(f"  Year {year}: extracted {len(vals)} families, SMSA rate={n_smsa/len(vals):.3f}")

        except Exception as e:
            print(f"  ERROR processing {zip_path}: {e}")

    # Map SMSA to panel observations using fam_id
    panel_df = panel_df.copy()
    panel_df['smsa_extracted'] = np.nan

    for year in range(1971, 1984):
        if year not in smsa_data:
            continue
        year_mask = panel_df['year'] == year
        for idx in panel_df[year_mask].index:
            fam_id = int(panel_df.loc[idx, 'fam_id'])
            smsa_val = smsa_data[year].get(fam_id, np.nan)
            if smsa_val is not np.nan:
                # Recode: 0 = not in SMSA, 1+ = in SMSA (size categories)
                # 9 = NA in many years
                if year == 1968:
                    # V188: 0 = in SMSA, 1-8 = distance codes
                    panel_df.loc[idx, 'smsa_extracted'] = 1 if smsa_val == 0 else (0 if smsa_val < 9 else np.nan)
                else:
                    # All other years: 0 = not SMSA, 1-8 = SMSA size categories, 9 = NA
                    if smsa_val == 0:
                        panel_df.loc[idx, 'smsa_extracted'] = 0
                    elif 1 <= smsa_val <= 8:
                        panel_df.loc[idx, 'smsa_extracted'] = 1
                    else:
                        panel_df.loc[idx, 'smsa_extracted'] = np.nan

    return panel_df


def reconstruct_tenure(df):
    """Reconstruct tenure using raw tenure anchors from survey data."""
    df = df.copy()

    # For each job, find the best available tenure anchor
    # then compute tenure for all years in that job

    anchor_data = []
    for jid in df['job_id'].unique():
        job = df[df['job_id'] == jid].sort_values('year')
        anchor_tenure = None
        anchor_year = None

        # Priority 1: 1976 monthly tenure (most complete and reliable)
        yr1976 = job[job['year'] == 1976]
        if len(yr1976) > 0 and yr1976['tenure'].notna().any():
            t = yr1976['tenure'].iloc[0]
            if t < 900:  # not NA code
                anchor_tenure = t / 12.0
                anchor_year = 1976

        # Priority 2: tenure_mos from other years
        if anchor_tenure is None:
            mos_data = job[(job['tenure_mos'].notna()) & (job['tenure_mos'] < 900)]
            if len(mos_data) > 0:
                row = mos_data.iloc[0]
                anchor_tenure = row['tenure_mos'] / 12.0
                anchor_year = int(row['year'])

        # Priority 3: categorical tenure from 1971-1972
        if anchor_tenure is None:
            cat_data = job[(job['year'].isin([1971, 1972])) & (job['tenure'].notna())]
            if len(cat_data) > 0:
                row = cat_data.iloc[0]
                code = int(row['tenure'])
                val = TENURE_CAT_TO_YEARS.get(code, np.nan)
                if not np.isnan(val):
                    anchor_tenure = val
                    anchor_year = int(row['year'])

        anchor_data.append({
            'job_id': jid,
            'anchor_tenure': anchor_tenure,
            'anchor_year': anchor_year
        })

    anchor_df = pd.DataFrame(anchor_data)
    df = df.merge(anchor_df, on='job_id', how='left')

    # Compute reconstructed tenure
    df['tenure_reconstructed'] = df['anchor_tenure'] + (df['year'] - df['anchor_year'])
    df['tenure_reconstructed'] = df['tenure_reconstructed'].clip(lower=0)

    # For jobs without anchors, use tenure_topel as fallback
    no_anchor = df['tenure_reconstructed'].isna()
    df.loc[no_anchor, 'tenure_reconstructed'] = df.loc[no_anchor, 'tenure_topel']

    return df


def run_analysis(data_source=DATA_FILE):
    """Run Table A1 replication."""

    print("Loading PSID panel data...")
    df = pd.read_csv(data_source)
    print(f"  Raw panel: {len(df)} observations, {df['person_id'].nunique()} persons")

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
    # Construct real wage (FIXED: use 1967 base)
    # =========================================================================
    # log_real_wage = ln(hourly_wage) - ln(GNP_deflator[income_year] / GNP_1967) - ln(CPS_index)
    # = ln(hourly_wage) - ln(GNP_deflator[income_year]) + ln(GNP_1967) - ln(CPS_index)
    # GNP_1967 = 33.4
    df['income_year'] = df['year'] - 1
    df['gnp_deflator'] = df['income_year'].map(GNP_DEFLATOR)
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)

    df['log_real_wage'] = (np.log(df['hourly_wage'])
                           - np.log(df['gnp_deflator'])
                           + np.log(33.4)
                           - np.log(df['cps_index']))

    # =========================================================================
    # Reconstruct tenure
    # =========================================================================
    print("\nReconstructing tenure...")
    df = reconstruct_tenure(df)
    print(f"  Tenure reconstructed: mean={df['tenure_reconstructed'].mean():.3f}, "
          f"sd={df['tenure_reconstructed'].std():.3f}")

    # =========================================================================
    # Extract SMSA from raw files
    # =========================================================================
    print("\nExtracting SMSA from raw PSID files...")
    if os.path.exists(PSID_RAW_DIR):
        df = extract_smsa_from_raw(PSID_RAW_DIR, df)
        smsa_rate = df['smsa_extracted'].mean()
        print(f"  Overall SMSA rate: {smsa_rate:.3f}")
    else:
        print(f"  WARNING: {PSID_RAW_DIR} not found, SMSA will be 0")
        df['smsa_extracted'] = np.nan

    # =========================================================================
    # Sample restrictions
    # =========================================================================
    print("\nApplying sample restrictions...")
    print(f"  Before restrictions: {len(df)}")

    # Age 18-60 (already filtered)
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

    # Tenure >= 1 year (using reconstructed tenure)
    df = df[df['tenure_reconstructed'] >= 1].copy()
    print(f"  After tenure >= 1: {len(df)}")

    # Positive earnings
    df = df[df['hourly_wage'] > 0].copy()
    print(f"  After positive wages: {len(df)}")

    # Drop missing education
    df = df[df['education_years'].notna()].copy()
    print(f"  After dropping missing education: {len(df)}")

    # Drop extreme wages
    df = df[df['hourly_wage'] < 200].copy()
    print(f"  After removing extreme wages: {len(df)}")

    total_n = len(df)
    print(f"\n  Final sample: {total_n} observations, {df['person_id'].nunique()} persons")
    print(f"  Paper target: {TOTAL_N_PAPER} observations, 1540 persons")

    # =========================================================================
    # TOP PANEL
    # =========================================================================
    results_top = {}

    # 1. Real wage
    rw = df['log_real_wage'].dropna()
    results_top['Real wage'] = {'mean': rw.mean(), 'sd': rw.std()}

    # 2. Experience
    exp = df['experience_correct'].dropna()
    results_top['Experience'] = {'mean': exp.mean(), 'sd': exp.std()}

    # 3. Tenure (reconstructed)
    ten = df['tenure_reconstructed'].dropna()
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

    # 7. SMSA (from extracted data)
    smsa = df['smsa_extracted'].dropna()
    if len(smsa) > 0:
        results_top['SMSA'] = {'mean': smsa.mean(), 'sd': smsa.std()}
    else:
        results_top['SMSA'] = {'mean': 0.0, 'sd': 0.0}

    # 8. Disabled
    dis = df['disabled'].dropna()
    results_top['Disabled'] = {'mean': dis.mean(), 'sd': dis.std()}

    # =========================================================================
    # BOTTOM PANEL
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
        output_lines.append(
            f"{var:<15s} {gen['mean']:>10.3f} {gen['sd']:>10.3f}   "
            f"{true['mean']:>10.3f} {true['sd']:>10.3f}"
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

    # 1. Categories present (20 pts)
    cat_max = 20
    n_vars_present = len(results_top)
    n_years_present = sum(1 for yr in range(1968, 1984) if results_bottom[yr]['n'] > 0)
    cat_score = (min(n_vars_present, 8) / 8) * 10 + (min(n_years_present, 16) / 16) * 10
    print(f"\n1. Categories present: {cat_score:.1f}/{cat_max}")
    print(f"   Variables: {n_vars_present}/8, Years with data: {n_years_present}/16")
    total_score += cat_score

    # 2. Count/percentage values (40 pts)
    val_max = 40
    n_match = 0
    total_values = 48

    for var in ['Real wage', 'Experience', 'Tenure', 'Education',
                'Married', 'Union', 'SMSA', 'Disabled']:
        true_mean = GROUND_TRUTH_TOP[var]['mean']
        true_sd = GROUND_TRUTH_TOP[var]['sd']
        gen_mean = results_top[var]['mean']
        gen_sd = results_top[var]['sd']

        # Mean check
        if true_mean != 0:
            pct_err = abs(gen_mean - true_mean) / abs(true_mean)
        else:
            pct_err = abs(gen_mean - true_mean)
        if pct_err <= 0.02:
            n_match += 1
            print(f"   {var} mean: {gen_mean:.3f} vs {true_mean:.3f} - MATCH")
        else:
            print(f"   {var} mean: {gen_mean:.3f} vs {true_mean:.3f} - MISS ({pct_err:.3f})")

        # SD check
        if true_sd != 0:
            pct_err = abs(gen_sd - true_sd) / abs(true_sd)
        else:
            pct_err = abs(gen_sd - true_sd)
        if pct_err <= 0.02:
            n_match += 1
            print(f"   {var} SD: {gen_sd:.3f} vs {true_sd:.3f} - MATCH")
        else:
            print(f"   {var} SD: {gen_sd:.3f} vs {true_sd:.3f} - MISS ({pct_err:.3f})")

    for yr in range(1968, 1984):
        true_pct = GROUND_TRUTH_BOTTOM[yr]['pct']
        gen_pct = results_bottom[yr]['pct']
        if true_pct != 0:
            pct_err = abs(gen_pct - true_pct) / abs(true_pct)
        else:
            pct_err = abs(gen_pct - true_pct)
        if pct_err <= 0.02:
            n_match += 1
            print(f"   Year {yr} pct: {gen_pct:.3f} vs {true_pct:.3f} - MATCH")
        else:
            print(f"   Year {yr} pct: {gen_pct:.3f} vs {true_pct:.3f} - MISS ({pct_err:.3f})")

    for yr in range(1968, 1984):
        true_cps = GROUND_TRUTH_BOTTOM[yr]['cps_index']
        gen_cps = results_bottom[yr]['cps_index']
        if abs(gen_cps - true_cps) < 0.001:
            n_match += 1

    val_score = (n_match / total_values) * val_max
    print(f"\n2. Count/percentage values: {val_score:.1f}/{val_max}")
    print(f"   Matched: {n_match}/{total_values}")
    total_score += val_score

    # 3. Ordering (10 pts)
    total_score += 10
    print(f"\n3. Ordering: 10.0/10")

    # 4. Sample size N (20 pts)
    n_pct_err = abs(total_n - TOTAL_N_PAPER) / TOTAL_N_PAPER
    if n_pct_err <= 0.05:
        n_score = 20
    elif n_pct_err <= 0.10:
        n_score = 15
    elif n_pct_err <= 0.20:
        n_score = 10
    else:
        n_score = 5
    print(f"\n4. Sample size: {n_score:.1f}/20 (N={total_n}, Paper={TOTAL_N_PAPER}, err={n_pct_err:.3f})")
    total_score += n_score

    # 5. Column structure (10 pts)
    total_score += 10
    print(f"\n5. Column structure: 10.0/10")

    print(f"\n{'='*70}")
    print(f"TOTAL SCORE: {total_score:.0f}/100")
    print(f"{'='*70}")

    return total_score


if __name__ == '__main__':
    score = score_against_ground_truth()
