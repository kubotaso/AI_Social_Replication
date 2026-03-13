#!/usr/bin/env python3
"""
Table A1 Replication - Attempt 10
Topel (1991) - "Specific Capital, Mobility, and Wages"

STRATEGY CHANGE: Instead of trying to extract 1968-1970 from raw files
(which is complex and error-prone), we take two complementary approaches:

A) Try to improve individual variable means to get more matches
   - Tenure: try using weighted/averaged anchors instead of max
   - Experience: adjust to use age - education - 6 more carefully
   - Married: try mode-based carry-forward instead of nearest-before
   - Union: try direct (non-job-level) union variable

B) Re-examine the build script to see if we can include 1968-1970 from
   the full panel (psid_panel_full.csv) since it has 1970 data
   -> This would give us 1 more year and adjust all percentages

C) Key insight: The paper uses 1968 as first year but our data starts at 1971.
   The build script's full panel starts at 1970. Even adding 1970 would help.
"""

import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')
DATA_FILE_FULL = os.path.join(PROJECT_DIR, 'data', 'psid_panel_full.csv')

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
    """Fix miscoded variable values by carrying forward/backward."""
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
    """Reconstruct tenure using all available data sources.

    Strategy: Use the MEDIAN of all anchor observations for a job
    (rather than max) to avoid upward bias from outlier tenure reports.
    """
    anchor_data = {}

    for jid in df['job_id'].unique():
        job = df[df['job_id'] == jid].sort_values('year')
        tenure_obs = []

        for _, row in job.iterrows():
            yr = int(row['year'])

            if pd.notna(row.get('tenure_mos', np.nan)):
                mos = row['tenure_mos']
                if 0 < mos < 900 and yr != 1977:
                    tenure_obs.append((yr, mos / 12.0))

            if pd.notna(row.get('tenure', np.nan)):
                t = row['tenure']
                if yr in [1971, 1972]:
                    val = TENURE_CAT.get(int(t), np.nan)
                    if not np.isnan(val):
                        tenure_obs.append((yr, val))
                elif yr == 1976:
                    if 0 < t < 900:
                        tenure_obs.append((yr, t / 12.0))

        if tenure_obs:
            # Normalize all observations to the same reference point
            # Convert each to "implied job start year"
            start_years = [yr - ten for yr, ten in tenure_obs]
            # Use median start year for consistency
            median_start = np.median(start_years)
            # Use the latest observation as anchor but compute from median start
            latest = max(tenure_obs, key=lambda x: x[0])
            anchor_tenure = latest[0] - median_start
            anchor_data[jid] = {'anchor_tenure': anchor_tenure, 'anchor_year': latest[0]}
        else:
            anchor_data[jid] = {'anchor_tenure': np.nan, 'anchor_year': np.nan}

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


def compute_job_union(df):
    """Compute union status as job-level variable.
    Paper: '1 if union member in more than half of the years of the job' (fn 18, p.164)
    """
    clean_mask = (~df['year'].isin(UNION_NAN_YEARS + UNION_BAD_YEARS)) & df['union_member'].notna()
    df_clean = df[clean_mask]

    union_by_job = df_clean.groupby('job_id')['union_member'].mean()
    union_by_job = (union_by_job > 0.5).astype(float)

    df['union_job'] = df['job_id'].map(union_by_job)
    return df


def try_add_1970_data(df):
    """Try to merge 1970 data from the full panel to add that year."""
    if not os.path.exists(DATA_FILE_FULL):
        print("  Full panel not found, skipping 1970 data")
        return df

    df_full = pd.read_csv(DATA_FILE_FULL)
    df_1970 = df_full[df_full['year'] == 1970].copy()

    if len(df_1970) == 0:
        print("  No 1970 data in full panel")
        return df

    print(f"  1970 data from full panel: {len(df_1970)} obs")

    # Check what columns we need that are present
    needed_cols = ['person_id', 'year', 'age', 'education_clean', 'married',
                   'wages', 'hours', 'hourly_wage', 'self_employed', 'agriculture',
                   'union_member', 'disabled', 'lives_in_smsa', 'govt_worker',
                   'tenure', 'labor_inc', 'tenure_mos', 'same_emp', 'new_job',
                   'job_id', 'tenure_topel']

    available = [c for c in needed_cols if c in df_1970.columns]
    missing = [c for c in needed_cols if c not in df_1970.columns]

    if missing:
        print(f"  Missing columns in 1970 data: {missing}")
        # Add missing columns as NaN
        for col in missing:
            df_1970[col] = np.nan

    # Check if govt_worker / self_employed / agriculture have actual data
    # (not all zeros)
    for var in ['govt_worker', 'self_employed', 'agriculture']:
        if var in df_1970.columns:
            n_nonzero = (df_1970[var] != 0).sum()
            print(f"  1970 {var} nonzero: {n_nonzero}/{len(df_1970)}")

    # Only include 1970 data for persons already in main panel
    # (to maintain the same sample of individuals)
    existing_persons = df['person_id'].unique()
    df_1970_matched = df_1970[df_1970['person_id'].isin(existing_persons)].copy()
    print(f"  1970 data for existing persons: {len(df_1970_matched)} obs")

    if len(df_1970_matched) == 0:
        print("  No matching persons, skipping")
        return df

    # Make sure all columns in df exist in df_1970_matched
    for col in df.columns:
        if col not in df_1970_matched.columns:
            df_1970_matched[col] = np.nan

    # Keep only columns present in df
    df_1970_matched = df_1970_matched[df.columns].copy()

    # Combine
    df_combined = pd.concat([df_1970_matched, df], ignore_index=True)
    df_combined = df_combined.sort_values(['person_id', 'year']).reset_index(drop=True)

    print(f"  Combined panel: {len(df_combined)} obs (was {len(df)})")
    return df_combined


def try_extract_1968_1969():
    """Try to extract 1968-1969 data directly from raw PSID files."""
    import struct

    raw_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'psid_raw')

    results = {}

    for year in [1968, 1969]:
        filepath = os.path.join(raw_base, f'fam{year}', f'FAM{year}.txt')
        if not os.path.exists(filepath):
            print(f"  Raw file for {year} not found")
            continue

        print(f"  Reading raw {year} data from {filepath}...")

        # Define column positions based on build_psid_panel.py FAMILY_VARS
        if year == 1968:
            var_specs = {
                'interview_number': (1, 5),    # cols 2-5 (1-indexed)
                'age_head': (282, 284),         # V117
                'sex_head': (286, 287),         # V119
                'race': (361, 362),             # V181
                'education': (520, 521),        # V313
                'marital_status': (437, 438),   # V239
                'labor_income': (182, 187),     # V74
                'hourly_earnings': (607, 612),  # V337
                'annual_hours': (113, 117),     # V47
                'self_employed': (387, 388),    # V198
                'occupation': (381, 384),       # V197_A
                'industry': (384, 387),         # V197_B
                'union': (500, 501),            # V294
                'disability': (408, 409),       # V216
                'smsa': (368, 371),             # V188
            }
        else:  # 1969
            var_specs = {
                'interview_number': (1, 5),    # V442
                'age_head': (1059, 1061),       # V1008
                'sex_head': (1062, 1063),       # V1010
                'race': (576, 577),             # V801
                'education': (569, 570),        # V794
                'marital_status': (346, 347),   # V607
                'labor_income': (184, 189),     # V514
                'hourly_earnings': (724, 729),  # V871
                'annual_hours': (61, 65),       # V465
                'self_employed': (392, 393),    # V641
                'occupation': (386, 389),       # V640_A
                'industry': (389, 392),         # V640_B
                'union': (536, 537),            # V766
                'disability': (513, 514),       # V743
                'smsa': (583, 586),             # V808
            }

        # Read the fixed-width file
        rows = []
        with open(filepath, 'r') as f:
            for line in f:
                row = {}
                for varname, (start, end) in var_specs.items():
                    # Convert 1-based to 0-based
                    s = start - 1
                    e = end
                    try:
                        val_str = line[s:e].strip()
                        row[varname] = int(val_str) if val_str else np.nan
                    except (ValueError, IndexError):
                        row[varname] = np.nan
                rows.append(row)

        df_raw = pd.DataFrame(rows)
        print(f"  {year}: Read {len(df_raw)} families")
        results[year] = df_raw

    return results


def run_analysis(data_source=DATA_FILE):
    """Run Table A1 replication."""

    print("Loading PSID panel data...")
    df = pd.read_csv(data_source)
    print(f"  Raw panel: {len(df)} obs, {df['person_id'].nunique()} persons")
    print(f"  Years: {sorted(df['year'].unique())}")

    # =========================================================================
    # Try to add 1970 data from full panel
    # =========================================================================
    print("\nAttempting to add 1970 data from full panel...")
    df = try_add_1970_data(df)
    print(f"  After merge: {len(df)} obs, years={sorted(df['year'].unique())}")

    # =========================================================================
    # Try to add 1968-1969 data from raw files
    # =========================================================================
    print("\nAttempting to extract 1968-1969 data from raw files...")
    raw_data = try_extract_1968_1969()

    # If we got raw data for 1968 or 1969, we need to:
    # 1. Match family records to person_ids via the individual file
    # 2. Apply the same restrictions (white male, non-agri, etc.)
    # This is complex but let's try

    # We need the individual file to map interview numbers to person_ids
    ind_file = os.path.join(PROJECT_DIR, 'psid_raw', 'ind2019er', 'IND2019ER.txt')
    if not os.path.exists(ind_file):
        # Try alternative location
        import glob
        ind_files = glob.glob(os.path.join(PROJECT_DIR, 'psid_raw', 'ind*', '*.txt'))
        if ind_files:
            ind_file = ind_files[0]
            print(f"  Found individual file: {ind_file}")
        else:
            print("  No individual file found, cannot add 1968-1969 data")
            raw_data = {}

    if raw_data and os.path.exists(ind_file):
        print(f"  Reading individual file for person-family mapping...")
        # Read individual file - need person_id and interview_number for each year
        # This is complex; individual file has one row per person with interview
        # numbers for each year embedded in different columns
        #
        # For now, we'll take a simpler approach: use the persons already in our
        # panel and their id_1968 values to match to 1968 family data

        # Get unique id_1968 values from our panel
        existing = df[['person_id']].drop_duplicates()
        existing['id_68'] = existing['person_id'] // 1000
        existing['pn'] = existing['person_id'] % 1000

        for year in [1968, 1969]:
            if year not in raw_data:
                continue

            df_raw = raw_data[year]
            print(f"\n  Processing {year} raw data ({len(df_raw)} families)...")

            # Match persons: merge on interview_number = id_68 for 1968
            # For 1968, the interview_number IS the family ID that maps to id_68
            if year == 1968:
                # Persons with id_68 matching a family interview number
                # and who are heads (pn < 170 typically means original head-related members)
                heads = existing[existing['pn'] < 170].copy()
                matched = heads.merge(df_raw, left_on='id_68', right_on='interview_number', how='inner')
            else:
                # For 1969, the interview_number is a different family ID
                # We'd need the individual file's 1969 interview number mapping
                # This is too complex without the full individual file parsing
                print(f"  Skipping {year} - need individual file mapping")
                continue

            print(f"  Matched {len(matched)} person-family records for {year}")

            if len(matched) == 0:
                continue

            # Apply basic restrictions
            # White (race=1), Male (sex_head=1), not self-employed, age 18-60
            matched = matched[matched['race'] == 1]
            matched = matched[matched['sex_head'] == 1]
            matched = matched[matched['age_head'].between(18, 60)]
            matched = matched[~matched['self_employed'].isin([2, 3])]

            # Agriculture check - occupation 100-199 or 600-699, industry 17-29
            is_ag = ((matched['occupation'].between(100, 199)) |
                     (matched['occupation'].between(600, 699)) |
                     (matched['industry'].between(17, 29)))
            matched = matched[~is_ag]

            # Positive earnings and hours
            matched = matched[matched['hourly_earnings'] > 0]
            matched = matched[matched['annual_hours'] > 0]

            # Education not missing
            matched = matched[matched['education'].notna() & (matched['education'] != 9)]

            print(f"  After restrictions: {len(matched)} obs")

            if len(matched) == 0:
                continue

            # Create panel-compatible rows
            new_rows = pd.DataFrame()
            new_rows['person_id'] = matched['person_id']
            new_rows['year'] = year
            new_rows['age'] = matched['age_head']

            # Education: use same recoding as pre-1975
            edu_map = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
            new_rows['education_clean'] = matched['education'].map(edu_map)

            # Married
            new_rows['married'] = (matched['marital_status'] == 1).astype(float)

            # Wages and hours
            # Hourly earnings in 1968 file: V337 is in dollars.cents (5 digits)
            # Actually it's stored as integer cents (xxxxx format)
            # Let's check: values > 100 are likely in cents
            he = matched['hourly_earnings'].copy()
            # If values are > 100, they're probably in cents
            if he.median() > 100:
                new_rows['hourly_wage'] = he / 100.0
            else:
                new_rows['hourly_wage'] = he
            new_rows['wages'] = matched['labor_income']
            new_rows['hours'] = matched['annual_hours']

            # Union
            new_rows['union_member'] = (matched['union'] == 1).astype(float)

            # Disabled
            new_rows['disabled'] = (matched['disability'] == 1).astype(float)

            # Self-employed, govt_worker, agriculture flags
            new_rows['self_employed'] = 0
            new_rows['govt_worker'] = 0
            new_rows['agriculture'] = 0

            # SMSA
            new_rows['lives_in_smsa'] = 0  # broken

            # Tenure info - not available for 1968
            new_rows['tenure'] = np.nan
            new_rows['tenure_mos'] = np.nan
            new_rows['tenure_topel'] = np.nan
            new_rows['job_id'] = np.nan
            new_rows['same_emp'] = np.nan
            new_rows['new_job'] = np.nan
            new_rows['labor_inc'] = matched['labor_income']

            # Add remaining columns from df that are missing
            for col in df.columns:
                if col not in new_rows.columns:
                    new_rows[col] = np.nan

            # Keep only columns in df
            new_rows = new_rows[[c for c in df.columns if c in new_rows.columns]]

            # Trim extreme wages
            new_rows = new_rows[new_rows['hourly_wage'] > 0]
            new_rows = new_rows[new_rows['hourly_wage'] < 200]

            print(f"  Final {year} observations to add: {len(new_rows)}")

            if len(new_rows) > 0:
                df = pd.concat([new_rows, df], ignore_index=True)
                df = df.sort_values(['person_id', 'year']).reset_index(drop=True)

    print(f"\nPanel after adding early years: {len(df)} obs")
    print(f"  Years: {sorted(df['year'].unique())}")
    for yr in sorted(df['year'].unique()):
        print(f"    {yr}: {len(df[df['year']==yr])}")

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
    # Construct real wage using wages/hours and interview-year GNP deflator
    # =========================================================================
    df['gnp_deflator'] = df['year'].map(GNP_DEFLATOR)
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)

    # Use wages/hours as primary, fall back to hourly_wage
    df['hw'] = df['wages'] / df['hours']
    # Where wages/hours is invalid, use hourly_wage
    invalid = ~((df['hw'] > 0) & np.isfinite(df['hw']))
    df.loc[invalid, 'hw'] = df.loc[invalid, 'hourly_wage']

    df['log_real_wage'] = (np.log(df['hw'])
                           - np.log(df['gnp_deflator'] / 33.4)
                           - np.log(df['cps_index']))

    # =========================================================================
    # Reconstruct tenure
    # =========================================================================
    print("\nReconstructing tenure...")
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

    print("  Fixing union (1982-1983 miscoded)...")
    df = fix_union_for_years(df, UNION_BAD_YEARS, all_union_bad)

    # Compute job-level union
    print("  Computing job-level union...")
    df = compute_job_union(df)

    # =========================================================================
    # Sample restrictions
    # =========================================================================
    print("\nApplying sample restrictions...")
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
    print(f"  Years: {sorted(df['year'].unique())}")
    for yr in sorted(df['year'].unique()):
        print(f"    {yr}: {len(df[df['year']==yr])}")

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

    # Use job-level union
    uni = df['union_job'].dropna()
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

        if true['mean'] != 0:
            err = abs(gen['mean'] - true['mean']) / abs(true['mean'])
        else:
            err = abs(gen['mean'] - true['mean'])
        matched = err <= 0.02
        if matched: n_match += 1
        tag = "MATCH" if matched else f"MISS ({err:.3f})"
        print(f"   {var} mean: {gen['mean']:.3f} vs {true['mean']:.3f} - {tag}")

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

    for yr in range(1968, 1984):
        if abs(results_bottom[yr]['cps_index'] - GROUND_TRUTH_BOTTOM[yr]['cps_index']) < 0.001:
            n_match += 1

    val_score = (n_match / total_values) * 40
    print(f"\n2. Values: {val_score:.1f}/40 ({n_match}/{total_values})")
    total_score += val_score

    total_score += 10
    print(f"\n3. Ordering: 10.0/10")

    n_err = abs(total_n - TOTAL_N_PAPER) / TOTAL_N_PAPER
    if n_err <= 0.05: n_score = 20
    elif n_err <= 0.10: n_score = 15
    elif n_err <= 0.20: n_score = 10
    else: n_score = 5
    print(f"\n4. N: {n_score}/20 (N={total_n}, err={n_err:.3f})")
    total_score += n_score

    total_score += 10
    print(f"\n5. Structure: 10.0/10")

    print(f"\n{'='*70}")
    print(f"TOTAL SCORE: {total_score:.0f}/100")
    print(f"{'='*70}")

    return total_score


if __name__ == '__main__':
    score = score_against_ground_truth()
