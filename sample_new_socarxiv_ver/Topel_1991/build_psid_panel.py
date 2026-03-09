#!/usr/bin/env python3
"""
Build PSID person-year panel dataset for Topel (1991) replication.

Reads raw PSID fixed-width family files (1968-1983) and the cross-year
individual file, constructs required variables, applies sample restrictions,
and outputs a panel dataset.

Usage:
    .venv/bin/python build_psid_panel.py
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from collections import OrderedDict

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PSID_RAW = os.path.join(BASE_DIR, 'psid_raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'psid_panel.csv')

YEARS = list(range(1968, 1984))

# =============================================================================
# Part 1: Parse .do files to extract column positions
# =============================================================================

def parse_do_file(do_path):
    """
    Parse a Stata .do file to extract:
      - Variable column positions from the 'infix' block
      - Variable labels from 'label variable' commands
    Returns:
        colspecs: dict of {varname: (start, end)} with 0-based positions for pd.read_fwf
        labels: dict of {varname: label_string}
    """
    with open(do_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Parse infix block
    # Find the infix ... using block
    infix_match = re.search(r'infix\s*\n(.*?)using', content, re.DOTALL | re.IGNORECASE)
    colspecs = OrderedDict()
    if infix_match:
        infix_text = infix_match.group(1)
        # Match patterns like: V1   1 - 1   or   long ER30033   68 - 72
        # Variable names can be V### or ER#####, optionally preceded by 'long'
        pattern = r'(?:long\s+)?(\w+)\s+(\d+)\s*-\s*(\d+)'
        for m in re.finditer(pattern, infix_text):
            varname = m.group(1).upper()
            start = int(m.group(2)) - 1  # Convert to 0-based
            end = int(m.group(3))        # end is exclusive in pandas
            colspecs[varname] = (start, end)

    # Parse variable labels
    labels = {}
    label_pattern = r'label\s+variable\s+(\w+)\s+"([^"]*)"'
    for m in re.finditer(label_pattern, content, re.IGNORECASE):
        varname = m.group(1).upper()
        label = m.group(2).strip()
        labels[varname] = label

    return colspecs, labels


def read_fixed_width(data_path, colspecs_dict, usecols=None):
    """
    Read a fixed-width file using column specs from the .do file.

    Args:
        data_path: path to the ASCII data file
        colspecs_dict: OrderedDict of {varname: (start, end)}
        usecols: list of variable names to read (None = all)
    Returns:
        DataFrame with requested columns
    """
    if usecols is not None:
        # Filter to only the columns we need
        usecols_upper = [c.upper() for c in usecols]
        filtered = OrderedDict()
        for var, spec in colspecs_dict.items():
            if var.upper() in usecols_upper:
                filtered[var] = spec
        if not filtered:
            print(f"  WARNING: None of the requested columns found in {data_path}")
            return pd.DataFrame()
        names = list(filtered.keys())
        specs = list(filtered.values())
    else:
        names = list(colspecs_dict.keys())
        specs = list(colspecs_dict.values())

    df = pd.read_fwf(data_path, colspecs=specs, names=names, header=None)
    return df


# =============================================================================
# Part 2: Variable mapping by year
# =============================================================================

# Each year maps concept -> variable name in that year's family file
# The variable names are the V-numbers from the .do files

FAMILY_VARS = {
    1968: {
        'fam_id': 'V2',         # Interview number 68
        'age': 'V117',          # Age of head
        'sex': 'V119',          # Sex of head (1=M, 2=F)
        'race': 'V181',         # Race (1=White, 2=Negro, 3=Other)
        'education': 'V313',    # Head's education (years)
        'marital': 'V239',      # Marital status
        'labor_inc': 'V74',     # Head's labor income (1967)
        'wages': 'V74',         # Head's labor income (no separate wage var in 1968)
        'hours': 'V47',         # Yearly head's hours (1967)
        'self_emp': 'V198',     # Self-employed (0=no, 1=yes)
        'occupation': 'V197',   # Occupation (1-digit); V197_A = 3-digit
        # Industry: no base var; use V197_B (3-digit) handled in standardize_family_vars
        'disability': 'V216',   # Disability
        'region': 'V361',       # Region now
        'smsa': 'V188',         # Nearest SMSA
        'union': 'V294',        # Union and dues
        'tenure': 'V200',       # How long with employer
        'emp_status': 'V196',   # Working now
    },
    1969: {
        'fam_id': 'V442',
        'age': 'V1008',
        'sex': 'V1010',
        'race': 'V801',
        'education': 'V794',
        'marital': 'V607',
        'labor_inc': 'V514',
        'wages': 'V514',
        'hours': 'V465',
        'self_emp': 'V641',
        'occupation': 'V640',
        # Industry: no separate base var; use V640_B (3-digit) handled in standardize_family_vars
        'disability': 'V743',
        'region': 'V876',
        'smsa': 'V539',
        'union': 'V766',
        'tenure': 'V642',
        'emp_status': 'V639',
    },
    1970: {
        'fam_id': 'V1102',
        'age': 'V1239',
        'sex': 'V1240',
        'race': 'V1490',
        'education': 'V1485',
        'marital': 'V1365',
        'labor_inc': 'V1196',
        'wages': 'V1191',
        'hours': 'V1138',
        'self_emp': 'V1280',
        'occupation': 'V1279',
        # Industry: no separate base var; use V1279_B (3-digit) handled in standardize_family_vars
        'disability': 'V1409',
        'region': 'V1572',
        'smsa': 'V1506',
        'union': 'V1434',
        'tenure': 'V1281',
        'emp_status': 'V1278',
    },
    1971: {
        'fam_id': 'V1802',
        'age': 'V1942',
        'sex': 'V1943',
        'race': 'V2202',
        'education': 'V2197',
        'marital': 'V2072',
        'labor_inc': 'V1897',
        'wages': 'V1892',
        'hours': 'V1839',
        'self_emp': 'V1986',
        'occupation': 'V1984',
        'industry': 'V1985',
        'disability': 'V2121',
        'region': 'V2284',
        'smsa': 'V1816',
        'union': 'V2145',
        'tenure': 'V1987',
        'emp_status': 'V1983',
    },
    1972: {
        'fam_id': 'V2402',
        'age': 'V2542',
        'sex': 'V2543',
        'race': 'V2828',
        'education': 'V2823',
        'marital': 'V2670',
        'labor_inc': 'V2498',
        'wages': 'V2493',
        'hours': 'V2439',
        'self_emp': 'V2584',
        'occupation': 'V2582',
        'industry': 'V2583',
        'disability': 'V2718',
        'region': 'V2911',
        'smsa': 'V2406',
        'union': 'V2787',
        'tenure': 'V2585',
        'emp_status': 'V2581',
    },
    1973: {
        'fam_id': 'V3002',
        'age': 'V3095',
        'sex': 'V3096',
        'race': 'V3300',
        'education': 'V3241',
        'marital': 'V3181',
        'labor_inc': 'V3051',
        'wages': 'V3046',
        'hours': 'V3027',
        'self_emp': 'V3117',
        'occupation': 'V3115',
        'industry': 'V3116',
        'disability': 'V3244',
        'region': 'V3279',
        'smsa': 'V3006',
        'union': None,          # Not asked in 1973
        'tenure': None,         # Not directly available
        'emp_status': 'V3114',
    },
    1974: {
        'fam_id': 'V3402',
        'age': 'V3508',
        'sex': 'V3509',
        'race': 'V3720',
        'education': 'V3663',
        'marital': 'V3598',
        'labor_inc': 'V3463',
        'wages': 'V3458',
        'hours': 'V3423',
        'self_emp': 'V3532',
        'occupation': 'V3530',
        'industry': 'V3531',
        'disability': 'V3666',
        'region': 'V3699',
        'smsa': 'V3406',
        'union': None,          # Not asked in 1974
        'tenure': None,         # Not directly available
        'emp_status': 'V3528',
    },
    1975: {
        'fam_id': 'V3802',
        'age': 'V3921',
        'sex': 'V3922',
        'race': 'V4204',
        'education': 'V4093',
        'marital': 'V3815',     # FU composition (need to decode)
        'labor_inc': 'V3863',
        'wages': 'V3858',
        'hours': 'V3823',
        'self_emp': 'V3970',
        'occupation': 'V3968',
        'industry': 'V3969',
        'disability': 'V4146',
        'region': 'V4178',
        'smsa': 'V3806',
        'union': 'V4087',
        'tenure': None,
        'emp_status': 'V3967',
        'govt': 'V3971',
    },
    1976: {
        'fam_id': 'V4302',
        'age': 'V4436',
        'sex': 'V4437',
        'race': 'V5096',
        'education': 'V4684',
        'marital': 'V4603',
        'labor_inc': 'V5031',
        'wages': 'V4373',
        'hours': 'V4332',
        'self_emp': 'V4461',     # "WRK SOMEONEELSE/SELF D5": 1=someone else, 2=self
        'occupation': 'V4459',
        'industry': 'V4460',
        'disability': 'V4625',
        'region': 'V5054',
        'smsa': 'V4306',
        'union': 'V4624',
        'tenure': 'V4480',       # How long pres employer
        'tenure_mos': 'V4863',   # # months worked this employer
        'emp_status': 'V4458',
        'govt': 'V4462',
    },
    1977: {
        'fam_id': 'V5202',
        'age': 'V5350',
        'sex': 'V5351',
        'race': 'V5662',
        'education': 'V5647',
        'marital': 'V5502',
        'labor_inc': 'V5627',
        'wages': 'V5283',
        'hours': 'V5232',
        'self_emp': 'V5376',
        'occupation': 'V5374',
        'industry': 'V5375',
        'disability': 'V5560',
        'region': 'V5633',
        'smsa': 'V5206',
        'union': 'V5559',
        'tenure_mos': 'V5392',   # months worked for employer (both)
        'emp_status': 'V5373',
        'govt': 'V5377',         # OTR employer govt; V5385 = BTH
    },
    1978: {
        'fam_id': 'V5702',
        'age': 'V5850',
        'sex': 'V5851',
        'race': 'V6209',
        'education': 'V6194',
        'marital': 'V6197',
        'labor_inc': 'V6174',
        'wages': 'V5782',
        'hours': 'V5731',
        'self_emp': 'V5875',
        'occupation': 'V5873',
        'industry': 'V5874',
        'disability': 'V6102',
        'region': 'V6180',
        'smsa': 'V5706',
        'union': 'V6101',
        'emp_status': 'V5872',
        'govt': 'V5876',
    },
    1979: {
        'fam_id': 'V6302',
        'age': 'V6462',
        'sex': 'V6463',
        'race': 'V6802',
        'education': 'V6787',
        'marital': 'V6790',
        'labor_inc': 'V6767',
        'wages': 'V6391',
        'hours': 'V6336',
        'self_emp': 'V6493',
        'occupation': 'V6497',
        'industry': 'V6498',
        'disability': 'V6710',
        'region': 'V6773',
        'smsa': 'V6306',
        'union': 'V6707',
        'emp_status': 'V6492',
        'govt': 'V6494',
    },
    1980: {
        'fam_id': 'V6902',
        'age': 'V7067',
        'sex': 'V7068',
        'race': 'V7447',
        'education': 'V7433',
        'marital': 'V7435',
        'labor_inc': 'V7413',
        'wages': 'V6981',
        'hours': 'V6934',
        'self_emp': 'V7096',
        'occupation': 'V7100',
        'industry': 'V7101',
        'disability': 'V7343',
        'region': 'V7419',
        'smsa': 'V6906',
        'union': 'V7340',
        'tenure_mos': 'V7102',   # C9 # months this job
        'emp_status': 'V7095',
        'govt': 'V7097',
    },
    1981: {
        'fam_id': 'V7502',
        'age': 'V7658',
        'sex': 'V7659',
        'race': 'V8099',
        'education': 'V8085',
        'marital': 'V8087',
        'labor_inc': 'V8066',
        'wages': 'V7573',
        'hours': 'V7530',
        'self_emp': 'V7707',     # C2 work self/other
        'occupation': 'V7712',
        'industry': 'V7713',
        'disability': 'V7974',
        'region': 'V8071',
        'smsa': 'V7506',
        'union': 'V7971',
        'tenure_mos': 'V7711',   # C6 # months this employer
        'same_emp': 'V7779',     # C77 return same employer
        'emp_status': 'V7706',   # C1 employment status
        'govt': 'V7708',
    },
    1982: {
        'fam_id': 'V8202',
        'age': 'V8352',
        'sex': 'V8353',
        'race': 'V8723',
        'education': 'V8709',
        'marital': 'V8711',
        'labor_inc': 'V8690',
        'wages': 'V8265',
        'hours': 'V8228',
        'self_emp': 'V8375',     # C2 work self/other
        'occupation': 'V8380',
        'industry': 'V8381',
        'disability': 'V8616',
        'region': 'V8695',
        'smsa': 'V8206',
        'union': 'V8378',        # C5 belong union
        'tenure_mos': 'V8379',   # C6 # months this employer
        'same_emp': 'V8444',     # C71 return same employer
        'emp_status': 'V8374',   # C1 employment status
        'govt': 'V8376',
    },
    1983: {
        'fam_id': 'V8802',
        'age': 'V8961',
        'sex': 'V8962',
        'race': 'V9408',
        'education': 'V9395',
        'marital': 'V9419',
        'labor_inc': 'V9376',
        'wages': 'V8873',
        'hours': 'V8830',
        'self_emp': 'V9006',     # C2 work self/other
        'occupation': 'V9011',
        'industry': 'V9012',
        'disability': 'V9290',
        'region': 'V9381',
        'smsa': 'V8806',
        'union': 'V9009',        # C5 belong union
        'tenure_mos': 'V9010',   # C6 # months this employer
        'same_emp': 'V9075',     # C72 return same employer
        'emp_status': 'V9005',   # C1 employment status
        'govt': 'V9007',
    },
}

# Cross-year individual file variable mapping
IND_VARS = {
    'id_68': 'ER30001',      # 1968 interview number
    'pn': 'ER30002',          # Person number within family
    'sex': 'ER32000',         # Sex of individual (1=M, 2=F)
}

# For each year: (interview_number_var, sequence_number_var, relationship_to_head_var)
IND_YEAR_VARS = {
    1968: ('ER30001', None,      'ER30003'),  # 1968 has no sequence number
    1969: ('ER30020', 'ER30021', 'ER30022'),
    1970: ('ER30043', 'ER30044', 'ER30045'),
    1971: ('ER30067', 'ER30068', 'ER30069'),
    1972: ('ER30091', 'ER30092', 'ER30093'),
    1973: ('ER30117', 'ER30118', 'ER30119'),
    1974: ('ER30138', 'ER30139', 'ER30140'),
    1975: ('ER30160', 'ER30161', 'ER30162'),
    1976: ('ER30188', 'ER30189', 'ER30190'),
    1977: ('ER30217', 'ER30218', 'ER30219'),
    1978: ('ER30246', 'ER30247', 'ER30248'),
    1979: ('ER30283', 'ER30284', 'ER30285'),
    1980: ('ER30313', 'ER30314', 'ER30315'),
    1981: ('ER30343', 'ER30344', 'ER30345'),
    1982: ('ER30373', 'ER30374', 'ER30375'),
    1983: ('ER30399', 'ER30400', 'ER30401'),
}


# =============================================================================
# Part 3: Read and process data
# =============================================================================

def load_family_file(year):
    """Load a single year's family file using the .do file for column specs."""
    fam_dir = os.path.join(PSID_RAW, f'fam{year}')
    do_file = os.path.join(fam_dir, f'FAM{year}.do')
    data_file = os.path.join(fam_dir, f'FAM{year}.txt')

    if not os.path.exists(data_file):
        # Try lowercase
        data_file = os.path.join(fam_dir, f'fam{year}.txt')
    if not os.path.exists(data_file):
        print(f"  ERROR: Data file not found for {year}")
        return None

    print(f"  Parsing .do file: {do_file}")
    colspecs, labels = parse_do_file(do_file)
    print(f"    Found {len(colspecs)} variables, {len(labels)} labels")

    # Determine which variables to read for this year
    year_vars = FAMILY_VARS[year]
    needed_vars = [v for v in year_vars.values() if v is not None]

    # Also try to get the _A and _B suffix versions for occupation and industry
    occ_var = year_vars.get('occupation')
    ind_var = year_vars.get('industry')
    if occ_var:
        for suffix in ['_A', '_B']:
            candidate = occ_var + suffix
            if candidate in colspecs:
                needed_vars.append(candidate)
    if ind_var and ind_var != occ_var:
        for suffix in ['_A', '_B']:
            candidate = ind_var + suffix
            if candidate in colspecs:
                needed_vars.append(candidate)
    # For 1968-1970, industry is in the _B suffix of the occupation variable
    # (already handled by the occ_var suffix loop above, but be explicit)
    if year in [1968, 1969, 1970] and occ_var:
        for sfx in ['_A', '_B']:
            cand = occ_var + sfx
            if cand in colspecs:
                needed_vars.append(cand)

    # Remove duplicates while preserving order
    seen = set()
    unique_vars = []
    for v in needed_vars:
        if v not in seen:
            seen.add(v)
            unique_vars.append(v)

    # Filter to only vars that actually exist in the colspecs
    available = [v for v in unique_vars if v in colspecs]
    missing = [v for v in unique_vars if v not in colspecs]
    if missing:
        print(f"    WARNING: Variables not found in .do file: {missing}")

    print(f"  Reading data file: {data_file}")
    print(f"    Reading {len(available)} variables: {available[:5]}...")
    df = read_fixed_width(data_file, colspecs, usecols=available)
    print(f"    Loaded {len(df)} rows")

    return df, colspecs, labels


def load_individual_file():
    """Load the cross-year individual file."""
    ind_dir = os.path.join(PSID_RAW, 'ind2023er')
    do_file = os.path.join(ind_dir, 'IND2023ER.do')
    data_file = os.path.join(ind_dir, 'IND2023ER.txt')

    if not os.path.exists(data_file):
        data_file = os.path.join(ind_dir, 'ind2023er.txt')

    print(f"  Parsing individual .do file: {do_file}")
    colspecs, labels = parse_do_file(do_file)
    print(f"    Found {len(colspecs)} variables")

    # Collect all needed variables
    needed = list(IND_VARS.values())
    for year in YEARS:
        int_var, seq_var, rel_var = IND_YEAR_VARS[year]
        needed.append(int_var)
        if seq_var:
            needed.append(seq_var)
        needed.append(rel_var)

    # Remove duplicates
    seen = set()
    unique_needed = []
    for v in needed:
        if v not in seen:
            seen.add(v)
            unique_needed.append(v)

    available = [v for v in unique_needed if v in colspecs]
    missing = [v for v in unique_needed if v not in colspecs]
    if missing:
        print(f"    WARNING: Individual file vars not found: {missing}")

    print(f"  Reading individual file: {data_file}")
    print(f"    Reading {len(available)} variables")
    df = read_fixed_width(data_file, colspecs, usecols=available)
    print(f"    Loaded {len(df)} rows (individuals)")

    return df


def standardize_family_vars(df, year, colspecs):
    """
    Rename family file variables to standardized names for the given year.
    Returns a new DataFrame with standardized column names.
    """
    year_vars = FAMILY_VARS[year]
    rename_map = {}
    for concept, varname in year_vars.items():
        if varname is not None and varname in df.columns:
            rename_map[varname] = concept

    # Handle occupation and industry with suffix variables
    occ_var = year_vars.get('occupation')
    ind_var = year_vars.get('industry')

    # For 3-digit occupation: try _A suffix first
    if occ_var:
        occ_3dig = occ_var + '_A'
        if occ_3dig in df.columns:
            rename_map[occ_3dig] = 'occupation_3dig'
        elif occ_var in df.columns and 'occupation' not in rename_map.values():
            pass  # Already handled above

    # For 3-digit industry
    # In 1968-1970, occupation and industry share a base variable with _A/_B suffixes
    if year in [1968, 1969, 1970]:
        if occ_var:
            occ_b = occ_var + '_B'
            occ_a = occ_var + '_A'
            if occ_b in df.columns:
                rename_map[occ_b] = 'industry_3dig'
            if occ_a in df.columns:
                rename_map[occ_a] = 'occupation_3dig'
    elif ind_var:
        ind_3dig_a = ind_var + '_A'
        ind_3dig_b = ind_var + '_B'
        if ind_3dig_a in df.columns:
            rename_map[ind_3dig_a] = 'industry_3dig'
        elif ind_3dig_b in df.columns:
            rename_map[ind_3dig_b] = 'industry_3dig'

    # For occupation, if we have a separate _A variable and the base is already
    # mapped to 'occupation', also keep the 3-digit version
    # In some years, occupation base variable IS the 1-digit code
    # In other years (1979+), the base variable has a different meaning

    result = df.rename(columns=rename_map)

    # Handle special cases
    # For 1976, the self_emp variable is actually the employment status
    if year == 1976:
        # V4458 is employment status, not self-employment
        # We need to check industry/occupation to determine self-employment
        # For now, mark as needing special handling
        pass

    return result


def recode_self_employment(df, year):
    """
    Recode self-employment variable to a consistent 0/1 coding.
    0 = not self-employed, 1 = self-employed
    """
    if 'self_emp' not in df.columns:
        df['self_employed'] = np.nan
        return df

    se = df['self_emp'].copy()

    if year == 1968:
        # 0=working for someone else, 1=self-employed, 9=NA
        df['self_employed'] = np.where(se == 1, 1,
                              np.where(se == 0, 0, np.nan))
    elif year in [1969, 1970, 1971, 1972]:
        # 1=someone else, 2=both, 3=self only, 0/9=NA
        df['self_employed'] = np.where(se.isin([2, 3]), 1,
                              np.where(se == 1, 0, np.nan))
    elif year in [1973, 1974, 1975]:
        # 1=someone else, 2=self-employed, 9=NA
        df['self_employed'] = np.where(se == 2, 1,
                              np.where(se == 1, 0, np.nan))
    elif year == 1976:
        # V4461 "WRK SOMEONEELSE/SELF D5": 1=someone else, 2=self, 3=both
        df['self_employed'] = np.where(se.isin([2, 3]), 1,
                              np.where(se == 1, 0, np.nan))
    elif year in range(1977, 1984):
        # C2/D5: 1=someone else, 2=self, 3=both
        df['self_employed'] = np.where(se.isin([2, 3]), 1,
                              np.where(se == 1, 0, np.nan))

    return df


def recode_marital(df, year):
    """
    Recode marital status to married dummy (1=married, spouse present).
    """
    if 'marital' not in df.columns:
        df['married'] = np.nan
        return df

    ms = df['marital'].copy()

    if year == 1975:
        # Family composition variable; first digit indicates wife presence
        # Composition codes: 1=Head and Wife only, 2=H+W+kids, etc.
        # If first digit < 4 (roughly: has wife), mark as married
        # Actually, the composition codes for 1975:
        # 1 = Head only, 2 = H + wife only, 3 = H + wife + others,
        # 4 = H + others (no wife)
        # We'll be conservative: check if value indicates wife present
        df['married'] = np.where(ms.isin([2, 3]), 1, 0)
    else:
        # Standard marital status coding:
        # 1 = Married, spouse present
        # 2 = Single, never married
        # 3 = Widowed
        # 4 = Divorced, annulled
        # 5 = Separated
        df['married'] = np.where(ms == 1, 1, 0)

    return df


def recode_race(df, year):
    """
    Recode race to white dummy (1=white).
    """
    if 'race' not in df.columns:
        df['white'] = np.nan
        return df

    r = df['race'].copy()
    # All years: 1=White, 2=Black/Negro, 3+=Other
    df['white'] = np.where(r == 1, 1,
                  np.where(r.isin([0, 7, 8, 9]), np.nan, 0))
    return df


def recode_union(df, year):
    """
    Recode union membership to 0/1 dummy.
    """
    if 'union' not in df.columns:
        df['union_member'] = np.nan
        return df

    u = df['union'].copy()

    if year == 1968:
        # V294 "UNION AND DUES": 1=yes, member; 5=no; 0/9=NA
        df['union_member'] = np.where(u == 1, 1,
                             np.where(u == 5, 0, np.nan))
    elif year in [1969, 1970, 1971, 1972]:
        # 1=yes, 5=no, 0/9=NA
        df['union_member'] = np.where(u == 1, 1,
                             np.where(u == 5, 0, np.nan))
    elif year in [1975, 1976, 1977, 1978, 1979, 1980, 1981]:
        # 1=yes, 5=no, 0/8/9=NA
        df['union_member'] = np.where(u == 1, 1,
                             np.where(u == 5, 0, np.nan))
    elif year in [1982, 1983]:
        # C5 belong union: 1=yes, 5=no, 0/9=NA
        df['union_member'] = np.where(u == 1, 1,
                             np.where(u == 5, 0, np.nan))
    else:
        df['union_member'] = np.nan

    return df


def recode_disability(df, year):
    """
    Recode disability/health limitation to 0/1 dummy.
    """
    if 'disability' not in df.columns:
        df['disabled'] = np.nan
        return df

    d = df['disability'].copy()

    if year == 1968:
        # V216: 1=yes, 5=no, 0/9=NA
        df['disabled'] = np.where(d == 1, 1,
                         np.where(d == 5, 0, np.nan))
    elif year in [1969, 1970, 1971, 1972]:
        # "DISAB LIM KIND WRK": 1=yes (limits type), 3=limits amount,
        # 5=not at all, 0/9=NA
        df['disabled'] = np.where(d.isin([1, 3]), 1,
                         np.where(d == 5, 0, np.nan))
    elif year in [1973, 1974, 1975]:
        # Health limits type of work: 1=yes, 3=partially, 5=no
        df['disabled'] = np.where(d.isin([1, 3]), 1,
                         np.where(d == 5, 0, np.nan))
    elif year in [1976, 1977, 1978, 1979, 1980]:
        # Physical/nervous condition or health limit: 1=yes, 5=no
        df['disabled'] = np.where(d == 1, 1,
                         np.where(d == 5, 0, np.nan))
    elif year in [1981, 1982, 1983]:
        # "WTR PHYS-NERV PROB-H": 1=yes, 5=no
        df['disabled'] = np.where(d == 1, 1,
                         np.where(d == 5, 0, np.nan))
    else:
        df['disabled'] = np.nan

    return df


def recode_government(df, year):
    """
    Recode government employment to 0/1 dummy.
    """
    if 'govt' not in df.columns:
        df['govt_worker'] = np.nan
        return df

    g = df['govt'].copy()

    if year == 1975:
        # "EMP BY GOVT? (E)": 1=federal, 2=state, 3=local, 4=other, 5=no, 9=NA
        df['govt_worker'] = np.where(g.isin([1, 2, 3]), 1,
                            np.where(g.isin([4, 5]), 0, np.nan))
    elif year == 1976:
        # "FED STATE OR LOC GOV D6": 1=federal, 2=state, 3=local, 4=other
        # 0=not government
        df['govt_worker'] = np.where(g.isin([1, 2, 3]), 1,
                            np.where(g.isin([0, 4]), 0, np.nan))
    elif year in range(1977, 1984):
        # "WRK FOR GOVT? / C3": 1=yes, 5=no, 0/9=NA
        df['govt_worker'] = np.where(g == 1, 1,
                            np.where(g == 5, 0, np.nan))

    return df


def recode_smsa(df, year):
    """
    Recode SMSA residence to 0/1 dummy.
    Lives in SMSA = 1, not in SMSA = 0.
    """
    if 'smsa' not in df.columns:
        df['lives_in_smsa'] = np.nan
        return df

    s = df['smsa'].copy()

    if year == 1968:
        # V188 "NEAREST SMSA": 0=lives in SMSA, 1-9=distance categories
        df['lives_in_smsa'] = np.where(s == 0, 1,
                              np.where(s.isin([9]), np.nan, 0))
    elif year in [1969, 1970, 1971, 1972, 1973, 1974, 1975]:
        # "LRGST PLAC/SMSA PSU" or similar
        # Coding varies; typically: 0=not in SMSA, 1-6=SMSA size categories
        # For 1969-1975: 0=not SMSA; 1+=SMSA
        # But some years use 0=rural, 9=NA
        # Check if there's a "NEAREST SMSA" variable (coded 0=in SMSA)
        # For safety, use > 0 and < 9 as in SMSA
        if year in [1969, 1970, 1971, 1972]:
            # LRGST PLAC/SMSA PSU:
            # 0=not in SMSA; 1-6=SMSA size categories
            df['lives_in_smsa'] = np.where(s > 0, 1,
                                  np.where(s == 0, 0, np.nan))
        else:
            # 1973-1975: Similar coding
            df['lives_in_smsa'] = np.where((s > 0) & (s < 9), 1,
                                  np.where(s == 0, 0, np.nan))
    elif year == 1976:
        # "SIZE LGST CITY PSU": 0=not in SMSA, 1-8=SMSA sizes
        df['lives_in_smsa'] = np.where((s > 0) & (s < 9), 1,
                              np.where(s == 0, 0, np.nan))
    elif year in range(1977, 1984):
        # "SIZE LGST CTY SMSA": 0=not in SMSA, 1-8=SMSA size categories
        # 9=NA
        df['lives_in_smsa'] = np.where((s > 0) & (s < 9), 1,
                              np.where(s == 0, 0, np.nan))

    return df


def recode_region(df, year):
    """
    Create census region indicators (NE, NC, S, W).
    Region coding: 1=NE, 2=NC, 3=South, 4=West, 5=AK/HI (some years), 0/9=NA
    """
    if 'region' not in df.columns:
        for r in ['region_ne', 'region_nc', 'region_south', 'region_west']:
            df[r] = np.nan
        return df

    r = df['region'].copy()
    df['region_ne'] = np.where(r == 1, 1, np.where(r.isin([2, 3, 4]), 0, np.nan))
    df['region_nc'] = np.where(r == 2, 1, np.where(r.isin([1, 3, 4]), 0, np.nan))
    df['region_south'] = np.where(r == 3, 1, np.where(r.isin([1, 2, 4]), 0, np.nan))
    df['region_west'] = np.where(r == 4, 1, np.where(r.isin([1, 2, 3]), 0, np.nan))
    return df


def recode_occupation_1digit(df, year):
    """
    Create 1-digit occupation from 3-digit census codes.
    Also create broad occupation dummies.
    """
    if 'occupation_3dig' in df.columns:
        occ3 = df['occupation_3dig'].copy()
        # First digit of 3-digit code gives broad category
        df['occ_1digit'] = occ3 // 100
    elif 'occupation' in df.columns:
        occ = df['occupation'].copy()
        # In early years, the base variable is already 1-digit
        df['occ_1digit'] = occ
    else:
        df['occ_1digit'] = np.nan

    return df


def detect_agriculture(df, year):
    """
    Detect if head works in agriculture based on industry codes.
    Agriculture = Census industry codes 017-029 (3-digit).
    For 1-digit industry: category 1 = agriculture typically.
    """
    if 'industry_3dig' in df.columns:
        ind3 = df['industry_3dig'].copy()
        df['agriculture'] = np.where((ind3 >= 17) & (ind3 <= 29), 1, 0)
    elif 'industry' in df.columns:
        # For years without 3-digit, use 1-digit or broad categories
        # Agriculture is typically industry code 0 or 1 at broad level
        # This is approximate
        ind = df['industry'].copy()
        df['agriculture'] = np.where(ind.isin([0, 1]), 1, 0)
    else:
        df['agriculture'] = np.nan

    return df


# =============================================================================
# Part 4: Main panel construction
# =============================================================================

def build_panel():
    """Build the complete person-year panel dataset."""

    print("=" * 70)
    print("BUILDING PSID PANEL FOR TOPEL (1991) REPLICATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load individual file
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading cross-year individual file...")
    ind_df = load_individual_file()

    # Filter to person number 001 in 1968 (original sample members from the
    # 1968 cross-section) -- actually, we want all individuals but will filter
    # to heads later. The key identifier is (ER30001, ER30002) = (1968 ID, Person Number).
    # Create a unique person_id
    ind_df['person_id'] = ind_df['ER30001'] * 1000 + ind_df['ER30002']

    # Get sex (time-invariant)
    ind_df['ind_sex'] = ind_df['ER32000']

    # Filter to SRC sample: 1968 interview number 1-2930
    # (SEO/poverty sample: 3001-3511; Latino sample: 7001+; Immigrant: 4001+)
    print(f"  Total individuals: {len(ind_df)}")
    ind_df_src = ind_df[ind_df['ER30001'].between(1, 2930)].copy()
    print(f"  SRC sample individuals: {len(ind_df_src)}")

    # Filter to males
    ind_df_src_male = ind_df_src[ind_df_src['ind_sex'] == 1].copy()
    print(f"  SRC male individuals: {len(ind_df_src_male)}")

    # ------------------------------------------------------------------
    # Step 2: Load and process each year's family file
    # ------------------------------------------------------------------
    all_years_data = []

    for year in YEARS:
        print(f"\n[Step 2] Processing year {year}...")
        result = load_family_file(year)
        if result is None:
            continue
        fam_df, colspecs, labels = result

        # Standardize variable names
        fam_df = standardize_family_vars(fam_df, year, colspecs)

        # Fix 1968 family ID: V2 stores IDs as multiples of 10
        if year == 1968 and 'fam_id' in fam_df.columns:
            fam_df['fam_id'] = (fam_df['fam_id'] / 10).astype(int)

        # Add year column
        fam_df['year'] = year

        # Apply recodings
        fam_df = recode_self_employment(fam_df, year)
        fam_df = recode_marital(fam_df, year)
        fam_df = recode_race(fam_df, year)
        fam_df = recode_union(fam_df, year)
        fam_df = recode_disability(fam_df, year)
        fam_df = recode_government(fam_df, year)
        fam_df = recode_smsa(fam_df, year)
        fam_df = recode_region(fam_df, year)
        fam_df = recode_occupation_1digit(fam_df, year)
        fam_df = detect_agriculture(fam_df, year)

        # Compute hourly wage = labor_inc / hours
        if 'labor_inc' in fam_df.columns and 'hours' in fam_df.columns:
            fam_df['hourly_wage'] = np.where(
                (fam_df['hours'] > 0) & (fam_df['labor_inc'] > 0),
                fam_df['labor_inc'] / fam_df['hours'],
                np.nan
            )
            fam_df['log_hourly_wage'] = np.where(
                fam_df['hourly_wage'] > 0,
                np.log(fam_df['hourly_wage']),
                np.nan
            )
        else:
            fam_df['hourly_wage'] = np.nan
            fam_df['log_hourly_wage'] = np.nan

        # Compute experience = age - education - 6
        if 'age' in fam_df.columns and 'education' in fam_df.columns:
            edu = fam_df['education'].copy()
            # Clean education: values > 17 or == 99 are typically NA
            edu = np.where((edu >= 0) & (edu <= 17), edu, np.nan)
            fam_df['education_clean'] = edu
            fam_df['experience'] = np.where(
                (fam_df['age'] > 0) & (~np.isnan(edu)),
                fam_df['age'] - edu - 6,
                np.nan
            )
            # Ensure experience >= 0
            fam_df['experience'] = np.where(
                fam_df['experience'] < 0, 0, fam_df['experience']
            )
        else:
            fam_df['education_clean'] = np.nan
            fam_df['experience'] = np.nan

        # Now merge with individual file to get person_id
        # We need to match family file rows with individuals who are heads
        int_var, seq_var, rel_var = IND_YEAR_VARS[year]

        # Get the individual-level data for this year
        ind_year_cols = list(dict.fromkeys(['person_id', 'ER30001', 'ER30002', 'ind_sex', int_var, rel_var]))
        if seq_var and seq_var not in ind_year_cols:
            ind_year_cols.append(seq_var)
        ind_year = ind_df_src_male[ind_year_cols].copy()

        # Filter to heads of household for this year
        # Relationship to head coding changed over time:
        # 1968: 1=head
        # 1969-1982: 1=head
        # 1983: 10=head (2-digit codes introduced)
        if year >= 1983:
            heads = ind_year[ind_year[rel_var] == 10].copy()
        else:
            heads = ind_year[ind_year[rel_var] == 1].copy()

        # For years after 1968, also filter by sequence number (must be in FU)
        if seq_var:
            heads = heads[heads[seq_var].between(1, 20)].copy()

        # Filter to non-zero interview numbers
        heads = heads[heads[int_var] > 0].copy()

        print(f"    Male SRC heads for {year}: {len(heads)}")

        # Merge family data with individual data on interview number
        # Family file's fam_id matches the individual file's interview number for that year
        if 'fam_id' in fam_df.columns:
            merge_cols = list(dict.fromkeys(['person_id', 'ER30001', int_var]))
            merged = fam_df.merge(
                heads[merge_cols],
                left_on='fam_id',
                right_on=int_var,
                how='inner'
            )
            print(f"    Merged records: {len(merged)}")
        else:
            print(f"    WARNING: No fam_id variable for {year}")
            continue

        # Store the 1968 ID for SRC sample check
        merged['id_1968'] = merged['ER30001']

        # Select and standardize columns for the panel
        panel_cols = [
            'person_id', 'id_1968', 'year', 'fam_id',
            'age', 'sex', 'white', 'education_clean',
            'married', 'labor_inc', 'wages', 'hours',
            'hourly_wage', 'log_hourly_wage',
            'self_employed', 'experience',
            'occ_1digit', 'agriculture',
            'union_member', 'disabled',
            'lives_in_smsa', 'region',
            'region_ne', 'region_nc', 'region_south', 'region_west',
        ]

        # Add government worker if available
        if 'govt_worker' in merged.columns:
            panel_cols.append('govt_worker')

        # Add tenure-related variables if available
        for tc in ['tenure', 'tenure_mos', 'same_emp']:
            if tc in merged.columns:
                panel_cols.append(tc)

        # Filter to columns that exist
        existing_cols = [c for c in panel_cols if c in merged.columns]
        merged_panel = merged[existing_cols].copy()

        all_years_data.append(merged_panel)

    # ------------------------------------------------------------------
    # Step 3: Combine all years
    # ------------------------------------------------------------------
    print("\n[Step 3] Combining all years...")
    panel = pd.concat(all_years_data, ignore_index=True, sort=False)
    print(f"  Total person-year records: {len(panel)}")
    print(f"  Unique persons: {panel['person_id'].nunique()}")

    # ------------------------------------------------------------------
    # Step 4: Apply sample restrictions
    # ------------------------------------------------------------------
    print("\n[Step 4] Applying sample restrictions...")

    n0 = len(panel)

    # 4a. White males only
    # Sex should already be male from individual file filter, but double-check
    # from family file
    if 'sex' in panel.columns:
        panel = panel[panel['sex'].isin([1, np.nan])].copy()
    panel = panel[panel['white'] == 1].copy()
    print(f"  After white males: {len(panel)} (dropped {n0 - len(panel)})")
    n0 = len(panel)

    # 4b. Ages 18-60
    panel = panel[(panel['age'] >= 18) & (panel['age'] <= 60)].copy()
    print(f"  After age 18-60: {len(panel)} (dropped {n0 - len(panel)})")
    n0 = len(panel)

    # 4c. Positive earnings (labor income > 0)
    panel = panel[panel['labor_inc'] > 0].copy()
    print(f"  After positive earnings: {len(panel)} (dropped {n0 - len(panel)})")
    n0 = len(panel)

    # 4d. Not self-employed
    # If self_employed is NaN, we keep the observation (conservative)
    panel = panel[~(panel['self_employed'] == 1)].copy()
    print(f"  After not self-employed: {len(panel)} (dropped {n0 - len(panel)})")
    n0 = len(panel)

    # 4e. Not in agriculture
    panel = panel[~(panel['agriculture'] == 1)].copy()
    print(f"  After not agriculture: {len(panel)} (dropped {n0 - len(panel)})")
    n0 = len(panel)

    # 4f. Not in government (when data available)
    if 'govt_worker' in panel.columns:
        panel = panel[~(panel['govt_worker'] == 1)].copy()
        print(f"  After not government: {len(panel)} (dropped {n0 - len(panel)})")
        n0 = len(panel)

    # 4g. Positive hours (at least some threshold)
    panel = panel[panel['hours'] > 0].copy()
    print(f"  After positive hours: {len(panel)} (dropped {n0 - len(panel)})")
    n0 = len(panel)

    # 4h. Valid hourly wage
    panel = panel[panel['hourly_wage'] > 0].copy()
    print(f"  After valid hourly wage: {len(panel)} (dropped {n0 - len(panel)})")
    n0 = len(panel)

    # ------------------------------------------------------------------
    # Step 5: Construct job identifiers and tenure
    # ------------------------------------------------------------------
    print("\n[Step 5] Constructing job identifiers and tenure...")

    # Sort by person and year
    panel = panel.sort_values(['person_id', 'year']).reset_index(drop=True)

    # Method for detecting job changes:
    # 1. For years with "tenure with employer" (months): if tenure_mos < 12,
    #    likely a new job in the prior year
    # 2. For years with "same employer" question: if same_emp != 1, new job
    # 3. For early years with "HOW LONG HAD JOB": use coded values
    # 4. Fallback: use occupation changes as a proxy
    #
    # For Topel's approach: he uses a question about whether the worker changed
    # employers between interviews. We'll approximate this.

    # Create a "new job" indicator for each person-year
    panel['new_job'] = 0

    # Process each person
    for pid in panel['person_id'].unique():
        mask = panel['person_id'] == pid
        idx = panel.index[mask]
        person_data = panel.loc[idx].copy()

        if len(person_data) == 0:
            continue

        # First observation for a person always starts a new job
        panel.loc[idx[0], 'new_job'] = 1

        for i in range(1, len(person_data)):
            curr_year = person_data.iloc[i]['year']
            prev_year = person_data.iloc[i-1]['year']
            curr_idx = idx[i]

            # If years are not consecutive, assume new job
            if curr_year - prev_year > 1:
                panel.loc[curr_idx, 'new_job'] = 1
                continue

            # Check "same employer" question (1981-1983)
            if 'same_emp' in person_data.columns:
                same = person_data.iloc[i].get('same_emp', np.nan)
                if not np.isnan(same) if isinstance(same, float) else True:
                    try:
                        same = float(same)
                        if same == 5:  # No, different employer
                            panel.loc[curr_idx, 'new_job'] = 1
                            continue
                        elif same == 1:  # Yes, same employer
                            continue
                    except (ValueError, TypeError):
                        pass

            # Check tenure months (when available)
            if 'tenure_mos' in person_data.columns:
                mos = person_data.iloc[i].get('tenure_mos', np.nan)
                if not np.isnan(mos) if isinstance(mos, float) else True:
                    try:
                        mos = float(mos)
                        if mos > 0 and mos < 12:
                            panel.loc[curr_idx, 'new_job'] = 1
                            continue
                    except (ValueError, TypeError):
                        pass

            # For early years: check "HOW LONG HAD JOB" (tenure variable)
            if 'tenure' in person_data.columns:
                tenure_val = person_data.iloc[i].get('tenure', np.nan)
                prev_tenure = person_data.iloc[i-1].get('tenure', np.nan)
                if not (np.isnan(tenure_val) if isinstance(tenure_val, float) else False):
                    try:
                        tenure_val = float(tenure_val)
                        # Coded values: typically 1=<6mo, 2=6-11mo, 3=1-2yrs, etc.
                        # A value of 1 or 2 suggests new job
                        if tenure_val in [1, 2]:
                            panel.loc[curr_idx, 'new_job'] = 1
                            continue
                    except (ValueError, TypeError):
                        pass

            # Check for occupation change as fallback
            if 'occ_1digit' in person_data.columns:
                curr_occ = person_data.iloc[i].get('occ_1digit', np.nan)
                prev_occ = person_data.iloc[i-1].get('occ_1digit', np.nan)
                try:
                    curr_occ_f = float(curr_occ)
                    prev_occ_f = float(prev_occ)
                    if not np.isnan(curr_occ_f) and not np.isnan(prev_occ_f):
                        if curr_occ_f != prev_occ_f:
                            # Occupation change suggests possible job change
                            # but is not definitive -- leave as no change
                            pass
                except (ValueError, TypeError):
                    pass

    # Construct job_id: each new_job=1 starts a new job spell
    panel['job_id'] = 0
    job_counter = 0
    current_person = None

    for idx in panel.index:
        pid = panel.loc[idx, 'person_id']
        if pid != current_person:
            current_person = pid
            job_counter += 1
        elif panel.loc[idx, 'new_job'] == 1:
            job_counter += 1
        panel.loc[idx, 'job_id'] = job_counter

    # Construct tenure within job (years since job started)
    panel['tenure_topel'] = 0
    for jid in panel['job_id'].unique():
        mask = panel['job_id'] == jid
        job_data = panel.loc[mask].sort_values('year')
        for i, idx in enumerate(job_data.index):
            panel.loc[idx, 'tenure_topel'] = i

    # ------------------------------------------------------------------
    # Step 6: Additional restrictions per Topel
    # ------------------------------------------------------------------
    print("\n[Step 6] Additional restrictions...")

    n0 = len(panel)

    # Topel requires current job tenure >= 1 year
    # This means we keep observations where tenure_topel >= 1
    # (i.e., they've been observed in the same job for at least one prior year)
    # Actually, Topel's requirement is that the person has been with the
    # current employer for >= 1 year at the time of interview.
    # The tenure_topel variable counts years observed in the same job.
    # tenure_topel == 0 means first year observed in this job.
    # If tenure_topel >= 1, the person has been in the job for >= 1 year.
    # However, Topel might also use the raw tenure data.
    # For now, let's keep tenure_topel >= 1 as a reasonable proxy.
    panel_restricted = panel[panel['tenure_topel'] >= 1].copy()
    print(f"  After tenure >= 1 year: {len(panel_restricted)} (dropped {n0 - len(panel_restricted)})")

    # ------------------------------------------------------------------
    # Step 7: Create year dummies
    # ------------------------------------------------------------------
    print("\n[Step 7] Creating year dummies...")
    for y in YEARS:
        panel_restricted[f'year_{y}'] = np.where(panel_restricted['year'] == y, 1, 0)

    # Create occupation dummies (1-digit)
    for occ in range(10):
        panel_restricted[f'occ_{occ}'] = np.where(
            panel_restricted['occ_1digit'] == occ, 1, 0
        )

    # ------------------------------------------------------------------
    # Step 8: Compute first-difference and lag variables
    # ------------------------------------------------------------------
    print("\n[Step 8] Computing first-difference and lag variables...")

    panel_restricted = panel_restricted.sort_values(['person_id', 'year'])

    # Within-job differences
    panel_restricted['d_log_wage'] = panel_restricted.groupby('job_id')['log_hourly_wage'].diff()
    panel_restricted['d_experience'] = panel_restricted.groupby('person_id')['experience'].diff()

    # Lag variables
    panel_restricted['lag_tenure'] = panel_restricted.groupby('person_id')['tenure_topel'].shift(1)

    # Experience squared
    panel_restricted['experience_sq'] = panel_restricted['experience'] ** 2

    # Tenure squared
    panel_restricted['tenure_sq'] = panel_restricted['tenure_topel'] ** 2

    # ------------------------------------------------------------------
    # Step 9: Final summary and output
    # ------------------------------------------------------------------
    print("\n[Step 9] Summary statistics...")
    print(f"  Total person-year observations: {len(panel_restricted)}")
    print(f"  Unique persons: {panel_restricted['person_id'].nunique()}")
    print(f"  Unique jobs: {panel_restricted['job_id'].nunique()}")
    print(f"  Year range: {panel_restricted['year'].min()} - {panel_restricted['year'].max()}")
    print(f"  Mean age: {panel_restricted['age'].mean():.1f}")
    print(f"  Mean education: {panel_restricted['education_clean'].mean():.1f}")
    print(f"  Mean experience: {panel_restricted['experience'].mean():.1f}")
    print(f"  Mean tenure: {panel_restricted['tenure_topel'].mean():.1f}")
    print(f"  Mean log hourly wage: {panel_restricted['log_hourly_wage'].mean():.3f}")
    print(f"  Mean hourly wage: {panel_restricted['hourly_wage'].mean():.2f}")
    if 'married' in panel_restricted.columns:
        print(f"  % Married: {panel_restricted['married'].mean()*100:.1f}")
    if 'union_member' in panel_restricted.columns:
        print(f"  % Union member: {panel_restricted['union_member'].mean()*100:.1f}")
    if 'disabled' in panel_restricted.columns:
        print(f"  % Disabled: {panel_restricted['disabled'].mean()*100:.1f}")

    # Topel's Table 1 targets:
    print(f"\n  Paper targets: 1,540 individuals, 3,228 jobs, 13,128 job-years")

    # Year distribution
    print(f"\n  Observations by year:")
    year_counts = panel_restricted['year'].value_counts().sort_index()
    for y, n in year_counts.items():
        print(f"    {y}: {n}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    panel_restricted.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Panel saved to: {OUTPUT_FILE}")

    # Also save the full (unrestricted) panel for reference
    full_output = os.path.join(OUTPUT_DIR, 'psid_panel_full.csv')
    panel.to_csv(full_output, index=False)
    print(f"  Full panel (before tenure restriction) saved to: {full_output}")

    return panel_restricted


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    panel = build_panel()
