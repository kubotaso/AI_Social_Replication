#!/usr/bin/env python3
"""
Parse PSID .do files to extract variable names, column positions, and labels.
Output a comprehensive mapping document for all years 1968-1983.
"""

import re
import os
import json

BASE = "/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/psid_raw"

def parse_do_file(filepath):
    """Parse a Stata .do file to extract variable definitions and labels."""
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    # Extract infix definitions: variable_name  start - end
    # Handle both "long" prefix and regular vars
    infix_pattern = re.compile(r'(?:long\s+)?(\w+)\s+(\d+)\s*-\s*(\d+)')

    # Find the infix block
    infix_match = re.search(r'infix\s*\n(.*?)using', content, re.DOTALL)
    if not infix_match:
        # Try alternative pattern
        infix_match = re.search(r'infix\s+(.*?)using', content, re.DOTALL)

    col_specs = {}
    if infix_match:
        infix_block = infix_match.group(1)
        for m in infix_pattern.finditer(infix_block):
            varname = m.group(1)
            start = int(m.group(2))
            end = int(m.group(3))
            col_specs[varname] = (start, end)

    # Extract labels
    label_pattern = re.compile(r'label\s+variable\s+(\w+)\s+"([^"]*)"', re.IGNORECASE)
    labels = {}
    for m in label_pattern.finditer(content):
        varname = m.group(1)
        label = m.group(2).strip()
        labels[varname] = label

    return col_specs, labels


def find_matching_vars(labels, patterns, exclude_patterns=None):
    """Find variables whose labels match given patterns."""
    results = []
    for varname, label in sorted(labels.items()):
        label_upper = label.upper()
        matched = False
        for pat in patterns:
            if re.search(pat, label_upper):
                matched = True
                break
        if matched and exclude_patterns:
            for epat in exclude_patterns:
                if re.search(epat, label_upper):
                    matched = False
                    break
        if matched:
            results.append((varname, label))
    return results


def main():
    all_data = {}

    # Parse family files
    for year in range(1968, 1984):
        do_file = os.path.join(BASE, f"fam{year}", f"FAM{year}.do")
        if os.path.exists(do_file):
            col_specs, labels = parse_do_file(do_file)
            all_data[year] = {'col_specs': col_specs, 'labels': labels}
            print(f"\n{'='*80}")
            print(f"YEAR {year}: {len(col_specs)} variables, {len(labels)} labels")
            print(f"{'='*80}")

            # Search categories
            categories = {
                'INTERVIEW_NUMBER': [r'INTERVIEW.*NUMBER', r'INTERVIEW.*\#', r'^ID\b'],
                'AGE_HEAD': [r'AGE OF HEAD', r'AGE.*HEAD'],
                'SEX_HEAD': [r'SEX OF HEAD', r'SEX.*HEAD'],
                'RACE': [r'\bRACE\b'],
                'EDUCATION_HEAD': [r'EDUCATION.*HEAD', r'HEAD.*EDUCATION', r'GRADES?\s+COMPLT', r'YRS\s+SCHOOL'],
                'MARITAL_STATUS': [r'MARITAL STATUS'],
                'LABOR_INCOME_HEAD': [r'LABOR\s*(INC|INCOME).*H(EA)?D', r'HDS?\s*LABOR', r'HEAD.*LABOR\s*INCOME'],
                'WAGES_HEAD': [r'HDS?\s*WAGES', r'WAGES.*HEAD', r'HEAD.*WAGES'],
                'HOURLY_EARNINGS_HEAD': [r'HEAD.*HOURLY', r'HOURLY.*EARN.*H', r'H(EA)?D.*HRLY', r'HRLY.*EARN'],
                'ANNUAL_HOURS_HEAD': [r'HEAD.*H(OU)?RS\s*WORK', r'H(EA)?D.*ANNUAL.*HRS', r'YRLY.*H(EA)?DS?\s*HRS', r'ANN.*WORK.*HRS.*H', r'HRS\s*HEAD\s*WORK', r'TOTAL.*LABOR.*HRS.*H', r'HEAD.*TOTAL.*HRS'],
                'SELF_EMPLOYED': [r'SELF.EMPL', r'SELF\s+EMPL'],
                'OCCUPATION_HEAD': [r'OCCUPATION.*HEAD', r'HEAD.*OCC', r'OCC\s*(CODE|OF).*HEAD'],
                'INDUSTRY_HEAD': [r'INDUSTRY.*HEAD', r'HEAD.*IND(USTRY)?'],
                'UNION': [r'UNION\b'],
                'STATE': [r'\bSTATE\b.*NOW', r'STATE\s*\(\d{2}\)', r'\bSTATE\s*$'],
                'REGION': [r'\bREGION\b.*NOW', r'CURRENT.*REGION'],
                'SMSA': [r'SMSA', r'METRO'],
                'DISABILITY': [r'DISAB'],
                'TENURE_POSITION': [r'POSIT', r'TENURE', r'YRS.*EMPL', r'LENGTH.*JOB', r'MOS?\s*(THIS)?\s*POSIT', r'MOS?\s*WKD\s*EMPL', r'YRS?\s*WITH\s*EMPL'],
                'SAME_EMPLOYER': [r'SAME\s*EMPL', r'RET.*SAME.*EMPL', r'SAME\s*JOB', r'SAME\s*COMP'],
                'EMPLOYER_CHANGE': [r'CHANG.*EMPL', r'NEW\s*JOB', r'START.*JOB', r'MO\s*START', r'YR\s*START', r'BEGAN.*JOB', r'MO.*BEGAN'],
                'EMPLOYMENT_STATUS': [r'EMPLOYMENT\s*STATUS', r'EMPL.*STATUS'],
                'GOVT_EMPLOYEE': [r'GOVT', r'GOVERNMENT', r'FED.*GOVT', r'FED.*EMP'],
                'SAMPLE_SRC_SEO': [r'\bSRC\b', r'\bSEO\b', r'SAMPLE\b', r'WHY\s*IN\s*SAMPLE'],
            }

            for cat_name, patterns in categories.items():
                matches = find_matching_vars(labels, patterns)
                if matches:
                    for varname, label in matches:
                        cols = col_specs.get(varname, ('?', '?'))
                        print(f"  {cat_name}: {varname} = \"{label}\" (cols {cols[0]}-{cols[1]})")
        else:
            print(f"WARNING: {do_file} not found")

    # Parse individual file
    ind_do = os.path.join(BASE, "ind2023er", "IND2023ER.do")
    if os.path.exists(ind_do):
        col_specs, labels = parse_do_file(ind_do)
        print(f"\n{'='*80}")
        print(f"INDIVIDUAL FILE: {len(col_specs)} variables, {len(labels)} labels")
        print(f"{'='*80}")

        # Key individual file vars
        ind_vars = {}
        for varname, label in sorted(labels.items()):
            if 'INTERVIEW NUMBER' in label:
                print(f"  INTERVIEW: {varname} = \"{label}\" (cols {col_specs.get(varname, ('?','?'))})")
            elif 'PERSON NUMBER' in label:
                print(f"  PERSON_NUM: {varname} = \"{label}\" (cols {col_specs.get(varname, ('?','?'))})")
            elif 'RELATIONSHIP TO HEAD' in label:
                yr_match = re.search(r'(\d{2})\s*$', label)
                if yr_match:
                    yr = yr_match.group(1)
                    if int(yr) <= 83 or int(yr) >= 68:
                        print(f"  REL_HEAD: {varname} = \"{label}\" (cols {col_specs.get(varname, ('?','?'))})")
            elif 'SEX OF INDIVIDUAL' in label:
                print(f"  SEX: {varname} = \"{label}\" (cols {col_specs.get(varname, ('?','?'))})")
            elif 'INDIVIDUAL WEIGHT' in label:
                yr_match = re.search(r'(\d{2})\s*$', label)
                if yr_match and int(yr_match.group(1)) <= 83:
                    print(f"  WEIGHT: {varname} = \"{label}\" (cols {col_specs.get(varname, ('?','?'))})")

if __name__ == '__main__':
    main()
