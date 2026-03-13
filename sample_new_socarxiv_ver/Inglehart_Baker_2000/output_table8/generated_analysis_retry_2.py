"""
Replication of Table 8 from Inglehart & Baker (2000):
Percentage Saying They "Often" Think About the Meaning and Purpose of Life, by Country and Year

Variable: F001 - "How often do you think about the meaning and purpose of life?"
  1 = Often, 2 = Sometimes, 3 = Rarely, 4 = Never
  Compute: % saying "Often" (value == 1) among valid responses (F001 > 0)

Data sources:
  - WVS Time Series (waves 1, 2, 3) for non-European countries and some European ones
  - EVS 1990 (ZA4460) for European countries in wave 2 (1990-91)

Note: 1981 European data (EVS Wave 1) is not available in our datasets.
The WVS Time Series v5 only includes WVS Wave 1 countries (8 countries).
"""

import pandas as pd
import numpy as np
import math
import os

def run_analysis(wvs_path, evs_long_path):
    """
    Compute Table 8: % saying "Often" think about meaning and purpose of life.
    """
    # ========== LOAD DATA ==========
    wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F001', 'COUNTRY_ALPHA', 'X048WVS'])
    evs_long = pd.read_stata(evs_long_path, convert_categoricals=False)

    def pct_often(f001_series):
        """Compute % saying 'Often' (F001=1) among valid responses."""
        valid = f001_series[f001_series > 0]
        if len(valid) == 0:
            return None
        return (valid == 1).mean() * 100

    def pct_often_evs(q322_series):
        """Compute % for EVS q322 variable."""
        valid = q322_series[q322_series > 0]
        if len(valid) == 0:
            return None
        return (valid == 1).mean() * 100

    def smart_round(val):
        """Round to nearest integer, using standard rounding."""
        if val is None:
            return None
        return round(val)

    # ========== DEFINE COUNTRY GROUPINGS ==========
    advanced = [
        'Australia', 'Belgium', 'Canada', 'Finland', 'France',
        'East Germany', 'West Germany', 'Great Britain', 'Iceland',
        'Ireland', 'Northern Ireland', 'South Korea', 'Italy',
        'Japan', 'Netherlands', 'Norway', 'Spain', 'Sweden',
        'Switzerland', 'United States'
    ]

    ex_communist = [
        'Belarus', 'Bulgaria', 'China', 'Estonia', 'Hungary',
        'Latvia', 'Lithuania', 'Russia', 'Slovenia'
    ]

    developing = [
        'Argentina', 'Brazil', 'Chile', 'India', 'Mexico',
        'Nigeria', 'South Africa', 'Turkey'
    ]

    results = {}

    # ========== WVS WAVE 1 (1981) ==========
    w1 = wvs[(wvs['S002VS'] == 1)]
    w1_countries = {
        'Australia': 'AUS', 'Finland': 'FIN', 'Hungary': 'HUN',
        'Japan': 'JPN', 'South Korea': 'KOR', 'Mexico': 'MEX',
        'Argentina': 'ARG', 'South Africa': 'ZAF'
    }
    for name, code in w1_countries.items():
        val = pct_often(w1[w1['COUNTRY_ALPHA'] == code]['F001'])
        if val is not None:
            results.setdefault(name, {})['1981'] = smart_round(val)

    # ========== EVS 1990 (WAVE 2) FOR EUROPEAN COUNTRIES ==========
    evs_w2_countries = {
        'Belgium': 'BE', 'Canada': 'CA', 'Finland': 'FI', 'France': 'FR',
        'Great Britain': 'GB-GBN', 'Iceland': 'IS', 'Ireland': 'IE',
        'Northern Ireland': 'GB-NIR', 'Italy': 'IT', 'Netherlands': 'NL',
        'Norway': 'NO', 'Spain': 'ES', 'Sweden': 'SE', 'United States': 'US',
        'Hungary': 'HU', 'Bulgaria': 'BG', 'Estonia': 'EE',
        'Latvia': 'LV', 'Lithuania': 'LT', 'Slovenia': 'SI',
    }

    for name, code in evs_w2_countries.items():
        val = pct_often_evs(evs_long[evs_long['c_abrv'] == code]['q322'])
        if val is not None:
            results.setdefault(name, {})['1990-1991'] = smart_round(val)

    # East/West Germany from EVS
    for name, code in [('East Germany', 'DE-E'), ('West Germany', 'DE-W')]:
        val = pct_often_evs(evs_long[evs_long['c_abrv1'] == code]['q322'])
        if val is not None:
            results.setdefault(name, {})['1990-1991'] = smart_round(val)

    # ========== WVS WAVE 2 FOR NON-EUROPEAN COUNTRIES ==========
    w2 = wvs[(wvs['S002VS'] == 2)]
    wvs_w2_countries = {
        'South Korea': 'KOR', 'Japan': 'JPN', 'Switzerland': 'CHE',
        'Argentina': 'ARG', 'Brazil': 'BRA', 'Chile': 'CHL',
        'China': 'CHN', 'India': 'IND', 'Mexico': 'MEX',
        'Nigeria': 'NGA', 'South Africa': 'ZAF', 'Turkey': 'TUR',
        'Belarus': 'BLR', 'Russia': 'RUS',
    }

    for name, code in wvs_w2_countries.items():
        if '1990-1991' not in results.get(name, {}):
            val = pct_often(w2[w2['COUNTRY_ALPHA'] == code]['F001'])
            if val is not None:
                results.setdefault(name, {})['1990-1991'] = smart_round(val)

    # ========== WVS WAVE 3 (1995-1998) ==========
    w3 = wvs[(wvs['S002VS'] == 3)]
    w3_countries = {
        'Australia': 'AUS', 'Finland': 'FIN', 'Japan': 'JPN',
        'Norway': 'NOR', 'Spain': 'ESP', 'Sweden': 'SWE',
        'Switzerland': 'CHE', 'United States': 'USA',
        'Belarus': 'BLR', 'Bulgaria': 'BGR', 'China': 'CHN',
        'Estonia': 'EST', 'Latvia': 'LVA', 'Lithuania': 'LTU',
        'Russia': 'RUS', 'Slovenia': 'SVN',
        'Argentina': 'ARG', 'Brazil': 'BRA', 'Chile': 'CHL',
        'India': 'IND', 'Mexico': 'MEX', 'Nigeria': 'NGA',
        'South Africa': 'ZAF', 'Turkey': 'TUR',
    }

    for name, code in w3_countries.items():
        val = pct_often(w3[w3['COUNTRY_ALPHA'] == code]['F001'])
        if val is not None:
            results.setdefault(name, {})['1995-1998'] = smart_round(val)

    # East/West Germany Wave 3 using X048WVS region codes
    # West German Laender: 01-10 (Schleswig-Holstein to Saarland)
    # East German Laender: 12-16 (Brandenburg, MV, Sachsen, Sachsen-Anhalt, Thueringen)
    deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
    west_codes = [276001, 276002, 276003, 276004, 276005, 276006, 276007, 276008, 276009, 276010]
    east_codes = [276012, 276013, 276014, 276015, 276016]

    val_w = pct_often(deu_w3[deu_w3['X048WVS'].isin(west_codes)]['F001'])
    val_e = pct_often(deu_w3[deu_w3['X048WVS'].isin(east_codes)]['F001'])
    if val_w is not None:
        results.setdefault('West Germany', {})['1995-1998'] = smart_round(val_w)
    if val_e is not None:
        results.setdefault('East Germany', {})['1995-1998'] = smart_round(val_e)

    # ========== COMPUTE NET CHANGE ==========
    for name in results:
        vals = results[name]
        periods = sorted([k for k in vals.keys() if k in ['1981', '1990-1991', '1995-1998']])
        if len(periods) >= 2:
            earliest = vals[periods[0]]
            latest = vals[periods[-1]]
            if earliest is not None and latest is not None:
                vals['net_change'] = latest - earliest

    # ========== FORMAT OUTPUT ==========
    output_lines = []
    output_lines.append("Table 8: Percentage Saying They \"Often\" Think About the Meaning and Purpose of Life, by Country and Year")
    output_lines.append("=" * 100)
    output_lines.append("")

    def format_val(val):
        if val is None:
            return "---"
        return str(int(val))

    def format_net(val):
        if val is None:
            return "---"
        if val > 0:
            return f"+{int(val)}"
        elif val == 0:
            return "0"
        else:
            return str(int(val))

    header = f"{'Country':<25} {'1981':>8} {'1990-1991':>12} {'1995-1998':>12} {'Net Change':>12}"
    separator = "-" * 75

    def print_group(group_name, countries, group_results):
        lines = []
        lines.append(f"\n{group_name}")
        lines.append(separator)
        lines.append(header)
        lines.append(separator)

        increased = 0
        total_with_change = 0
        changes = []

        for c in countries:
            vals = group_results.get(c, {})
            v1981 = vals.get('1981')
            v1990 = vals.get('1990-1991')
            v1995 = vals.get('1995-1998')
            nc = vals.get('net_change')

            lines.append(f"{c:<25} {format_val(v1981):>8} {format_val(v1990):>12} {format_val(v1995):>12} {format_net(nc):>12}")

            if nc is not None:
                total_with_change += 1
                if nc > 0:
                    increased += 1
                changes.append(nc)

        lines.append(separator)
        if changes:
            mean_change = sum(changes) / len(changes)
            lines.append(f"{increased} of {total_with_change} increased; mean change = {mean_change:+.0f}.")
        lines.append("")
        return lines

    output_lines.extend(print_group("ADVANCED INDUSTRIAL DEMOCRACIES", advanced, results))
    output_lines.extend(print_group("EX-COMMUNIST SOCIETIES", ex_communist, results))
    output_lines.extend(print_group("DEVELOPING AND LOW-INCOME SOCIETIES", developing, results))

    output_lines.append("\nNotes:")
    output_lines.append("--- indicates data not available for that time period.")
    output_lines.append("Net Change = latest available value - earliest available value.")
    output_lines.append("1981 values for European countries require EVS Wave 1 data (not available in current dataset).")

    result_text = "\n".join(output_lines)
    return result_text, results


def score_against_ground_truth(results):
    """Score the replication against paper values."""
    ground_truth = {
        'Australia':        {'1981': 34, '1995-1998': 44, 'net_change': 10},
        'Belgium':          {'1981': 22, '1990-1991': 29, 'net_change': 7},
        'Canada':           {'1981': 38, '1990-1991': 43, 'net_change': 5},
        'Finland':          {'1981': 32, '1990-1991': 38, '1995-1998': 40, 'net_change': 8},
        'France':           {'1981': 36, '1990-1991': 39, 'net_change': 3},
        'East Germany':     {'1990-1991': 40, '1995-1998': 47, 'net_change': 7},
        'West Germany':     {'1981': 29, '1990-1991': 30, '1995-1998': 41, 'net_change': 12},
        'Great Britain':    {'1981': 34, '1990-1991': 36, 'net_change': 2},
        'Iceland':          {'1981': 39, '1990-1991': 36, 'net_change': -3},
        'Ireland':          {'1981': 25, '1990-1991': 34, 'net_change': 9},
        'Northern Ireland': {'1981': 29, '1990-1991': 33, 'net_change': 4},
        'South Korea':      {'1981': 29, '1990-1991': 39, 'net_change': 10},
        'Italy':            {'1981': 36, '1990-1991': 48, 'net_change': 12},
        'Japan':            {'1981': 21, '1990-1991': 21, '1995-1998': 26, 'net_change': 5},
        'Netherlands':      {'1981': 21, '1990-1991': 31, 'net_change': 10},
        'Norway':           {'1981': 26, '1990-1991': 31, '1995-1998': 32, 'net_change': 6},
        'Spain':            {'1981': 24, '1990-1991': 27, '1995-1998': 24, 'net_change': 0},
        'Sweden':           {'1981': 20, '1990-1991': 24, '1995-1998': 28, 'net_change': 8},
        'Switzerland':      {'1990-1991': 44, '1995-1998': 43, 'net_change': -1},
        'United States':    {'1981': 48, '1990-1991': 48, '1995-1998': 46, 'net_change': -2},
        'Belarus':          {'1990-1991': 35, '1995-1998': 47, 'net_change': 12},
        'Bulgaria':         {'1990-1991': 44, '1995-1998': 33, 'net_change': -11},
        'China':            {'1990-1991': 30, '1995-1998': 26, 'net_change': -4},
        'Estonia':          {'1990-1991': 35, '1995-1998': 39, 'net_change': 4},
        'Hungary':          {'1981': 44, '1990-1991': 45, 'net_change': 1},
        'Latvia':           {'1990-1991': 36, '1995-1998': 43, 'net_change': 7},
        'Lithuania':        {'1990-1991': 41, '1995-1998': 42, 'net_change': 1},
        'Russia':           {'1990-1991': 41, '1995-1998': 45, 'net_change': 4},
        'Slovenia':         {'1990-1991': 37, '1995-1998': 33, 'net_change': -4},
        'Argentina':        {'1981': 30, '1990-1991': 57, '1995-1998': 51, 'net_change': 21},
        'Brazil':           {'1990-1991': 44, '1995-1998': 37, 'net_change': -7},
        'Chile':            {'1990-1991': 54, '1995-1998': 50, 'net_change': -4},
        'India':            {'1990-1991': 28, '1995-1998': 23, 'net_change': -5},
        'Mexico':           {'1981': 32, '1990-1991': 40, '1995-1998': 39, 'net_change': 7},
        'Nigeria':          {'1990-1991': 60, '1995-1998': 50, 'net_change': -10},
        'South Africa':     {'1981': 38, '1990-1991': 57, '1995-1998': 46, 'net_change': 8},
        'Turkey':           {'1990-1991': 38, '1995-1998': 50, 'net_change': 12},
    }

    total_values = 0
    matched_values = 0
    partial_values = 0
    missing_values = 0
    miss_values = 0
    details = []

    for country, gt in ground_truth.items():
        rep = results.get(country, {})
        for period in ['1981', '1990-1991', '1995-1998', 'net_change']:
            if period in gt:
                total_values += 1
                gt_val = gt[period]
                rep_val = rep.get(period)

                if rep_val is None:
                    missing_values += 1
                    details.append(f"  MISSING: {country} {period}: paper={gt_val}, replicated=N/A")
                elif abs(rep_val - gt_val) == 0:
                    matched_values += 1
                elif abs(rep_val - gt_val) <= 2:
                    partial_values += 1
                    details.append(f"  PARTIAL: {country} {period}: paper={gt_val}, replicated={rep_val}, diff={rep_val-gt_val:+d}")
                else:
                    miss_values += 1
                    details.append(f"  MISS: {country} {period}: paper={gt_val}, replicated={rep_val}, diff={rep_val-gt_val:+d}")

    # Scoring: Categories present (20%), Values (40%), Ordering (10%), N/summary (20%), Structure (10%)
    # Adapt for this frequency table:
    # - Categories present: all 37 countries present = 20 pts
    # - Values accuracy: exact match (within 0), partial (within 2), miss, missing
    # - Ordering correct by group
    # - Summary stats (increased counts, mean change)

    n_computable = total_values - missing_values  # values we can actually compute
    if n_computable > 0:
        value_accuracy = (matched_values * 1.0 + partial_values * 0.7 + miss_values * 0.2) / total_values
    else:
        value_accuracy = 0

    score = value_accuracy * 100

    print(f"\n{'='*60}")
    print(f"SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"Total values to check: {total_values}")
    print(f"  Full match (exact): {matched_values}")
    print(f"  Partial match (within 2): {partial_values}")
    print(f"  Miss (off by >2): {miss_values}")
    print(f"  Missing (no data): {missing_values}")
    print(f"Score: {score:.1f}/100")
    print(f"\nDetails:")
    for d in details:
        print(d)

    return score


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)

    wvs_path = os.path.join(project_dir, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
    evs_long_path = os.path.join(project_dir, "data", "ZA4460_v3-0-0.dta")

    result_text, results = run_analysis(wvs_path, evs_long_path)
    print(result_text)
    score = score_against_ground_truth(results)
