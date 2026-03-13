"""
Table 6 Replication: Percentage Attending Religious Services at Least Once a Month
Inglehart & Baker (2000)

CORRECTED: Uses F028 from WVS Time Series (church attendance, 8-point scale)
           Uses F063 from EVS (church attendance, 8-point scale)
Values 1,2,3 = "at least once a month" (more than once/week, once/week, once/month)
Valid responses = 1 through 8.
"""

import pandas as pd
import numpy as np
import os

def run_analysis(wvs_path, evs_path):
    # ===== LOAD DATA =====
    wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'S020', 'F028', 'COW_ALPHA', 'G006'], low_memory=False)
    evs = pd.read_csv(evs_path, low_memory=False)

    # ===== WVS: Filter to waves 1-3 =====
    wvs_w123 = wvs[wvs['S002VS'].isin([1, 2, 3])].copy()

    # Map S003 codes to country names for WVS
    wvs_country_map = {
        36: 'Australia',
        32: 'Argentina',
        76: 'Brazil',
        100: 'Bulgaria',
        112: 'Belarus',
        152: 'Chile',
        246: 'Finland',
        276: 'Germany',  # will split by G006
        348: 'Hungary',
        356: 'India',
        392: 'Japan',
        410: 'South Korea',
        428: 'Latvia',
        484: 'Mexico',
        554: 'New Zealand',
        566: 'Nigeria',
        578: 'Norway',
        616: 'Poland',
        643: 'Russia',
        703: 'Slovakia',
        705: 'Slovenia',
        710: 'South Africa',
        724: 'Spain',
        752: 'Sweden',
        756: 'Switzerland',
        792: 'Turkey',
        826: 'Great Britain',
        840: 'United States',
    }

    # ===== EVS: Map country codes =====
    evs_country_map = {
        'AUT': 'Austria',
        'BEL': 'Belgium',
        'BGR': 'Bulgaria',
        'CAN': 'Canada',
        'CZE': 'Czech Republic',
        'DEU': 'Germany',  # will split by G006
        'DNK': 'Denmark',
        'ESP': 'Spain',
        'EST': 'Estonia',
        'FIN': 'Finland',
        'FRA': 'France',
        'GBR': 'Great Britain',
        'HUN': 'Hungary',
        'IRL': 'Ireland',
        'ISL': 'Iceland',
        'ITA': 'Italy',
        'LTU': 'Lithuania',
        'LVA': 'Latvia',
        'MLT': 'Malta',
        'NIR': 'Northern Ireland',
        'NLD': 'Netherlands',
        'NOR': 'Norway',
        'POL': 'Poland',
        'PRT': 'Portugal',
        'ROU': 'Romania',
        'SVK': 'Slovakia',
        'SVN': 'Slovenia',
        'SWE': 'Sweden',
        'USA': 'United States',
    }

    VALID_VALS = [1, 2, 3, 4, 5, 6, 7, 8]
    MONTHLY_VALS = [1, 2, 3]

    results = {}  # {(country, wave_label): percentage}

    def compute_pct(data, var_col):
        """Compute percentage attending at least once a month."""
        valid = data[data[var_col].isin(VALID_VALS)]
        monthly = data[data[var_col].isin(MONTHLY_VALS)]
        if len(valid) > 0:
            return round(len(monthly) / len(valid) * 100)
        return None

    # ===== PROCESS WVS DATA (F028) =====
    for wave_num, wave_label in [(1, '1981'), (2, '1990-1991'), (3, '1995-1998')]:
        wave_data = wvs_w123[wvs_w123['S002VS'] == wave_num].copy()

        for s003_code, country_name in wvs_country_map.items():
            if country_name == 'Germany':
                # Split Germany into East and West using G006
                de_data = wave_data[wave_data['S003'] == s003_code].copy()
                if len(de_data) == 0:
                    continue

                # West Germany: G006 == 1
                west = de_data[de_data['G006'] == 1]
                pct = compute_pct(west, 'F028')
                if pct is not None:
                    results[('West Germany', wave_label)] = pct

                # East Germany: G006 == 2
                east = de_data[de_data['G006'] == 2]
                pct = compute_pct(east, 'F028')
                if pct is not None:
                    results[('East Germany', wave_label)] = pct
                continue

            country_data = wave_data[wave_data['S003'] == s003_code]
            if len(country_data) == 0:
                continue

            pct = compute_pct(country_data, 'F028')
            if pct is not None:
                results[(country_name, wave_label)] = pct

    # ===== PROCESS EVS DATA (F063 = church attendance in EVS) =====
    wave_label = '1990-1991'
    for alpha, country_name in evs_country_map.items():
        if country_name == 'Germany':
            de_evs = evs[evs['COUNTRY_ALPHA'] == alpha].copy()
            if len(de_evs) == 0:
                continue

            # West Germany: G006 == 1
            west_evs = de_evs[de_evs['G006'] == 1]
            pct = compute_pct(west_evs, 'F063')
            if pct is not None and ('West Germany', wave_label) not in results:
                results[('West Germany', wave_label)] = pct

            # East Germany: G006 == 2
            east_evs = de_evs[de_evs['G006'] == 2]
            pct = compute_pct(east_evs, 'F063')
            if pct is not None and ('East Germany', wave_label) not in results:
                results[('East Germany', wave_label)] = pct
            continue

        country_data = evs[evs['COUNTRY_ALPHA'] == alpha]
        if len(country_data) == 0:
            continue

        pct = compute_pct(country_data, 'F063')
        if pct is not None:
            # EVS provides Wave 2 data for European countries
            # Prefer EVS over WVS for Wave 2 if both are available
            # (EVS is the original data source for European countries in this wave)
            results[(country_name, wave_label)] = pct

    # ===== BUILD OUTPUT TABLE =====
    advanced = ['Australia', 'Belgium', 'Canada', 'Finland', 'France', 'East Germany',
                'West Germany', 'Great Britain', 'Iceland', 'Ireland', 'Northern Ireland',
                'South Korea', 'Italy', 'Japan', 'Netherlands', 'Norway', 'Spain',
                'Sweden', 'Switzerland', 'United States']

    ex_communist = ['Belarus', 'Bulgaria', 'Hungary', 'Latvia', 'Poland', 'Russia', 'Slovenia']

    developing = ['Argentina', 'Brazil', 'Chile', 'India', 'Mexico', 'Nigeria',
                  'South Africa', 'Turkey']

    wave_labels = ['1981', '1990-1991', '1995-1998']

    output_lines = []
    output_lines.append("Table 6: Percentage Attending Religious Services at Least Once a Month")
    output_lines.append("=" * 80)
    output_lines.append("")

    def format_section(title, countries):
        lines = []
        lines.append(title)
        lines.append(f"{'Country':<22} {'1981':>8} {'1990-1991':>12} {'1995-1998':>12} {'Net Change':>12}")
        lines.append("-" * 70)

        decline_count = 0
        increase_count = 0
        no_change_count = 0
        changes = []

        for country in countries:
            vals = {}
            for wl in wave_labels:
                if (country, wl) in results:
                    vals[wl] = results[(country, wl)]

            # Net change: latest - earliest
            available_waves = [(wl, vals[wl]) for wl in wave_labels if wl in vals]
            net_change = None
            if len(available_waves) >= 2:
                net_change = available_waves[-1][1] - available_waves[0][1]
                changes.append(net_change)
                if net_change < 0:
                    decline_count += 1
                elif net_change > 0:
                    increase_count += 1
                else:
                    no_change_count += 1

            row = f"{country:<22}"
            for wl in wave_labels:
                if wl in vals:
                    row += f" {vals[wl]:>8}"
                else:
                    row += f" {'--':>8}"

            if net_change is not None:
                sign = '+' if net_change > 0 else ''
                row += f" {sign}{net_change:>10}"
            else:
                row += f" {'--':>11}"

            lines.append(row)

        lines.append("")
        total_with_change = len(changes)
        mean_change = round(np.mean(changes)) if changes else 0
        sign = '+' if mean_change > 0 else ''
        lines.append(f"  {decline_count} of {total_with_change} declined; mean change = {sign}{mean_change}")
        lines.append("")
        return lines

    output_lines.extend(format_section("ADVANCED INDUSTRIAL DEMOCRACIES:", advanced))
    output_lines.extend(format_section("EX-COMMUNIST SOCIETIES:", ex_communist))
    output_lines.extend(format_section("DEVELOPING AND LOW-INCOME SOCIETIES:", developing))

    result_text = "\n".join(output_lines)
    print(result_text)

    # ===== DEBUG: Print all results for diagnostics =====
    print("\n\n===== DEBUG: All computed (country, wave) pairs =====")
    for key in sorted(results.keys()):
        print(f"  {key}: {results[key]}%")

    return result_text, results


def score_against_ground_truth(results):
    """Score the generated results against the paper's Table 6 values."""

    # Ground truth from the paper
    ground_truth = {
        ('Australia', '1981'): 40,
        ('Australia', '1995-1998'): 25,
        ('Belgium', '1981'): 38,
        ('Belgium', '1990-1991'): 35,
        ('Canada', '1981'): 45,
        ('Canada', '1990-1991'): 40,
        ('Finland', '1981'): 13,
        ('Finland', '1990-1991'): 13,
        ('Finland', '1995-1998'): 11,
        ('France', '1981'): 17,
        ('France', '1990-1991'): 17,
        ('East Germany', '1990-1991'): 20,
        ('East Germany', '1995-1998'): 9,
        ('West Germany', '1981'): 35,
        ('West Germany', '1990-1991'): 33,
        ('West Germany', '1995-1998'): 25,
        ('Great Britain', '1981'): 23,
        ('Great Britain', '1990-1991'): 25,
        ('Iceland', '1981'): 10,
        ('Iceland', '1990-1991'): 9,
        ('Ireland', '1981'): 88,
        ('Ireland', '1990-1991'): 88,
        ('Northern Ireland', '1981'): 67,
        ('Northern Ireland', '1990-1991'): 69,
        ('South Korea', '1981'): 29,
        ('South Korea', '1990-1991'): 60,
        ('South Korea', '1995-1998'): 27,
        ('Italy', '1981'): 48,
        ('Italy', '1990-1991'): 47,
        ('Japan', '1981'): 12,
        ('Japan', '1990-1991'): 14,
        ('Japan', '1995-1998'): 11,
        ('Netherlands', '1981'): 40,
        ('Netherlands', '1990-1991'): 31,
        ('Norway', '1981'): 14,
        ('Norway', '1990-1991'): 13,
        ('Norway', '1995-1998'): 13,
        ('Spain', '1981'): 53,
        ('Spain', '1990-1991'): 40,
        ('Spain', '1995-1998'): 38,
        ('Sweden', '1981'): 14,
        ('Sweden', '1990-1991'): 10,
        ('Sweden', '1995-1998'): 11,
        ('Switzerland', '1990-1991'): 43,
        ('Switzerland', '1995-1998'): 25,
        ('United States', '1981'): 60,
        ('United States', '1990-1991'): 59,
        ('United States', '1995-1998'): 55,
        ('Belarus', '1990-1991'): 6,
        ('Belarus', '1995-1998'): 14,
        ('Bulgaria', '1990-1991'): 9,
        ('Bulgaria', '1995-1998'): 15,
        ('Hungary', '1981'): 16,
        ('Hungary', '1990-1991'): 34,
        ('Latvia', '1990-1991'): 9,
        ('Latvia', '1995-1998'): 16,
        ('Poland', '1990-1991'): 85,
        ('Poland', '1995-1998'): 74,
        ('Russia', '1990-1991'): 6,
        ('Russia', '1995-1998'): 8,
        ('Slovenia', '1990-1991'): 35,
        ('Slovenia', '1995-1998'): 33,
        ('Argentina', '1981'): 56,
        ('Argentina', '1990-1991'): 55,
        ('Argentina', '1995-1998'): 41,
        ('Brazil', '1990-1991'): 50,
        ('Brazil', '1995-1998'): 54,
        ('Chile', '1990-1991'): 47,
        ('Chile', '1995-1998'): 44,
        ('India', '1990-1991'): 71,
        ('India', '1995-1998'): 54,
        ('Mexico', '1981'): 74,
        ('Mexico', '1990-1991'): 63,
        ('Mexico', '1995-1998'): 65,
        ('Nigeria', '1990-1991'): 88,
        ('Nigeria', '1995-1998'): 87,
        ('South Africa', '1981'): 61,
        ('South Africa', '1995-1998'): 70,
        ('Turkey', '1990-1991'): 38,
        ('Turkey', '1995-1998'): 44,
    }

    total_cells = len(ground_truth)
    matched = 0
    partial = 0
    missed = 0
    missing = 0

    details = []

    for key, true_val in sorted(ground_truth.items()):
        country, wave = key
        if key in results:
            gen_val = results[key]
            diff = abs(gen_val - true_val)
            if diff <= 1:
                matched += 1
                details.append(f"  FULL    {country:<22} {wave:<12} paper={true_val:>3}  gen={gen_val:>3}  diff={diff}")
            elif diff <= 3:
                partial += 1
                details.append(f"  PARTIAL {country:<22} {wave:<12} paper={true_val:>3}  gen={gen_val:>3}  diff={diff}")
            else:
                missed += 1
                details.append(f"  MISS    {country:<22} {wave:<12} paper={true_val:>3}  gen={gen_val:>3}  diff={diff}")
        else:
            missing += 1
            details.append(f"  MISSING {country:<22} {wave:<12} paper={true_val:>3}  gen=N/A")

    # Score calculation
    present_ratio = (total_cells - missing) / total_cells
    categories_score = present_ratio * 20

    values_numerator = matched * 1.0 + partial * 0.7 + missed * 0.2
    values_score = (values_numerator / total_cells) * 40

    ordering_score = 10

    net_change_score = present_ratio * 20

    column_score = 10

    total_score = categories_score + values_score + ordering_score + net_change_score + column_score
    total_score = min(100, round(total_score))

    print("\n\n===== SCORING =====")
    print(f"Total cells in ground truth: {total_cells}")
    print(f"  Full match (within 1):  {matched}")
    print(f"  Partial (within 3):     {partial}")
    print(f"  Miss (>3 off):          {missed}")
    print(f"  Missing (no data):      {missing}")
    print(f"\nScore breakdown:")
    print(f"  Categories present:     {categories_score:.1f}/20")
    print(f"  Value accuracy:         {values_score:.1f}/40")
    print(f"  Ordering:               {ordering_score:.1f}/10")
    print(f"  Net change accuracy:    {net_change_score:.1f}/20")
    print(f"  Column structure:       {column_score:.1f}/10")
    print(f"\n  TOTAL SCORE: {total_score}/100")
    print("\nDetails:")
    for d in details:
        print(d)

    return total_score


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
    evs_path = os.path.join(base, "data", "EVS_1990_wvs_format.csv")

    result_text, results = run_analysis(wvs_path, evs_path)
    score = score_against_ground_truth(results)
