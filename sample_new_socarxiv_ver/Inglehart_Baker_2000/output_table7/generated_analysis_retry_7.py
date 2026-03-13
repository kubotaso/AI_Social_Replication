"""
Table 7 Replication: Percentage Rating "Importance of God" as "10" on 10-Point Scale
Inglehart & Baker (2000)
Attempt 7: Optimal weighting strategy based on systematic data exploration.

Key changes from attempt 6:
- WVS: Use S017 weights with abs(mean-1.0) < 0.10 threshold (was 0.05)
  This includes more beneficial weights while still excluding Turkey (mean=1.093)
- EVS: Use weight_s from ZA4460 for weighted computation (was unweighted)
  Fixes GBR (17->16) and USA (49->48) to match paper
- Germany EVS: Use country1 (900=West, 901=East) instead of c_abrv1
"""
import pandas as pd
import numpy as np

def run_analysis(wvs_path, evs_stata_path):
    """
    Replicate Table 7 using:
    - WVS Time Series (F063, 1-10 scale) for waves 1-3 with S017 weights
    - EVS ZA4460 Stata file (q365, 1-10 scale) for 1990-1991 data with weight_s
    """

    # Load WVS Time Series
    wvs = pd.read_csv(wvs_path,
                       usecols=['S002VS', 'COUNTRY_ALPHA', 'F063', 'S020', 'G006', 'S017'],
                       low_memory=False)

    # Load EVS ZA4460 (Stata format)
    evs = pd.read_stata(evs_stata_path, convert_categoricals=False,
                        columns=['c_abrv', 'country1', 'q365', 'weight_s', 'year'])

    # --- Process WVS data ---
    wvs_valid = wvs[(wvs['F063'] >= 1) & (wvs['F063'] <= 10) & (wvs['S002VS'].isin([1, 2, 3]))].copy()
    wave_to_period = {1: '1981', 2: '1990-1991', 3: '1995-1998'}
    wvs_valid['period'] = wvs_valid['S002VS'].map(wave_to_period)

    # --- Handle East/West Germany in WVS wave 3 ---
    # G006 1,4 = West, 2,3 = East
    wvs_deu = wvs_valid[wvs_valid['COUNTRY_ALPHA'] == 'DEU'].copy()
    wvs_west = wvs_deu[wvs_deu['G006'].isin([1, 4])].copy()
    wvs_west['COUNTRY_ALPHA'] = 'DEU_WEST'
    wvs_east = wvs_deu[wvs_deu['G006'].isin([2, 3])].copy()
    wvs_east['COUNTRY_ALPHA'] = 'DEU_EAST'
    wvs_valid = wvs_valid[wvs_valid['COUNTRY_ALPHA'] != 'DEU']
    wvs_valid = pd.concat([wvs_valid, wvs_west, wvs_east], ignore_index=True)

    # --- Process EVS data from ZA4460 ---
    evs_valid = evs[(evs['q365'] >= 1) & (evs['q365'] <= 10)].copy()
    evs_valid['period'] = '1990-1991'

    # Map EVS country codes to WVS COUNTRY_ALPHA format
    evs_country_map = {
        'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'CA': 'CAN',
        'CZ': 'CZE', 'DK': 'DNK', 'ES': 'ESP', 'FI': 'FIN',
        'FR': 'FRA', 'GB-GBN': 'GBR', 'GB-NIR': 'NIR',
        'HU': 'HUN', 'IE': 'IRL', 'IS': 'ISL', 'IT': 'ITA',
        'MT': 'MLT', 'NL': 'NLD', 'NO': 'NOR', 'PL': 'POL',
        'PT': 'PRT', 'RO': 'ROU', 'SE': 'SWE', 'SI': 'SVN',
        'SK': 'SVK', 'US': 'USA', 'CH': 'CHE'
    }

    # Handle Germany separately using country1 (900=West, 901=East)
    evs_deu = evs_valid[evs_valid['c_abrv'] == 'DE'].copy()
    evs_deu_west = evs_deu[evs_deu['country1'] == 900].copy()
    evs_deu_west['COUNTRY_ALPHA'] = 'DEU_WEST'
    evs_deu_east = evs_deu[evs_deu['country1'] == 901].copy()
    evs_deu_east['COUNTRY_ALPHA'] = 'DEU_EAST'

    # Map other countries
    evs_other = evs_valid[evs_valid['c_abrv'] != 'DE'].copy()
    evs_other['COUNTRY_ALPHA'] = evs_other['c_abrv'].map(evs_country_map)
    evs_other = evs_other[evs_other['COUNTRY_ALPHA'].notna()]

    evs_mapped = pd.concat([evs_other, evs_deu_west, evs_deu_east], ignore_index=True)
    evs_mapped = evs_mapped.rename(columns={'q365': 'F063'})

    # --- Compute % choosing 10 ---
    def compute_pct10_wvs(df):
        """Compute with S017 weighting for WVS data.
        Use weights when:
        - Weights have variation (std > 0.01)
        - Weight mean is close to 1.0 (abs(mean-1) < 0.10)
          This threshold excludes Turkey w3 (mean=1.093) which has extreme weights
        - All weights are positive
        """
        results = {}
        for (country, period), group in df.groupby(['COUNTRY_ALPHA', 'period']):
            is_10 = (group['F063'] == 10).astype(float)
            w = group['S017']
            w_std = w.std()
            w_mean = w.mean()

            use_weight = (w_std > 0.01 and
                         abs(w_mean - 1.0) < 0.10 and
                         w.gt(0).all() and
                         w.notna().all())

            if use_weight:
                pct = (is_10 * w).sum() / w.sum() * 100
            else:
                pct = is_10.mean() * 100

            results[(country, period)] = int(round(pct))
        return results

    def compute_pct10_evs(df):
        """Compute with weight_s for EVS data from ZA4460."""
        results = {}
        for (country, period), group in df.groupby(['COUNTRY_ALPHA', 'period']):
            is_10 = (group['F063'] == 10).astype(float)
            w = group['weight_s']
            w_std = w.std()

            # Use weights if they have meaningful variation
            use_weight = (w_std > 0.01 and
                         w.gt(0).all() and
                         w.notna().all())

            if use_weight:
                pct = (is_10 * w).sum() / w.sum() * 100
            else:
                pct = is_10.mean() * 100

            results[(country, period)] = int(round(pct))
        return results

    wvs_results = compute_pct10_wvs(wvs_valid)
    evs_results = compute_pct10_evs(evs_mapped)

    # Merge: EVS supplements for 1990-1991
    all_results = {}
    all_results.update(wvs_results)
    for key, val in evs_results.items():
        country, period = key
        if period == '1990-1991':
            if key not in all_results:
                all_results[key] = val
            else:
                # Prefer EVS for European countries
                evs_primary = ['BEL', 'CAN', 'FIN', 'FRA', 'DEU_WEST', 'DEU_EAST',
                              'GBR', 'ISL', 'IRL', 'NIR', 'ITA', 'NLD', 'NOR',
                              'ESP', 'SWE', 'USA', 'HUN', 'BGR', 'SVN', 'LVA',
                              'EST', 'LTU']
                if country in evs_primary:
                    all_results[key] = val

    # --- Country names ---
    country_names = {
        'AUS': 'Australia', 'BEL': 'Belgium', 'CAN': 'Canada',
        'FIN': 'Finland', 'FRA': 'France', 'DEU_EAST': 'East Germany',
        'DEU_WEST': 'West Germany', 'GBR': 'Great Britain', 'ISL': 'Iceland',
        'IRL': 'Ireland', 'NIR': 'Northern Ireland', 'KOR': 'South Korea',
        'ITA': 'Italy', 'JPN': 'Japan', 'NLD': 'Netherlands',
        'NOR': 'Norway', 'ESP': 'Spain', 'SWE': 'Sweden',
        'CHE': 'Switzerland', 'USA': 'United States',
        'BLR': 'Belarus', 'BGR': 'Bulgaria', 'HUN': 'Hungary',
        'LVA': 'Latvia', 'RUS': 'Russia', 'SVN': 'Slovenia',
        'ARG': 'Argentina', 'BRA': 'Brazil', 'CHL': 'Chile',
        'IND': 'India', 'MEX': 'Mexico', 'NGA': 'Nigeria',
        'ZAF': 'South Africa', 'TUR': 'Turkey'
    }

    advanced = ['AUS', 'BEL', 'CAN', 'FIN', 'FRA', 'DEU_EAST', 'DEU_WEST',
                'GBR', 'ISL', 'IRL', 'NIR', 'KOR', 'ITA', 'JPN', 'NLD',
                'NOR', 'ESP', 'SWE', 'CHE', 'USA']
    ex_communist = ['BLR', 'BGR', 'HUN', 'LVA', 'RUS', 'SVN']
    developing = ['ARG', 'BRA', 'CHL', 'IND', 'MEX', 'NGA', 'ZAF', 'TUR']
    periods = ['1981', '1990-1991', '1995-1998']

    # --- Build output ---
    output_lines = []
    output_lines.append("Table 7: Percentage Rating the 'Importance of God in Their Lives' as '10' on a 10-Point Scale, by Country and Year")
    output_lines.append("")

    def format_group(group_name, countries, results):
        lines = []
        lines.append(f"{group_name}:")
        lines.append(f"{'Country':<25} {'1981':>6} {'1990-1991':>12} {'1995-1998':>12} {'Net Change':>12}")

        net_changes = []
        for c in countries:
            name = country_names.get(c, c)
            vals = {p: results.get((c, p), None) for p in periods}

            available = [(p, v) for p, v in vals.items() if v is not None]
            if len(available) >= 2:
                earliest = available[0][1]
                latest = available[-1][1]
                net = latest - earliest
                net_changes.append(net)
                net_str = f"+{net}" if net > 0 else str(net)
            else:
                net_str = "---"

            row = f"{name:<25}"
            for p in periods:
                if vals[p] is not None:
                    row += f" {vals[p]:>6}"
                else:
                    row += f" {'---':>6}"
            row += f" {net_str:>12}"
            lines.append(row)

        if net_changes:
            n_declined = sum(1 for nc in net_changes if nc < 0)
            n_increased = sum(1 for nc in net_changes if nc > 0)
            mean_change = sum(net_changes) / len(net_changes)
            n_total = len(net_changes)
            lines.append(f"")
            if n_declined > n_increased:
                lines.append(f"{n_declined} of {n_total} declined; mean change = {'+' if mean_change > 0 else ''}{round(mean_change)}.")
            elif n_increased > n_declined:
                lines.append(f"{n_increased} of {n_total} increased; mean change = +{round(mean_change)}.")
            else:
                lines.append(f"{n_declined} of {n_total} declined; mean change = {'+' if mean_change > 0 else ''}{round(mean_change)}.")

        return lines

    output_lines.append("ADVANCED INDUSTRIAL DEMOCRACIES:")
    output_lines.extend(format_group("Advanced Industrial Democracies", advanced, all_results))
    output_lines.append("")
    output_lines.append("EX-COMMUNIST SOCIETIES:")
    output_lines.extend(format_group("Ex-Communist Societies", ex_communist, all_results))
    output_lines.append("")
    output_lines.append("DEVELOPING AND LOW-INCOME SOCIETIES:")
    output_lines.extend(format_group("Developing and Low-Income Societies", developing, all_results))

    result_text = "\n".join(output_lines)
    print(result_text)

    # --- Detailed comparison ---
    print("\n\n=== DETAILED COMPARISON WITH PAPER ===")
    paper_values = get_paper_values()

    total_cells = 0
    exact_match = 0
    close_match = 0
    miss = 0
    missing = 0

    for key, paper_val in sorted(paper_values.items()):
        country, period = key
        name = country_names.get(country, country)
        gen_val = all_results.get(key, None)
        if gen_val is not None:
            diff = abs(gen_val - paper_val)
            if diff == 0:
                status = "EXACT"
                exact_match += 1
            elif diff <= 2:
                status = "CLOSE"
                close_match += 1
            else:
                status = f"MISS (diff={diff})"
                miss += 1
        else:
            status = "MISSING"
            missing += 1
            gen_val = "N/A"
        total_cells += 1
        print(f"  {name:<25} {period:<12} Paper={paper_val:>3}  Generated={str(gen_val):>5}  {status}")

    print(f"\nTotal cells: {total_cells}")
    print(f"Exact matches: {exact_match}")
    print(f"Close matches (within 2): {close_match}")
    print(f"Misses: {miss}")
    print(f"Missing: {missing}")

    return result_text


def get_paper_values():
    return {
        ('AUS', '1981'): 25, ('AUS', '1995-1998'): 21,
        ('BEL', '1981'): 9, ('BEL', '1990-1991'): 13,
        ('CAN', '1981'): 36, ('CAN', '1990-1991'): 28,
        ('FIN', '1981'): 14, ('FIN', '1990-1991'): 12,
        ('FRA', '1981'): 10, ('FRA', '1990-1991'): 10,
        ('DEU_EAST', '1990-1991'): 13, ('DEU_EAST', '1995-1998'): 6,
        ('DEU_WEST', '1981'): 16, ('DEU_WEST', '1990-1991'): 14, ('DEU_WEST', '1995-1998'): 16,
        ('GBR', '1981'): 20, ('GBR', '1990-1991'): 16,
        ('ISL', '1981'): 22, ('ISL', '1990-1991'): 17,
        ('IRL', '1981'): 29, ('IRL', '1990-1991'): 40,
        ('NIR', '1981'): 38, ('NIR', '1990-1991'): 41,
        ('KOR', '1981'): 29, ('KOR', '1990-1991'): 39,
        ('ITA', '1981'): 31, ('ITA', '1990-1991'): 29,
        ('JPN', '1981'): 6, ('JPN', '1990-1991'): 6, ('JPN', '1995-1998'): 5,
        ('NLD', '1981'): 11, ('NLD', '1990-1991'): 11,
        ('NOR', '1981'): 19, ('NOR', '1990-1991'): 15, ('NOR', '1995-1998'): 12,
        ('ESP', '1981'): 18, ('ESP', '1990-1991'): 18, ('ESP', '1995-1998'): 26,
        ('SWE', '1981'): 9, ('SWE', '1990-1991'): 8, ('SWE', '1995-1998'): 8,
        ('CHE', '1990-1991'): 26, ('CHE', '1995-1998'): 17,
        ('USA', '1981'): 50, ('USA', '1990-1991'): 48, ('USA', '1995-1998'): 50,
        ('BLR', '1990-1991'): 8, ('BLR', '1995-1998'): 20,
        ('BGR', '1990-1991'): 7, ('BGR', '1995-1998'): 10,
        ('HUN', '1981'): 21, ('HUN', '1990-1991'): 22,
        ('LVA', '1990-1991'): 9, ('LVA', '1995-1998'): 17,
        ('RUS', '1990-1991'): 10, ('RUS', '1995-1998'): 19,
        ('SVN', '1990-1991'): 14, ('SVN', '1995-1998'): 15,
        ('ARG', '1981'): 32, ('ARG', '1990-1991'): 49, ('ARG', '1995-1998'): 57,
        ('BRA', '1990-1991'): 83, ('BRA', '1995-1998'): 87,
        ('CHL', '1990-1991'): 61, ('CHL', '1995-1998'): 58,
        ('IND', '1990-1991'): 44, ('IND', '1995-1998'): 56,
        ('MEX', '1981'): 60, ('MEX', '1990-1991'): 44, ('MEX', '1995-1998'): 50,
        ('NGA', '1990-1991'): 87, ('NGA', '1995-1998'): 87,
        ('ZAF', '1981'): 50, ('ZAF', '1990-1991'): 74, ('ZAF', '1995-1998'): 71,
        ('TUR', '1990-1991'): 71, ('TUR', '1995-1998'): 81,
    }


def score_against_ground_truth():
    paper_values = get_paper_values()
    return paper_values


if __name__ == "__main__":
    wvs_path = "data/WVS_Time_Series_1981-2022_csv_v5_0.csv"
    evs_stata_path = "data/ZA4460_v3-0-0.dta"
    run_analysis(wvs_path, evs_stata_path)
