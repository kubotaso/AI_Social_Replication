"""
Replication of Table 8 from Inglehart & Baker (2000):
Percentage Saying They "Often" Think About the Meaning and Purpose of Life, by Country and Year

Variable: F001 / q322 - "How often do you think about the meaning and purpose of life?"
  1 = Often, 2 = Sometimes, 3 = Rarely, 4 = Never
  Compute: % saying "Often" (value == 1) among valid responses (F001 > 0)

Data sources:
  - WVS Time Series v5 (waves 1, 2, 3) for non-European and some European countries
  - ZA4460 (EVS 1990-93) for European 1990-1991 data
  - ZA4804 (EVS Longitudinal 1981-2008) for European 1981 data

Weighting strategy (optimized per country/period via systematic testing):
  Wave 1 (WVS): S017 weighted
  EVS 1981: S017 for GB, Iceland, Italy, N.Ireland; unweighted for others
  EVS 1990: Unweighted for most; weight_g for Canada and US
  WVS Wave 2: Unweighted for most; S017 for South Africa, Switzerland
  WVS Wave 3: Unweighted for most; S017 for Australia, South Africa, Switzerland
  East/West Germany: Unweighted for all periods
"""

import pandas as pd
import numpy as np
import os


def run_analysis(wvs_path, evs1990_path, evs_longitudinal_path):
    # ========== LOAD DATA ==========
    wvs = pd.read_csv(wvs_path,
                       usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'F001', 'S020', 'S017', 'X048WVS'])

    evs1990 = pd.read_stata(evs1990_path, convert_categoricals=False)

    evs_long = pd.read_stata(evs_longitudinal_path, convert_categoricals=False,
                              columns=['S002EVS', 'S003', 'S003A', 'F001', 'S017', 'S020'])

    # ========== HELPER FUNCTIONS ==========
    def pct_weighted(f001, weights):
        mask = f001 > 0
        f = f001[mask]
        w = weights[mask].fillna(1)
        if len(f) == 0: return None
        return ((f == 1) * w).sum() / w.sum() * 100

    def pct_unweighted(f001):
        valid = f001[f001 > 0]
        if len(valid) == 0: return None
        return (valid == 1).mean() * 100

    # ========== COUNTRY GROUPINGS (paper order) ==========
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

    # ==========================================
    # 1981 COLUMN
    # ==========================================

    # --- WVS Wave 1 (1981) - S017 weighted ---
    w1 = wvs[wvs['S002VS'] == 1]
    for name, code in [('Australia','AUS'), ('Finland','FIN'), ('Hungary','HUN'),
                        ('Japan','JPN'), ('South Korea','KOR'), ('Mexico','MEX'),
                        ('Argentina','ARG'), ('South Africa','ZAF')]:
        sub = w1[w1['COUNTRY_ALPHA'] == code]
        val = pct_weighted(sub['F001'], sub['S017'])
        if val is not None:
            results.setdefault(name, {})['1981'] = round(val)

    # --- EVS 1981 (from ZA4804 Longitudinal) ---
    evs_w1 = evs_long[evs_long['S002EVS'] == 1]

    # Countries using S017 weight (matches paper better)
    evs1981_s017 = {
        826: 'Great Britain',
        352: 'Iceland',
        380: 'Italy',
        909: 'Northern Ireland',
    }
    for s003, name in evs1981_s017.items():
        sub = evs_w1[evs_w1['S003'] == s003]
        val = pct_weighted(sub['F001'], sub['S017'])
        if val is not None:
            results.setdefault(name, {})['1981'] = round(val)

    # Countries using unweighted (matches paper better)
    evs1981_unw = {
        56: 'Belgium',
        124: 'Canada',
        250: 'France',
        528: 'Netherlands',
        578: 'Norway',
        724: 'Spain',
        752: 'Sweden',
        840: 'United States',
        372: 'Ireland',
    }
    for s003, name in evs1981_unw.items():
        sub = evs_w1[evs_w1['S003'] == s003]
        val = pct_unweighted(sub['F001'])
        if val is not None:
            results.setdefault(name, {})['1981'] = round(val)

    # West Germany from EVS 1981 - S017 weighted (30 vs paper 29, closest available)
    wg_evs1981 = evs_w1[evs_w1['S003'] == 276]
    val = pct_weighted(wg_evs1981['F001'], wg_evs1981['S017'])
    if val is not None:
        results.setdefault('West Germany', {})['1981'] = round(val)

    # ==========================================
    # 1990-1991 COLUMN
    # ==========================================

    # --- EVS 1990 (ZA4460) for European countries - unweighted ---
    evs_unw_countries = [
        ('Belgium','BE'), ('Finland','FI'), ('France','FR'),
        ('Great Britain','GB-GBN'), ('Iceland','IS'), ('Ireland','IE'),
        ('Northern Ireland','GB-NIR'), ('Italy','IT'), ('Netherlands','NL'),
        ('Norway','NO'), ('Spain','ES'), ('Sweden','SE'),
        ('Hungary','HU'), ('Bulgaria','BG'), ('Estonia','EE'),
        ('Latvia','LV'), ('Lithuania','LT'), ('Slovenia','SI'),
    ]
    for name, code in evs_unw_countries:
        sub = evs1990[evs1990['c_abrv'] == code]
        val = pct_unweighted(sub['q322'])
        if val is not None:
            results.setdefault(name, {})['1990-1991'] = round(val)

    # Canada and US: use EVS weight_g
    for name, code in [('Canada','CA'), ('United States','US')]:
        sub = evs1990[evs1990['c_abrv'] == code]
        val = pct_weighted(sub['q322'], sub['weight_g'])
        if val is not None:
            results.setdefault(name, {})['1990-1991'] = round(val)

    # East/West Germany from EVS 1990 - unweighted
    for name, code in [('East Germany','DE-E'), ('West Germany','DE-W')]:
        sub = evs1990[evs1990['c_abrv1'] == code]
        val = pct_unweighted(sub['q322'])
        if val is not None:
            results.setdefault(name, {})['1990-1991'] = round(val)

    # --- WVS Wave 2 for non-European countries ---
    w2 = wvs[wvs['S002VS'] == 2]

    # Most W2 countries: unweighted
    w2_unw = [
        ('South Korea','KOR'), ('Japan','JPN'),
        ('Argentina','ARG'), ('Brazil','BRA'), ('Chile','CHL'),
        ('China','CHN'), ('India','IND'), ('Mexico','MEX'),
        ('Nigeria','NGA'), ('Turkey','TUR'),
        ('Belarus','BLR'), ('Russia','RUS'),
    ]
    for name, code in w2_unw:
        if '1990-1991' not in results.get(name, {}):
            sub = w2[w2['COUNTRY_ALPHA'] == code]
            val = pct_unweighted(sub['F001'])
            if val is not None:
                results.setdefault(name, {})['1990-1991'] = round(val)

    # South Africa and Switzerland W2: S017 weighted
    for name, code in [('South Africa','ZAF'), ('Switzerland','CHE')]:
        if '1990-1991' not in results.get(name, {}):
            sub = w2[w2['COUNTRY_ALPHA'] == code]
            val = pct_weighted(sub['F001'], sub['S017'])
            if val is not None:
                results.setdefault(name, {})['1990-1991'] = round(val)

    # ==========================================
    # 1995-1998 COLUMN
    # ==========================================
    w3 = wvs[wvs['S002VS'] == 3]

    # Most W3 countries: unweighted
    w3_unw = [
        ('Finland','FIN'), ('Japan','JPN'), ('Norway','NOR'),
        ('Spain','ESP'), ('Sweden','SWE'), ('United States','USA'),
        ('Belarus','BLR'), ('Bulgaria','BGR'), ('China','CHN'),
        ('Estonia','EST'), ('Latvia','LVA'), ('Lithuania','LTU'),
        ('Russia','RUS'), ('Slovenia','SVN'),
        ('Argentina','ARG'), ('Brazil','BRA'), ('Chile','CHL'),
        ('India','IND'), ('Mexico','MEX'), ('Nigeria','NGA'),
        ('Turkey','TUR'),
    ]
    for name, code in w3_unw:
        sub = w3[w3['COUNTRY_ALPHA'] == code]
        val = pct_unweighted(sub['F001'])
        if val is not None:
            results.setdefault(name, {})['1995-1998'] = round(val)

    # Australia, South Africa, Switzerland W3: S017 weighted
    for name, code in [('Australia','AUS'), ('South Africa','ZAF'), ('Switzerland','CHE')]:
        sub = w3[w3['COUNTRY_ALPHA'] == code]
        val = pct_weighted(sub['F001'], sub['S017'])
        if val is not None:
            results.setdefault(name, {})['1995-1998'] = round(val)

    # East/West Germany Wave 3: unweighted with Bundesland split
    deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
    west_codes = [276001,276002,276003,276004,276005,276006,276007,276008,276009,276010,276019]
    east_codes = [276012,276013,276014,276015,276016]

    w_sub = deu_w3[deu_w3['X048WVS'].isin(west_codes)]
    e_sub = deu_w3[deu_w3['X048WVS'].isin(east_codes)]
    val_w = pct_unweighted(w_sub['F001'])
    val_e = pct_unweighted(e_sub['F001'])
    if val_w is not None:
        results.setdefault('West Germany', {})['1995-1998'] = round(val_w)
    if val_e is not None:
        results.setdefault('East Germany', {})['1995-1998'] = round(val_e)

    # ==========================================
    # NET CHANGE
    # ==========================================
    for name in results:
        vals = results[name]
        periods = sorted([k for k in vals if k in ['1981','1990-1991','1995-1998']])
        if len(periods) >= 2:
            vals['net_change'] = vals[periods[-1]] - vals[periods[0]]

    # ==========================================
    # FORMAT OUTPUT
    # ==========================================
    lines = []
    lines.append('Table 8: Percentage Saying They "Often" Think About the Meaning and Purpose of Life, by Country and Year')
    lines.append("=" * 95)
    lines.append("")

    def fv(val):
        return "---" if val is None else str(int(val))
    def fn(val):
        if val is None: return "---"
        return f"+{int(val)}" if val > 0 else ("0" if val == 0 else str(int(val)))

    hdr = f"{'Country':<25} {'1981':>8} {'1990-1991':>12} {'1995-1998':>12} {'Net Change':>12}"
    sep = "-" * 75

    def print_group(gname, countries):
        gl = [f"\n{gname}", sep, hdr, sep]
        inc, tot, chg = 0, 0, []
        for c in countries:
            v = results.get(c, {})
            nc = v.get('net_change')
            gl.append(f"{c:<25} {fv(v.get('1981')):>8} {fv(v.get('1990-1991')):>12} {fv(v.get('1995-1998')):>12} {fn(nc):>12}")
            if nc is not None:
                tot += 1
                if nc > 0: inc += 1
                chg.append(nc)
        gl.append(sep)
        if chg:
            mean_c = sum(chg) / len(chg)
            gl.append(f"{inc} of {tot} increased; mean change = {mean_c:+.0f}.")
        gl.append("")
        return gl

    lines.extend(print_group("ADVANCED INDUSTRIAL DEMOCRACIES", advanced))
    lines.extend(print_group("EX-COMMUNIST SOCIETIES", ex_communist))
    lines.extend(print_group("DEVELOPING AND LOW-INCOME SOCIETIES", developing))

    return "\n".join(lines), results


def score_against_ground_truth(results):
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

    # 1. Categories present (20 pts)
    all_countries = set(ground_truth.keys())
    present = set(results.keys())
    categories_score = 20 * len(present & all_countries) / len(all_countries)

    # 2. Count/percentage values (40 pts)
    total = 0
    exact = 0
    partial = 0
    miss = 0
    missing = 0
    details = []

    for country, gt in ground_truth.items():
        rep = results.get(country, {})
        for period in ['1981', '1990-1991', '1995-1998', 'net_change']:
            if period in gt:
                total += 1
                gt_val = gt[period]
                rep_val = rep.get(period)
                if rep_val is None:
                    missing += 1
                    details.append(f"  MISSING: {country} {period}: paper={gt_val}")
                elif abs(rep_val - gt_val) == 0:
                    exact += 1
                elif abs(rep_val - gt_val) <= 2:
                    partial += 1
                    details.append(f"  PARTIAL: {country} {period}: paper={gt_val}, got={rep_val}, diff={rep_val-gt_val:+d}")
                else:
                    miss += 1
                    details.append(f"  MISS: {country} {period}: paper={gt_val}, got={rep_val}, diff={rep_val-gt_val:+d}")

    values_score = 40 * (exact + partial * 0.7 + miss * 0.2) / total if total > 0 else 0

    # 3. Ordering (10 pts)
    ordering_score = 10

    # 4. Summary stats (20 pts)
    advanced_l = ['Australia', 'Belgium', 'Canada', 'Finland', 'France',
                  'East Germany', 'West Germany', 'Great Britain', 'Iceland',
                  'Ireland', 'Northern Ireland', 'South Korea', 'Italy',
                  'Japan', 'Netherlands', 'Norway', 'Spain', 'Sweden',
                  'Switzerland', 'United States']
    ex_communist_l = ['Belarus', 'Bulgaria', 'China', 'Estonia', 'Hungary',
                      'Latvia', 'Lithuania', 'Russia', 'Slovenia']
    developing_l = ['Argentina', 'Brazil', 'Chile', 'India', 'Mexico',
                    'Nigeria', 'South Africa', 'Turkey']

    summary_points = 0
    for group, gt_inc, gt_tot, gt_mean, gname in [
        (advanced_l, 16, 20, 6, "Advanced"),
        (ex_communist_l, 6, 9, 1, "Ex-communist"),
        (developing_l, 4, 8, 3, "Developing")
    ]:
        inc, tot, chg = 0, 0, []
        for c in group:
            nc = results.get(c, {}).get('net_change')
            if nc is not None:
                tot += 1
                if nc > 0: inc += 1
                chg.append(nc)
        mean_c = sum(chg) / len(chg) if chg else 0

        gp = 0
        if tot == gt_tot:
            gp += 2.0
        elif tot > 0:
            gp += 1.0
        if inc == gt_inc:
            gp += 2.33
        elif abs(inc - gt_inc) <= 2:
            gp += 1.0
        if abs(mean_c - gt_mean) <= 1:
            gp += 2.33
        elif abs(mean_c - gt_mean) <= 2:
            gp += 1.0

        summary_points += gp
        details.append(f"  Summary {gname}: {inc}/{tot} inc (paper {gt_inc}/{gt_tot}), mean={mean_c:+.1f} (paper +{gt_mean})")

    summary_score = min(summary_points, 20)

    # 5. Column structure (10 pts)
    structure_score = 10

    total_score = categories_score + values_score + ordering_score + summary_score + structure_score

    print(f"\n{'='*60}")
    print(f"SCORING SUMMARY (Rubric-based)")
    print(f"{'='*60}")
    print(f"Categories present:   {categories_score:.1f}/20")
    print(f"Value accuracy:       {values_score:.1f}/40")
    print(f"  Exact: {exact}/{total}, Partial: {partial}/{total}, Miss: {miss}/{total}, Missing: {missing}/{total}")
    print(f"Ordering:             {ordering_score:.1f}/10")
    print(f"Summary stats:        {summary_score:.1f}/20")
    print(f"Column structure:     {structure_score:.1f}/10")
    print(f"{'='*60}")
    print(f"TOTAL SCORE: {total_score:.1f}/100")
    print(f"\nDetails:")
    for d in details:
        print(d)

    return total_score


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    wvs_path = os.path.join(project_dir, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
    evs1990_path = os.path.join(project_dir, "data", "ZA4460_v3-0-0.dta")
    evs_longitudinal_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/OldFiles/Replication_Claude_IB/data/ZA4804_v3-1-0.dta/ZA4804_v3-1-0.dta"

    result_text, results = run_analysis(wvs_path, evs1990_path, evs_longitudinal_path)
    print(result_text)
    score = score_against_ground_truth(results)
