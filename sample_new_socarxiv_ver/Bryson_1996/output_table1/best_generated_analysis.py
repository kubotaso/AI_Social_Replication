import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def run_analysis(data_source):
    df = pd.read_csv(data_source)

    # === DEPENDENT VARIABLE: Musical exclusiveness ===
    music_vars = ['latin', 'jazz', 'blues', 'musicals', 'oldies', 'classicl',
                  'reggae', 'bigband', 'newage', 'opera', 'blugrass', 'folk',
                  'moodeasy', 'conrock', 'rap', 'hvymetal', 'country', 'gospel']

    for v in music_vars:
        df[v] = pd.to_numeric(df[v], errors='coerce')
        df.loc[~df[v].isin([1, 2, 3, 4, 5]), v] = np.nan

    # Listwise deletion: require all 18 genres valid
    df['music_excl'] = df[music_vars].apply(
        lambda row: (row >= 4).sum() if row.notna().all() else np.nan, axis=1
    )

    print(f"Music exclusiveness: N={df['music_excl'].notna().sum()}, "
          f"mean={df['music_excl'].dropna().mean():.2f}, "
          f"SD={df['music_excl'].dropna().std():.2f}")

    # === INDEPENDENT VARIABLES ===

    # Education
    df['education'] = pd.to_numeric(df['educ'], errors='coerce')

    # Household income per capita
    df['realinc_num'] = pd.to_numeric(df['realinc'], errors='coerce')
    df['hompop_num'] = pd.to_numeric(df['hompop'], errors='coerce')
    df.loc[df['hompop_num'] <= 0, 'hompop_num'] = np.nan
    df['income_pc'] = df['realinc_num'] / df['hompop_num']

    # Occupational prestige
    df['prestige'] = pd.to_numeric(df['prestg80'], errors='coerce')

    # Female
    df['sex_num'] = pd.to_numeric(df['sex'], errors='coerce')
    df['female'] = (df['sex_num'] == 2).astype(float)
    df.loc[df['sex_num'].isna(), 'female'] = np.nan

    # Age
    df['age_num'] = pd.to_numeric(df['age'], errors='coerce')

    # Black
    df['race_num'] = pd.to_numeric(df['race'], errors='coerce')
    df['black'] = (df['race_num'] == 2).astype(float)
    df.loc[df['race_num'].isna(), 'black'] = np.nan

    # Hispanic: treat missing ethnic as non-Hispanic
    df['ethnic_num'] = pd.to_numeric(df['ethnic'], errors='coerce')
    df['hispanic'] = df['ethnic_num'].isin([17, 22, 25]).astype(float)
    # Missing ethnic -> 0 (not Hispanic)

    # Other race
    df['other_race'] = (df['race_num'] == 3).astype(float)
    df.loc[df['race_num'].isna(), 'other_race'] = np.nan

    # Conservative Protestant: treat missing fund as non-conservative
    df['fund_num'] = pd.to_numeric(df['fund'], errors='coerce')
    df['cons_prot'] = (df['fund_num'] == 1).astype(float)
    df.loc[df['fund_num'].isna(), 'cons_prot'] = 0

    # No religion
    df['relig_num'] = pd.to_numeric(df['relig'], errors='coerce')
    df['no_religion'] = (df['relig_num'] == 4).astype(float)
    df.loc[df['relig_num'].isna(), 'no_religion'] = 0

    # Southern
    df['region_num'] = pd.to_numeric(df['region'], errors='coerce')
    df['southern'] = (df['region_num'] == 3).astype(float)
    df.loc[df['region_num'].isna(), 'southern'] = np.nan

    # Political intolerance: sum of intolerant responses
    spk_vars = ['spkath', 'spkrac', 'spkcom', 'spkmil', 'spkhomo']
    col_vars = ['colath', 'colrac', 'colcom', 'colmil', 'colhomo']
    lib_vars = ['libath', 'librac', 'libcom', 'libmil', 'libhomo']

    for v in spk_vars + col_vars + lib_vars:
        df[v] = pd.to_numeric(df[v], errors='coerce')

    for v in spk_vars:
        df[v + '_intol'] = np.where(df[v].isna(), np.nan, (df[v] == 2).astype(float))
    for v in col_vars:
        df[v + '_intol'] = np.where(df[v].isna(), np.nan, (df[v] == 5).astype(float))
    for v in lib_vars:
        df[v + '_intol'] = np.where(df[v].isna(), np.nan, (df[v] == 1).astype(float))

    intol_items = [v + '_intol' for v in spk_vars + col_vars + lib_vars]

    # Allow respondents with at least 13 of 15 items answered
    # Sum the available items (don't scale -- paper reports raw sum)
    df['tol_count'] = df[intol_items].notna().sum(axis=1)
    df['pol_intol'] = df[intol_items].apply(
        lambda row: row.sum() if row.notna().sum() >= 13 else np.nan, axis=1
    )

    print(f"Political intolerance: N={df['pol_intol'].notna().sum()}, "
          f"mean={df['pol_intol'].dropna().mean():.2f}, "
          f"SD={df['pol_intol'].dropna().std():.2f}")

    # === MODELS ===
    ses_vars = ['education', 'income_pc', 'prestige']
    demo_vars = ses_vars + ['female', 'age_num', 'black', 'hispanic',
                            'other_race', 'cons_prot', 'no_religion', 'southern']
    intol_vars = demo_vars + ['pol_intol']

    var_labels = {
        'education': 'Education',
        'income_pc': 'Household income per cap',
        'prestige': 'Occupational prestige',
        'female': 'Female',
        'age_num': 'Age',
        'black': 'Black',
        'hispanic': 'Hispanic',
        'other_race': 'Other race',
        'cons_prot': 'Conservative Protestant',
        'no_religion': 'No religion',
        'southern': 'Southern',
        'pol_intol': 'Political intolerance'
    }

    results_text = "Table 1: Standardized OLS Coefficients\n"
    results_text += "=" * 80 + "\n\n"

    model_specs = [
        ("SES Model", ses_vars),
        ("Demographic Model", demo_vars),
        ("Political Intolerance Model", intol_vars),
    ]

    model_results = {}

    for model_name, predictors in model_specs:
        all_vars = ['music_excl'] + predictors
        subset = df[all_vars].dropna()
        n = len(subset)

        y = subset['music_excl']
        X = subset[predictors]

        # Standardize
        y_std = (y - y.mean()) / y.std()
        X_std = (X - X.mean()) / X.std()

        X_std_c = sm.add_constant(X_std)
        model = sm.OLS(y_std, X_std_c).fit()

        # Unstandardized for constant
        X_c = sm.add_constant(X)
        model_unstd = sm.OLS(y, X_c).fit()
        constant = model_unstd.params['const']

        results_text += f"{model_name} (N={n})\n"
        results_text += "-" * 50 + "\n"

        model_data = {}
        for var in predictors:
            beta = model.params[var]
            pval = model.pvalues[var]
            sig = ''
            if pval < 0.001:
                sig = '***'
            elif pval < 0.01:
                sig = '**'
            elif pval < 0.05:
                sig = '*'
            label = var_labels.get(var, var)
            results_text += f"  {label:30s}  {beta:8.3f}{sig}\n"
            model_data[var] = {'beta': beta, 'pval': pval, 'sig': sig}

        results_text += f"  {'Constant':30s}  {constant:8.3f}\n"
        results_text += f"  R-squared:    {model.rsquared:.3f}\n"
        results_text += f"  Adj R-squared:{model.rsquared_adj:.3f}\n"
        results_text += f"  N:            {n}\n\n"

        model_results[model_name] = {
            'coeffs': model_data,
            'constant': constant,
            'r2': model.rsquared,
            'adj_r2': model.rsquared_adj,
            'n': n
        }

    results_text += "Significance: *p<.05 **p<.01 ***p<.001\n"

    score, score_detail = score_against_ground_truth(model_results)
    results_text += "\n" + "=" * 80 + "\n"
    results_text += f"SCORE: {score}/100\n"
    results_text += score_detail

    print(results_text)
    return results_text


def score_against_ground_truth(model_results):
    """Score against paper's Table 1."""
    ground_truth = {
        'SES Model': {
            'coeffs': {
                'education': {'beta': -0.322, 'sig': '***'},
                'income_pc': {'beta': -0.037, 'sig': ''},
                'prestige': {'beta': 0.016, 'sig': ''},
            },
            'constant': 10.920, 'r2': 0.107, 'adj_r2': 0.104, 'n': 787,
        },
        'Demographic Model': {
            'coeffs': {
                'education': {'beta': -0.246, 'sig': '***'},
                'income_pc': {'beta': -0.054, 'sig': ''},
                'prestige': {'beta': -0.006, 'sig': ''},
                'female': {'beta': -0.083, 'sig': '*'},
                'age_num': {'beta': 0.140, 'sig': '***'},
                'black': {'beta': 0.029, 'sig': ''},
                'hispanic': {'beta': -0.029, 'sig': ''},
                'other_race': {'beta': 0.005, 'sig': ''},
                'cons_prot': {'beta': 0.059, 'sig': ''},
                'no_religion': {'beta': -0.012, 'sig': ''},
                'southern': {'beta': 0.097, 'sig': '**'},
            },
            'constant': 8.507, 'r2': 0.151, 'adj_r2': 0.139, 'n': 756,
        },
        'Political Intolerance Model': {
            'coeffs': {
                'education': {'beta': -0.151, 'sig': '**'},
                'income_pc': {'beta': -0.009, 'sig': ''},
                'prestige': {'beta': -0.022, 'sig': ''},
                'female': {'beta': -0.095, 'sig': '*'},
                'age_num': {'beta': 0.110, 'sig': '*'},
                'black': {'beta': 0.049, 'sig': ''},
                'hispanic': {'beta': 0.031, 'sig': ''},
                'other_race': {'beta': 0.053, 'sig': ''},
                'cons_prot': {'beta': 0.066, 'sig': ''},
                'no_religion': {'beta': 0.024, 'sig': ''},
                'southern': {'beta': 0.121, 'sig': '**'},
                'pol_intol': {'beta': 0.164, 'sig': '***'},
            },
            'constant': 6.516, 'r2': 0.169, 'adj_r2': 0.148, 'n': 503,
        },
    }

    detail = "\n--- SCORING DETAIL ---\n"

    # 1. Coefficients (30 pts)
    total_c = matching_c = 0
    c_details = []
    for mn, gt in ground_truth.items():
        if mn not in model_results:
            continue
        gen = model_results[mn]
        for var, gi in gt['coeffs'].items():
            total_c += 1
            gb = gi['beta']
            rb = gen['coeffs'].get(var, {}).get('beta', None)
            if rb is not None:
                sign_ok = (gb >= 0) == (rb >= 0) or abs(gb) < 0.01
                diff = abs(gb - rb)
                if sign_ok and diff <= 0.05:
                    matching_c += 1
                    c_details.append(f"  OK  {mn}/{var}: paper={gb:.3f} gen={rb:.3f}")
                else:
                    c_details.append(f"  MISS {mn}/{var}: paper={gb:.3f} gen={rb:.3f} diff={diff:.3f}")
            else:
                c_details.append(f"  MISS {mn}/{var}: not found")
    cs = (matching_c / total_c) * 30 if total_c > 0 else 0
    detail += f"\n1. Coefficients ({matching_c}/{total_c}): {cs:.1f}/30\n" + "\n".join(c_details) + "\n"

    # 2. N (20 pts)
    ns = nc = 0
    n_det = []
    for mn, gt in ground_truth.items():
        if mn not in model_results:
            continue
        nc += 1
        gn = gt['n']
        rn = model_results[mn]['n']
        pd_ = abs(gn - rn) / gn
        if pd_ <= 0.05:
            ns += 1
            n_det.append(f"  OK  {mn}: paper={gn} gen={rn} ({pd_*100:.1f}%)")
        else:
            n_det.append(f"  MISS {mn}: paper={gn} gen={rn} ({pd_*100:.1f}%)")
    n_score = (ns / nc) * 20 if nc > 0 else 0
    detail += f"\n2. Sample sizes ({ns}/{nc}): {n_score:.1f}/20\n" + "\n".join(n_det) + "\n"

    # 3. Significance (30 pts)
    ts = ms = 0
    s_det = []
    for mn, gt in ground_truth.items():
        if mn not in model_results:
            continue
        gen = model_results[mn]
        for var, gi in gt['coeffs'].items():
            ts += 1
            gs = gi['sig']
            rs = gen['coeffs'].get(var, {}).get('sig', None)
            if rs is not None and rs == gs:
                ms += 1
                s_det.append(f"  OK  {mn}/{var}: '{gs}'='{rs}'")
            else:
                s_det.append(f"  MISS {mn}/{var}: paper='{gs}' gen='{rs}'")
    ss = (ms / ts) * 30 if ts > 0 else 0
    detail += f"\n3. Significance ({ms}/{ts}): {ss:.1f}/30\n" + "\n".join(s_det) + "\n"

    # 4. Variables (10 pts)
    tv = pv = 0
    for mn, gt in ground_truth.items():
        if mn not in model_results:
            continue
        for var in gt['coeffs']:
            tv += 1
            if var in model_results[mn]['coeffs']:
                pv += 1
    vs = (pv / tv) * 10 if tv > 0 else 0
    detail += f"\n4. Variables ({pv}/{tv}): {vs:.1f}/10\n"

    # 5. R2 (10 pts)
    rm = rt = 0
    r_det = []
    for mn, gt in ground_truth.items():
        if mn not in model_results:
            continue
        gen = model_results[mn]
        for m in ['r2', 'adj_r2']:
            rt += 1
            gv = gt[m]
            rv = gen[m]
            if abs(gv - rv) <= 0.02:
                rm += 1
                r_det.append(f"  OK  {mn}/{m}: paper={gv:.3f} gen={rv:.3f}")
            else:
                r_det.append(f"  MISS {mn}/{m}: paper={gv:.3f} gen={rv:.3f}")
    rs = (rm / rt) * 10 if rt > 0 else 0
    detail += f"\n5. R-squared ({rm}/{rt}): {rs:.1f}/10\n" + "\n".join(r_det) + "\n"

    total = round(cs + n_score + ss + vs + rs)
    detail += f"\nTOTAL SCORE: {total}/100\n"
    return total, detail


if __name__ == "__main__":
    result = run_analysis("gss1993_clean.csv")
