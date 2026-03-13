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
    # Key fix: if ethnic is missing, treat as non-Hispanic (0) rather than NA
    # This matches common practice in survey analysis

    # Other race
    df['other_race'] = (df['race_num'] == 3).astype(float)
    df.loc[df['race_num'].isna(), 'other_race'] = np.nan

    # Conservative Protestant: treat missing fund as non-conservative
    df['fund_num'] = pd.to_numeric(df['fund'], errors='coerce')
    df['cons_prot'] = (df['fund_num'] == 1).astype(float)
    # Key fix: if fund is missing, treat as 0 (not conservative protestant)
    df.loc[df['fund_num'].isna(), 'cons_prot'] = 0

    # No religion: treat missing relig as not "no religion"
    df['relig_num'] = pd.to_numeric(df['relig'], errors='coerce')
    df['no_religion'] = (df['relig_num'] == 4).astype(float)
    df.loc[df['relig_num'].isna(), 'no_religion'] = 0

    # Southern
    df['region_num'] = pd.to_numeric(df['region'], errors='coerce')
    df['southern'] = (df['region_num'] == 3).astype(float)
    df.loc[df['region_num'].isna(), 'southern'] = np.nan

    # Political intolerance: sum of intolerant responses across 15 items
    spk_vars = ['spkath', 'spkrac', 'spkcom', 'spkmil', 'spkhomo']
    col_vars = ['colath', 'colrac', 'colcom', 'colmil', 'colhomo']
    lib_vars = ['libath', 'librac', 'libcom', 'libmil', 'libhomo']

    for v in spk_vars + col_vars + lib_vars:
        df[v] = pd.to_numeric(df[v], errors='coerce')

    # spk: intolerant if == 2 (not allowed to speak)
    for v in spk_vars:
        df[v + '_intol'] = (df[v] == 2).astype(float)
        df.loc[df[v].isna(), v + '_intol'] = np.nan

    # col: coded 4=yes(allow), 5=no(fire/not allow) in this dataset
    # intolerant if == 5
    for v in col_vars:
        df[v + '_intol'] = (df[v] == 5).astype(float)
        df.loc[df[v].isna(), v + '_intol'] = np.nan

    # lib: intolerant if == 1 (remove book)
    for v in lib_vars:
        df[v + '_intol'] = (df[v] == 1).astype(float)
        df.loc[df[v].isna(), v + '_intol'] = np.nan

    intol_items = [v + '_intol' for v in spk_vars + col_vars + lib_vars]
    # Require all 15 items non-missing
    df['pol_intol'] = df[intol_items].apply(
        lambda row: row.sum() if row.notna().all() else np.nan, axis=1
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

        # Standardize all variables
        y_std = (y - y.mean()) / y.std()
        X_std = (X - X.mean()) / X.std()

        # Add constant
        X_std_c = sm.add_constant(X_std)
        model = sm.OLS(y_std, X_std_c).fit()

        # Unstandardized model for the constant
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
    """Score the generated results against the paper's Table 1 values."""

    ground_truth = {
        'SES Model': {
            'coeffs': {
                'education': {'beta': -0.322, 'sig': '***'},
                'income_pc': {'beta': -0.037, 'sig': ''},
                'prestige': {'beta': 0.016, 'sig': ''},
            },
            'constant': 10.920,
            'r2': 0.107,
            'adj_r2': 0.104,
            'n': 787,
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
            'constant': 8.507,
            'r2': 0.151,
            'adj_r2': 0.139,
            'n': 756,
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
            'constant': 6.516,
            'r2': 0.169,
            'adj_r2': 0.148,
            'n': 503,
        },
    }

    detail = "\n--- SCORING DETAIL ---\n"

    # 1. Coefficient signs and magnitudes (30 points)
    total_coeffs = 0
    matching_coeffs = 0
    coeff_details = []
    for model_name, gt in ground_truth.items():
        if model_name not in model_results:
            continue
        gen = model_results[model_name]
        for var, gt_info in gt['coeffs'].items():
            total_coeffs += 1
            gt_beta = gt_info['beta']
            gen_beta = gen['coeffs'].get(var, {}).get('beta', None)
            if gen_beta is not None:
                sign_match = (gt_beta >= 0) == (gen_beta >= 0) or abs(gt_beta) < 0.01
                diff = abs(gt_beta - gen_beta)
                if sign_match and diff <= 0.05:
                    matching_coeffs += 1
                    coeff_details.append(f"  OK  {model_name}/{var}: paper={gt_beta:.3f} gen={gen_beta:.3f}")
                else:
                    coeff_details.append(f"  MISS {model_name}/{var}: paper={gt_beta:.3f} gen={gen_beta:.3f} diff={diff:.3f}")
            else:
                coeff_details.append(f"  MISS {model_name}/{var}: not found")

    coeff_score = (matching_coeffs / total_coeffs) * 30 if total_coeffs > 0 else 0
    detail += f"\n1. Coefficients ({matching_coeffs}/{total_coeffs} match): {coeff_score:.1f}/30\n"
    detail += "\n".join(coeff_details) + "\n"

    # 2. Sample size N (20 points)
    n_details = []
    n_score_sum = 0
    n_count = 0
    for model_name, gt in ground_truth.items():
        if model_name not in model_results:
            continue
        n_count += 1
        gt_n = gt['n']
        gen_n = model_results[model_name]['n']
        pct_diff = abs(gt_n - gen_n) / gt_n
        if pct_diff <= 0.05:
            n_score_sum += 1
            n_details.append(f"  OK  {model_name}: paper={gt_n} gen={gen_n} ({pct_diff*100:.1f}%)")
        else:
            n_details.append(f"  MISS {model_name}: paper={gt_n} gen={gen_n} ({pct_diff*100:.1f}%)")

    n_score = (n_score_sum / n_count) * 20 if n_count > 0 else 0
    detail += f"\n2. Sample sizes ({n_score_sum}/{n_count} within 5%): {n_score:.1f}/20\n"
    detail += "\n".join(n_details) + "\n"

    # 3. Significance levels (30 points)
    total_sigs = 0
    matching_sigs = 0
    sig_details = []
    for model_name, gt in ground_truth.items():
        if model_name not in model_results:
            continue
        gen = model_results[model_name]
        for var, gt_info in gt['coeffs'].items():
            total_sigs += 1
            gt_sig = gt_info['sig']
            gen_sig = gen['coeffs'].get(var, {}).get('sig', None)
            if gen_sig is not None and gen_sig == gt_sig:
                matching_sigs += 1
                sig_details.append(f"  OK  {model_name}/{var}: paper='{gt_sig}' gen='{gen_sig}'")
            else:
                sig_details.append(f"  MISS {model_name}/{var}: paper='{gt_sig}' gen='{gen_sig}'")

    sig_score = (matching_sigs / total_sigs) * 30 if total_sigs > 0 else 0
    detail += f"\n3. Significance ({matching_sigs}/{total_sigs} match): {sig_score:.1f}/30\n"
    detail += "\n".join(sig_details) + "\n"

    # 4. Variables present (10 points)
    total_vars = 0
    present_vars = 0
    for model_name, gt in ground_truth.items():
        if model_name not in model_results:
            continue
        gen = model_results[model_name]
        for var in gt['coeffs']:
            total_vars += 1
            if var in gen['coeffs']:
                present_vars += 1

    var_score = (present_vars / total_vars) * 10 if total_vars > 0 else 0
    detail += f"\n4. Variables present ({present_vars}/{total_vars}): {var_score:.1f}/10\n"

    # 5. R-squared (10 points)
    r2_details = []
    r2_matches = 0
    r2_total = 0
    for model_name, gt in ground_truth.items():
        if model_name not in model_results:
            continue
        gen = model_results[model_name]
        for metric in ['r2', 'adj_r2']:
            r2_total += 1
            gt_val = gt[metric]
            gen_val = gen[metric]
            if abs(gt_val - gen_val) <= 0.02:
                r2_matches += 1
                r2_details.append(f"  OK  {model_name}/{metric}: paper={gt_val:.3f} gen={gen_val:.3f}")
            else:
                r2_details.append(f"  MISS {model_name}/{metric}: paper={gt_val:.3f} gen={gen_val:.3f}")

    r2_score = (r2_matches / r2_total) * 10 if r2_total > 0 else 0
    detail += f"\n5. R-squared ({r2_matches}/{r2_total} within 0.02): {r2_score:.1f}/10\n"
    detail += "\n".join(r2_details) + "\n"

    total_score = round(coeff_score + n_score + sig_score + var_score + r2_score)
    detail += f"\nTOTAL SCORE: {total_score}/100\n"

    return total_score, detail


if __name__ == "__main__":
    result = run_analysis("gss1993_clean.csv")
