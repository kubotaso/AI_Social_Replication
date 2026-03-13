import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_analysis(data_source):
    df = pd.read_csv(data_source)

    # ---- Dependent Variables ----
    minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
    remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                        'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']

    for g in minority_genres + remaining_genres:
        df[g] = pd.to_numeric(df[g], errors='coerce')

    minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
    df['dv_minority'] = np.nan
    df.loc[minority_valid, 'dv_minority'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)

    remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)
    df['dv_remaining'] = np.nan
    df.loc[remaining_valid, 'dv_remaining'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

    # ---- Racism score ----
    # Use original 5 items (racmost, busing, racdif1, racdif2, racdif3) with racdif3 FLIPPED
    # Person-mean imputation with 4+ items valid
    racism_items_raw = ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']
    for item in racism_items_raw:
        df[item] = pd.to_numeric(df[item], errors='coerce')

    df['r_racmost'] = np.where(df['racmost'].notna(), (df['racmost'] == 1).astype(float), np.nan)
    df['r_busing'] = np.where(df['busing'].notna(), (df['busing'] == 2).astype(float), np.nan)
    df['r_racdif1'] = np.where(df['racdif1'].notna(), (df['racdif1'] == 2).astype(float), np.nan)
    df['r_racdif2'] = np.where(df['racdif2'].notna(), (df['racdif2'] == 2).astype(float), np.nan)
    df['r_racdif3'] = np.where(df['racdif3'].notna(), (df['racdif3'] == 2).astype(float), np.nan)

    coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
    n_valid_racism = pd.concat([df[c] for c in coded], axis=1).notna().sum(axis=1)

    # Person-mean imputation
    coded_df = pd.concat([df[c] for c in coded], axis=1)
    pmean = coded_df.mean(axis=1)
    for c in coded:
        coded_df[c] = coded_df[c].fillna(pmean)
    df['racism_score'] = coded_df.sum(axis=1)
    df.loc[n_valid_racism < 4, 'racism_score'] = np.nan

    # ---- Other IVs ----
    df['education'] = pd.to_numeric(df['educ'], errors='coerce')
    df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
    df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
    df['income_pc'] = df['realinc'] / df['hompop']
    df['occ_prestige'] = pd.to_numeric(df['prestg80'], errors='coerce')
    df['sex'] = pd.to_numeric(df['sex'], errors='coerce')
    df['female'] = (df['sex'] == 2).astype(int)
    df['age_var'] = pd.to_numeric(df['age'], errors='coerce')
    df['race'] = pd.to_numeric(df['race'], errors='coerce')
    df['black'] = (df['race'] == 2).astype(int)
    df['ethnic'] = pd.to_numeric(df['ethnic'], errors='coerce')
    df['hispanic'] = df['ethnic'].isin([17, 22, 25]).astype(int)
    df['other_race'] = (df['race'] == 3).astype(int)
    df['fund'] = pd.to_numeric(df['fund'], errors='coerce')
    df['conservative_protestant'] = (df['fund'] == 1).astype(int)
    df['relig'] = pd.to_numeric(df['relig'], errors='coerce')
    df['no_religion'] = (df['relig'] == 4).astype(int)
    df['region'] = pd.to_numeric(df['region'], errors='coerce')
    df['southern'] = (df['region'] == 3).astype(int)

    predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
                  'female', 'age_var', 'black', 'hispanic', 'other_race',
                  'conservative_protestant', 'no_religion', 'southern']

    predictor_labels = ['Racism score', 'Education', 'Household income per cap',
                        'Occupational prestige', 'Female', 'Age', 'Black', 'Hispanic',
                        'Other race', 'Conservative Protestant', 'No religion', 'Southern']

    results_text = ""

    for model_num, dv_name, dv_label in [(1, 'dv_minority', 'Dislike of 6 Minority Genres'),
                                          (2, 'dv_remaining', 'Dislike of 12 Remaining Genres')]:
        vars_needed = [dv_name] + predictors
        model_df = df[vars_needed].dropna()
        N = len(model_df)

        y_raw = model_df[dv_name].values
        X_raw = model_df[predictors].values

        y_mean, y_std = y_raw.mean(), y_raw.std(ddof=1)
        y_z = (y_raw - y_mean) / y_std
        X_means = X_raw.mean(axis=0)
        X_stds = X_raw.std(axis=0, ddof=1)
        X_z = (X_raw - X_means) / X_stds

        X_z_const = sm.add_constant(X_z)
        model_z = sm.OLS(y_z, X_z_const).fit()
        std_betas = model_z.params[1:]
        pvalues = model_z.pvalues[1:]

        X_raw_const = sm.add_constant(X_raw)
        model_raw = sm.OLS(y_raw, X_raw_const).fit()
        constant = model_raw.params[0]
        constant_pvalue = model_raw.pvalues[0]
        r_squared = model_raw.rsquared
        adj_r_squared = model_raw.rsquared_adj

        results_text += f"\nModel {model_num}: {dv_label}\n"
        results_text += f"{'Variable':<30} {'Beta':>8} {'p-value':>10} {'Sig':>5}\n"
        results_text += "-" * 55 + "\n"

        for i, (label, beta, pval) in enumerate(zip(predictor_labels, std_betas, pvalues)):
            sig = ""
            if pval < 0.001: sig = "***"
            elif pval < 0.01: sig = "**"
            elif pval < 0.05: sig = "*"
            results_text += f"{label:<30} {beta:>8.3f} {pval:>10.4f} {sig:>5}\n"

        const_sig = ""
        if constant_pvalue < 0.001: const_sig = "***"
        elif constant_pvalue < 0.01: const_sig = "**"
        elif constant_pvalue < 0.05: const_sig = "*"
        results_text += f"{'Constant':<30} {constant:>8.3f} {constant_pvalue:>10.4f} {const_sig:>5}\n"
        results_text += f"\nR-squared: {r_squared:.3f}\n"
        results_text += f"Adjusted R-squared: {adj_r_squared:.3f}\n"
        results_text += f"N: {N}\n"

    return results_text


def score_against_ground_truth():
    true_values = {
        'model1': {
            'betas': {
                'Racism score': 0.130, 'Education': -0.175, 'Household income per cap': -0.037,
                'Occupational prestige': -0.020, 'Female': -0.057, 'Age': 0.163,
                'Black': -0.132, 'Hispanic': -0.058, 'Other race': -0.017,
                'Conservative Protestant': 0.063, 'No religion': 0.057, 'Southern': 0.024
            },
            'significance': {
                'Racism score': '**', 'Education': '***', 'Household income per cap': '',
                'Occupational prestige': '', 'Female': '', 'Age': '***',
                'Black': '***', 'Hispanic': '', 'Other race': '',
                'Conservative Protestant': '', 'No religion': '', 'Southern': ''
            },
            'constant': 2.415, 'constant_sig': '***',
            'r_squared': 0.145, 'adj_r_squared': 0.129, 'N': 644
        },
        'model2': {
            'betas': {
                'Racism score': 0.080, 'Education': -0.242, 'Household income per cap': -0.065,
                'Occupational prestige': 0.005, 'Female': -0.070, 'Age': 0.126,
                'Black': 0.042, 'Hispanic': -0.029, 'Other race': 0.047,
                'Conservative Protestant': 0.048, 'No religion': 0.024, 'Southern': 0.069
            },
            'significance': {
                'Racism score': '', 'Education': '***', 'Household income per cap': '',
                'Occupational prestige': '', 'Female': '', 'Age': '**',
                'Black': '', 'Hispanic': '', 'Other race': '',
                'Conservative Protestant': '', 'No religion': '', 'Southern': ''
            },
            'constant': 7.860, 'constant_sig': '',
            'r_squared': 0.147, 'adj_r_squared': 0.130, 'N': 605
        }
    }

    result = run_analysis("gss1993_clean.csv")
    print(result)

    lines = result.strip().split('\n')
    models = {}
    current_model = None

    for line in lines:
        if line.startswith('Model 1:'):
            current_model = 'model1'
            models[current_model] = {'betas': {}, 'significance': {}}
        elif line.startswith('Model 2:'):
            current_model = 'model2'
            models[current_model] = {'betas': {}, 'significance': {}}
        elif current_model and line.strip().startswith('Constant'):
            parts = line.split()
            models[current_model]['constant'] = float(parts[1])
        elif current_model and line.startswith('R-squared:'):
            models[current_model]['r_squared'] = float(line.split(':')[1].strip())
        elif current_model and line.startswith('Adjusted R-squared:'):
            models[current_model]['adj_r_squared'] = float(line.split(':')[1].strip())
        elif current_model and line.startswith('N:'):
            val = line.split(':')[1].strip()
            if val.isdigit():
                models[current_model]['N'] = int(val)
        elif current_model and not line.startswith('-') and not line.startswith('Variable') and line.strip():
            for label in true_values['model1']['betas'].keys():
                if line.strip().startswith(label):
                    rest = line.strip()[len(label):].split()
                    if len(rest) >= 2:
                        beta = float(rest[0])
                        sig = rest[2] if len(rest) >= 3 else ''
                        models[current_model]['betas'][label] = beta
                        models[current_model]['significance'][label] = sig
                    break

    total_score = 0
    for model_key, model_label in [('model1', 'Model 1'), ('model2', 'Model 2')]:
        true = true_values[model_key]
        gen = models.get(model_key, {})
        print(f"\n=== Scoring {model_label} ===")

        coeff_score = 0
        n_vars = len(true['betas'])
        for var in true['betas']:
            true_b = true['betas'][var]
            gen_b = gen.get('betas', {}).get(var, None)
            if gen_b is not None:
                diff = abs(true_b - gen_b)
                if diff <= 0.005: coeff_score += 1.0
                elif diff <= 0.02: coeff_score += 0.75
                elif diff <= 0.05: coeff_score += 0.5
                elif diff <= 0.10: coeff_score += 0.25
                print(f"  {var}: true={true_b:.3f}, gen={gen_b:.3f}, diff={diff:.3f}")
            else:
                print(f"  {var}: MISSING")
        coeff_pts = (coeff_score / n_vars) * 15
        total_score += coeff_pts
        print(f"  Coefficient score: {coeff_pts:.1f}/15")

        gen_n = gen.get('N', 0)
        n_diff_pct = abs(gen_n - true['N']) / true['N'] * 100
        if n_diff_pct <= 1: n_pts = 10
        elif n_diff_pct <= 5: n_pts = 8
        elif n_diff_pct <= 10: n_pts = 5
        elif n_diff_pct <= 20: n_pts = 3
        else: n_pts = 0
        total_score += n_pts
        print(f"  N: true={true['N']}, gen={gen_n}, diff={n_diff_pct:.1f}%, pts={n_pts}/10")

        sig_score = 0
        for var in true['significance']:
            true_s = true['significance'][var]
            gen_s = gen.get('significance', {}).get(var, '')
            if true_s == gen_s: sig_score += 1.0
            elif true_s != '' and gen_s != '': sig_score += 0.5
            print(f"  {var}: true_sig='{true_s}', gen_sig='{gen_s}'")
        sig_pts = (sig_score / n_vars) * 15
        total_score += sig_pts
        print(f"  Significance score: {sig_pts:.1f}/15")

        present = sum(1 for v in true['betas'] if v in gen.get('betas', {}))
        var_pts = (present / n_vars) * 5
        total_score += var_pts

        r2_pts = 0
        gen_r2 = gen.get('r_squared', 0)
        gen_adj_r2 = gen.get('adj_r_squared', 0)
        if abs(gen_r2 - true['r_squared']) <= 0.005: r2_pts += 2.5
        elif abs(gen_r2 - true['r_squared']) <= 0.02: r2_pts += 1.5
        elif abs(gen_r2 - true['r_squared']) <= 0.05: r2_pts += 0.5
        if abs(gen_adj_r2 - true['adj_r_squared']) <= 0.005: r2_pts += 2.5
        elif abs(gen_adj_r2 - true['adj_r_squared']) <= 0.02: r2_pts += 1.5
        elif abs(gen_adj_r2 - true['adj_r_squared']) <= 0.05: r2_pts += 0.5
        total_score += r2_pts
        print(f"  R2: true={true['r_squared']:.3f}, gen={gen_r2:.3f}, Adj R2: true={true['adj_r_squared']:.3f}, gen={gen_adj_r2:.3f}, pts={r2_pts:.1f}/5")

    print(f"\n=== TOTAL SCORE: {total_score:.1f}/100 ===")
    return total_score

if __name__ == "__main__":
    score = score_against_ground_truth()
