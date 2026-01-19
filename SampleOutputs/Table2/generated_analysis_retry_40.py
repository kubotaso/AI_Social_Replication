def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    os.makedirs("./output", exist_ok=True)

    # -----------------------
    # Helpers
    # -----------------------
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clean_series(s):
        """Convert to numeric and set common GSS missing sentinels to NaN."""
        x = to_num(s).copy()
        # Conservative sentinel set commonly used in GSS extracts
        sentinels = {8, 9, 98, 99, 998, 999, 9998, 9999}
        x = x.mask(x.isin(sentinels))
        return x

    def likert_dislike_indicator(s):
        """
        Music taste items: 1-5; dislike = {4,5}; like/neutral = {1,2,3}.
        Anything outside 1..5 (incl. NA-coded) -> missing.
        """
        x = clean_series(s)
        x = x.where(x.between(1, 5))
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin([1, 2, 3])] = 0.0
        out.loc[x.isin([4, 5])] = 1.0
        return out

    def binary_from_codes(s, true_codes, false_codes):
        x = clean_series(s)
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[x.isin(false_codes)] = 0.0
        out.loc[x.isin(true_codes)] = 1.0
        return out

    def build_dislike_count(df, item_cols):
        """Count dislikes across items; require complete data on all component items."""
        mat = pd.concat([likert_dislike_indicator(df[c]).rename(c) for c in item_cols], axis=1)
        # require all items observed to mirror "DK treated as missing; cases excluded"
        return mat.sum(axis=1, min_count=len(item_cols))

    def standardize_for_beta(x):
        """
        Standardize using sample SD with ddof=1 (common in standardized-beta reporting).
        If sd==0 -> all NaN.
        """
        x = to_num(x)
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=x.index, dtype="float64")
        return (x - mu) / sd

    def compute_standardized_betas_from_unstd(model, y, X_no_const):
        """
        beta_j = b_j * sd(x_j) / sd(y), computed on the analytic sample.
        Do not compute a standardized beta for the intercept.
        """
        y_sd = y.std(ddof=1)
        betas = {}
        for c in X_no_const.columns:
            x_sd = X_no_const[c].std(ddof=1)
            b = model.params.get(c, np.nan)
            if not np.isfinite(y_sd) or y_sd == 0 or not np.isfinite(x_sd) or x_sd == 0:
                betas[c] = np.nan
            else:
                betas[c] = b * (x_sd / y_sd)
        return pd.Series(betas)

    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def fit_table2_model(df, dv, x_map_ordered, model_name):
        """
        Fit OLS on unstandardized variables, then compute standardized betas (slopes only)
        using analytic-sample SDs. Output "paper style" table in canonical order with names.
        """
        # Build design with canonical variable names
        needed = [dv] + list(x_map_ordered.values())
        d = df[needed].copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(axis=0, how="any")

        # Fail early if sample too small
        if d.shape[0] < (len(x_map_ordered) + 5):
            raise ValueError(f"{model_name}: not enough complete cases after listwise deletion (n={d.shape[0]}).")

        # Prepare y and X
        y = to_num(d[dv])
        X = pd.DataFrame({k: to_num(d[v]) for k, v in x_map_ordered.items()}, index=d.index)

        # Check zero variance predictors; do NOT drop silently (Table 2 includes them)
        zero_var = [c for c in X.columns if X[c].std(ddof=1) == 0 or not np.isfinite(X[c].std(ddof=1))]
        if len(zero_var) > 0:
            # Provide diagnostics to help user fix data/coding rather than silently dropping
            raise ValueError(
                f"{model_name}: one or more predictors have zero variance in the analytic sample: {zero_var}."
                " Check coding/sample restrictions."
            )

        # Fit OLS with intercept
        Xc = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, Xc).fit()

        # Standardized betas (slopes only)
        beta = compute_standardized_betas_from_unstd(model, y, X)

        # Assemble canonical output
        rows = []
        for term in x_map_ordered.keys():
            rows.append(
                {
                    "term": term,
                    "beta": float(beta.get(term, np.nan)),
                    "p_value": float(model.pvalues.get(term, np.nan)),
                }
            )
        # Constant row (unstandardized intercept)
        rows.append(
            {
                "term": "Constant",
                "beta": float(model.params.get("const", np.nan)),  # keep intercept as unstd constant
                "p_value": float(model.pvalues.get("const", np.nan)),
            }
        )
        out = pd.DataFrame(rows)

        # Add stars for display (computed from replication p-values; not from paper)
        out["stars"] = out["p_value"].map(stars)
        out["beta_str"] = out["beta"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}") + out["stars"]

        # Fit stats
        fit = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "dv": dv,
                    "n": int(model.nobs),
                    "k_predictors": int(model.df_model),  # excludes intercept
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            ]
        )

        # Full coefficient table (replication output; SEs not in paper)
        full = pd.DataFrame(
            {
                "b_unstd": model.params,
                "std_err": model.bse,
                "t": model.tvalues,
                "p_value": model.pvalues,
            }
        )
        # Add standardized betas to full table for predictors (not intercept)
        full["beta_std"] = np.nan
        for term in x_map_ordered.keys():
            if term in full.index:
                full.loc[term, "beta_std"] = beta.get(term, np.nan)

        return model, out, full, fit

    # -----------------------
    # Load and filter data
    # -----------------------
    df = pd.read_csv(data_source)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required
    for c in ["year", "id"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["year"] = to_num(df["year"])
    df = df.loc[df["year"] == 1993].copy()

    # -----------------------
    # Construct DVs
    # -----------------------
    minority_items = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    other12_items = [
        "bigband",
        "blugrass",
        "country",
        "musicals",
        "classicl",
        "folk",
        "moodeasy",
        "newage",
        "opera",
        "conrock",
        "oldies",
        "hvymetal",
    ]
    for c in minority_items + other12_items:
        if c not in df.columns:
            raise ValueError(f"Missing music item column: {c}")

    df["dislike_minority_genres"] = build_dislike_count(df, minority_items)
    df["dislike_other12_genres"] = build_dislike_count(df, other12_items)

    # -----------------------
    # Construct racism score (0-5)
    # -----------------------
    for c in ["rachaf", "busing", "racdif1", "racdif3", "racdif4"]:
        if c not in df.columns:
            raise ValueError(f"Missing racism item column: {c}")

    rac1 = binary_from_codes(df["rachaf"], true_codes=[1], false_codes=[2])   # object to >half black school
    rac2 = binary_from_codes(df["busing"], true_codes=[2], false_codes=[1])   # oppose busing
    rac3 = binary_from_codes(df["racdif1"], true_codes=[2], false_codes=[1])  # deny discrimination
    rac4 = binary_from_codes(df["racdif3"], true_codes=[2], false_codes=[1])  # deny educational chance
    rac5 = binary_from_codes(df["racdif4"], true_codes=[1], false_codes=[2])  # endorse lack of motivation

    racism_mat = pd.concat([rac1, rac2, rac3, rac4, rac5], axis=1)
    # require all 5 components observed
    df["racism_score"] = racism_mat.sum(axis=1, min_count=5)

    # -----------------------
    # Controls
    # -----------------------
    # Education years (EDUC 0-20)
    if "educ" not in df.columns:
        raise ValueError("Missing column: educ")
    educ = clean_series(df["educ"]).where(clean_series(df["educ"]).between(0, 20))
    df["education_years"] = educ

    # Household income per capita = REALINC / HOMPOP
    for c in ["realinc", "hompop"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    realinc = clean_series(df["realinc"])
    hompop = clean_series(df["hompop"]).where(clean_series(df["hompop"]) > 0)
    df["hh_income_per_capita"] = (realinc / hompop).replace([np.inf, -np.inf], np.nan)

    # Occupational prestige PRESTG80
    if "prestg80" not in df.columns:
        raise ValueError("Missing column: prestg80")
    df["occ_prestige"] = clean_series(df["prestg80"])

    # Female (SEX: 1 male, 2 female)
    if "sex" not in df.columns:
        raise ValueError("Missing column: sex")
    df["female"] = binary_from_codes(df["sex"], true_codes=[2], false_codes=[1])

    # Age
    if "age" not in df.columns:
        raise ValueError("Missing column: age")
    df["age_years"] = clean_series(df["age"]).where(clean_series(df["age"]).between(18, 89))

    # Race indicators (RACE: 1 white, 2 black, 3 other)
    if "race" not in df.columns:
        raise ValueError("Missing column: race")
    race = clean_series(df["race"]).where(clean_series(df["race"]).isin([1, 2, 3]))
    df["black"] = np.where(race.isna(), np.nan, (race == 2).astype(float))
    df["other_race"] = np.where(race.isna(), np.nan, (race == 3).astype(float))

    # Hispanic: per instruction, no direct identifier in provided vars -> cannot construct faithfully.
    # Keep as missing; it will force listwise deletion if included.
    df["hispanic"] = np.nan

    # Conservative Protestant proxy: RELIG==1 and DENOM in {1,6,7}
    for c in ["relig", "denom"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    relig = clean_series(df["relig"])
    denom = clean_series(df["denom"])
    consprot = ((relig == 1) & (denom.isin([1, 6, 7]))).astype(float)
    consprot = pd.Series(consprot, index=df.index, dtype="float64")
    consprot.loc[relig.isna() | denom.isna()] = np.nan
    df["cons_protestant"] = consprot

    # No religion: RELIG==4
    norelig = (relig == 4).astype(float)
    norelig = pd.Series(norelig, index=df.index, dtype="float64")
    norelig.loc[relig.isna()] = np.nan
    df["no_religion"] = norelig

    # Southern: REGION==3
    if "region" not in df.columns:
        raise ValueError("Missing column: region")
    region = clean_series(df["region"]).where(clean_series(df["region"]).isin([1, 2, 3, 4]))
    south = (region == 3).astype(float)
    south = pd.Series(south, index=df.index, dtype="float64")
    south.loc[region.isna()] = np.nan
    df["south"] = south

    # -----------------------
    # Model specs (canonical order matching paper)
    # -----------------------
    # NOTE: Hispanic cannot be implemented with provided variables; including it would collapse N.
    # To keep models runnable and faithful to available data, we omit Hispanic from estimation
    # but still report a placeholder row in the paper-style table.
    x_order_terms = [
        "Racism score",
        "Education",
        "Household income per capita",
        "Occupational prestige",
        "Female",
        "Age",
        "Black",
        "Hispanic",
        "Other race",
        "Conservative Protestant",
        "No religion",
        "Southern",
    ]

    x_map_available = {
        "Racism score": "racism_score",
        "Education": "education_years",
        "Household income per capita": "hh_income_per_capita",
        "Occupational prestige": "occ_prestige",
        "Female": "female",
        "Age": "age_years",
        "Black": "black",
        # Hispanic omitted from model fit due to missing in provided data
        "Other race": "other_race",
        "Conservative Protestant": "cons_protestant",
        "No religion": "no_religion",
        "Southern": "south",
    }

    # -----------------------
    # Fit models (omit Hispanic from estimation; then insert placeholder row for reporting)
    # -----------------------
    def add_hispanic_placeholder(paper_df):
        # Insert Hispanic row in canonical order with blank coefficient
        paper_df = paper_df.copy()
        if "Hispanic" not in paper_df["term"].values:
            # Create placeholder
            placeholder = pd.DataFrame(
                [{"term": "Hispanic", "beta": np.nan, "p_value": np.nan, "stars": "", "beta_str": ""}]
            )
            # Reorder to canonical
            paper_df = pd.concat([paper_df, placeholder], axis=0, ignore_index=True)
        # enforce canonical order + Constant last
        order = x_order_terms + ["Constant"]
        paper_df["__order"] = paper_df["term"].map({t: i for i, t in enumerate(order)})
        paper_df = paper_df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
        return paper_df

    # Fit Model A
    mA, paperA, fullA, fitA = fit_table2_model(
        df,
        "dislike_minority_genres",
        x_map_available,
        "Table2_ModelA_dislike_minority6",
    )
    paperA = add_hispanic_placeholder(paperA)

    # Fit Model B
    mB, paperB, fullB, fitB = fit_table2_model(
        df,
        "dislike_other12_genres",
        x_map_available,
        "Table2_ModelB_dislike_other12",
    )
    paperB = add_hispanic_placeholder(paperB)

    # -----------------------
    # Save outputs
    # -----------------------
    def write_text(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # Human-readable summaries
    write_text("./output/Table2_ModelA_summary.txt", mA.summary().as_text())
    write_text("./output/Table2_ModelB_summary.txt", mB.summary().as_text())

    # Paper-style tables (standardized betas + stars; intercept unstandardized)
    # Note: Stars here are based on replication p-values; Table 2 does not report SEs.
    def format_paper_table(paper_df, title):
        lines = []
        lines.append(title)
        lines.append("Standardized coefficients (beta) for slopes; Constant is unstandardized intercept.")
        lines.append("Stars from replication p-values: * p<.05, ** p<.01, *** p<.001 (two-tailed).")
        lines.append("Note: 'Hispanic' cannot be constructed from provided variables in this extract; shown as blank.")
        lines.append("")
        tmp = paper_df[["term", "beta_str"]].copy()
        # nicer alignment
        width = int(max(tmp["term"].astype(str).map(len).max(), len("Variable")) + 2)
        lines.append(f"{'Variable'.ljust(width)}Beta")
        lines.append(f"{'-'*width}{'-'*10}")
        for _, r in tmp.iterrows():
            lines.append(f"{str(r['term']).ljust(width)}{str(r['beta_str'])}")
        return "\n".join(lines)

    write_text(
        "./output/Table2_ModelA_paper_style.txt",
        format_paper_table(paperA, "Table 2 Model A: DV = Dislike of minority-associated genres (count)"),
    )
    write_text(
        "./output/Table2_ModelB_paper_style.txt",
        format_paper_table(paperB, "Table 2 Model B: DV = Dislike of other 12 genres (count)"),
    )

    # Full replication coefficient tables
    fullA_out = fullA.copy()
    fullA_out.index.name = "term"
    fullB_out = fullB.copy()
    fullB_out.index.name = "term"
    write_text("./output/Table2_ModelA_full_table.txt", fullA_out.to_string(float_format=lambda x: f"{x: .6f}"))
    write_text("./output/Table2_ModelB_full_table.txt", fullB_out.to_string(float_format=lambda x: f"{x: .6f}"))

    # Fit tables
    write_text("./output/Table2_ModelA_fit.txt", fitA.to_string(index=False))
    write_text("./output/Table2_ModelB_fit.txt", fitB.to_string(index=False))

    # Diagnostics to help with the earlier runtime errors (variance checks)
    def diagnostics_for(df_in, model_label, dv, xcols):
        d = df_in[[dv] + xcols].replace([np.inf, -np.inf], np.nan).dropna()
        diag = []
        diag.append(f"{model_label} diagnostics")
        diag.append(f"Analytic n (listwise on DV + included predictors): {d.shape[0]}")
        diag.append("")
        for c in xcols:
            v = d[c]
            diag.append(f"{c}: mean={v.mean():.6f} sd={v.std(ddof=1):.6f} min={v.min():.6f} max={v.max():.6f}")
        return "\n".join(diag)

    included_predictors = list(x_map_available.values())
    write_text(
        "./output/Table2_ModelA_diagnostics.txt",
        diagnostics_for(df, "Model A", "dislike_minority_genres", included_predictors),
    )
    write_text(
        "./output/Table2_ModelB_diagnostics.txt",
        diagnostics_for(df, "Model B", "dislike_other12_genres", included_predictors),
    )

    # Return results as DataFrames
    return {
        "ModelA_paper_style": paperA,
        "ModelB_paper_style": paperB,
        "ModelA_full": fullA_out,
        "ModelB_full": fullB_out,
        "ModelA_fit": fitA,
        "ModelB_fit": fitB,
    }