import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_analysis(data_source: str, sep: Optional[str] = None, na_values: Optional[Union[List[Any], Any]] = None):
    # Helper functions
    def to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def dislike_indicator(s: pd.Series) -> pd.Series:
        # 1=like very much; 2=like it; 3=mixed feelings; 4=dislike it; 5=dislike very much
        s_num = to_num(s)
        out = pd.Series(np.nan, index=s_num.index)
        out.loc[s_num.isin([4, 5])] = 1.0
        out.loc[s_num.isin([1, 2, 3])] = 0.0
        return out

    def star(p: float) -> str:
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def standardize_betas(coefs: np.ndarray, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        sd_y = y.std(ddof=1)
        sds_x = X.apply(lambda col: col.std(ddof=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            betas = coefs * (sds_x.values / sd_y if sd_y not in [0, np.nan] else np.nan)
        return betas

    def build_racism(df: pd.DataFrame) -> pd.Series:
        # Items and codings:
        # busing: 2=oppose ->1; 1=favor->0; else NA
        # racdif1: 2=no ->1; 1=yes->0
        # racdif3: 2=no ->1; 1=yes->0
        # racdif4: 1=yes->1; 2=no->0
        items = {}
        if "busing" in df.columns:
            s = to_num(df["busing"])
            x = pd.Series(np.nan, index=df.index, dtype=float)
            x.loc[s == 2] = 1.0
            x.loc[s == 1] = 0.0
            items["busing"] = x
        if "racdif1" in df.columns:
            s = to_num(df["racdif1"])
            x = pd.Series(np.nan, index=df.index, dtype=float)
            x.loc[s == 2] = 1.0
            x.loc[s == 1] = 0.0
            items["racdif1"] = x
        if "racdif3" in df.columns:
            s = to_num(df["racdif3"])
            x = pd.Series(np.nan, index=df.index, dtype=float)
            x.loc[s == 2] = 1.0
            x.loc[s == 1] = 0.0
            items["racdif3"] = x
        if "racdif4" in df.columns:
            s = to_num(df["racdif4"])
            x = pd.Series(np.nan, index=df.index, dtype=float)
            x.loc[s == 1] = 1.0
            x.loc[s == 2] = 0.0
            items["racdif4"] = x
        if not items:
            return pd.Series(np.nan, index=df.index)
        item_df = pd.DataFrame(items)
        # Sum available (0/1), missing if all four missing
        score = item_df.sum(axis=1, min_count=1)
        return score

    def build_income_pc(df: pd.DataFrame) -> pd.Series:
        if ("realinc" not in df.columns) or ("adults" not in df.columns):
            return pd.Series(np.nan, index=df.index)
        inc = to_num(df["realinc"])
        adults = to_num(df["adults"])
        income_pc = inc / adults.replace({0: np.nan})
        income_pc[(inc.isna()) | (adults.isna())] = np.nan
        return income_pc

    def build_female(df: pd.DataFrame) -> pd.Series:
        if "sex" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        s = to_num(df["sex"])
        out = pd.Series(np.nan, index=df.index, dtype=float)
        out.loc[s == 2] = 1.0
        out.loc[s == 1] = 0.0
        return out

    def build_race_dummies(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        if "race" not in df.columns:
            return (pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index))
        r = to_num(df["race"])
        black = pd.Series(np.nan, index=df.index, dtype=float)
        other = pd.Series(np.nan, index=df.index, dtype=float)
        black.loc[r == 2] = 1.0
        black.loc[r.isin([1, 3])] = 0.0
        other.loc[r == 3] = 1.0
        other.loc[r.isin([1, 2])] = 0.0
        return black, other

    def build_hispanic(df: pd.DataFrame) -> Optional[pd.Series]:
        # Try to detect Hispanic origin if present
        cand_cols = [c for c in df.columns if c.lower() in ["hispanic", "hispan", "hispan1", "hispan2", "hisp_orig"]]
        if not cand_cols:
            return None
        s = to_num(df[cand_cols[0]])
        out = pd.Series(np.nan, index=df.index, dtype=float)
        # Common GSS coding: 1=non-Hispanic; 2/3/4=Hispanic subtypes
        if s.dropna().isin([0, 1, 2, 3, 4]).all():
            out.loc[s.notna() & (s != 1)] = 1.0
            out.loc[s == 1] = 0.0
        else:
            # Fallback: treat nonzero as Hispanic
            out.loc[s.notna() & (s != 0)] = 1.0
            out.loc[s == 0] = 0.0
        return out

    def build_norelig(df: pd.DataFrame) -> pd.Series:
        if "relig" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        r = to_num(df["relig"])
        out = pd.Series(np.nan, index=df.index, dtype=float)
        out.loc[r == 4] = 1.0
        out.loc[r.isin([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13])] = 0.0
        return out

    def build_south(df: pd.DataFrame) -> pd.Series:
        if "region" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        reg = to_num(df["region"])
        out = pd.Series(np.nan, index=df.index, dtype=float)
        out.loc[reg == 3] = 1.0
        out.loc[reg.isin([1, 2, 4])] = 0.0
        return out

    def build_cons_prot(df: pd.DataFrame) -> Optional[pd.Series]:
        # Try existing column(s)
        for c in df.columns:
            if c.lower() in ["cons_prot", "conservative_protestant", "evangelical", "evangelical_protestant", "reltrad"]:
                s = to_num(df[c])
                if c.lower() == "reltrad":
                    # Common reltrad coding puts evangelical as a unique category; assume code 2 or similar
                    # Without a codebook we cannot map reliably; skip unless binary present
                    continue
                # If binary already:
                vals = s.dropna().unique()
                if set(vals).issubset({0, 1}):
                    out = pd.Series(np.nan, index=df.index, dtype=float)
                    out.loc[s == 1] = 1.0
                    out.loc[s == 0] = 0.0
                    return out
        # Try FUND if present (Protestant fundamentalism)
        if "relig" in df.columns and "denom" in df.columns and "fund" in df.columns:
            relig = to_num(df["relig"])
            fund = to_num(df["fund"])
            out = pd.Series(np.nan, index=df.index, dtype=float)
            # cons_prot = Protestant & FUND=fundamentalist
            out.loc[(relig == 1) & (fund == 1)] = 1.0
            out.loc[(relig == 1) & (fund.isin([2, 3]))] = 0.0
            # Non-Protestants set to 0
            out.loc[relig.isin([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])] = 0.0
            return out
        # Without necessary fields, cannot construct; return None
        return None

    def fit_model(df: pd.DataFrame, dv: str, predictors: List[str], term_labels: Dict[str, str]):
        # Drop rows with missing DV and missing predictors (listwise deletion)
        model_df = df[[dv] + predictors].copy()
        model_df = model_df.dropna(axis=0, how="any")

        if model_df.empty or model_df.shape[0] < len(predictors) + 2:
            raise ValueError("Not enough complete cases to fit the model.")

        y = model_df[dv].astype(float)
        X = model_df[predictors].astype(float)
        X = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, X).fit()

        # Build unstandardized table aligned to our predictor order
        terms = predictors
        coefs = np.array([model.params.get(t, np.nan) for t in terms], dtype=float)
        ses = np.array([model.bse.get(t, np.nan) for t in terms], dtype=float)
        ts = np.array([model.tvalues.get(t, np.nan) for t in terms], dtype=float)
        ps = np.array([model.pvalues.get(t, np.nan) for t in terms], dtype=float)

        unstd_table = pd.DataFrame(
            {
                "term": terms,
                "term_label": [term_labels.get(t, t) for t in terms],
                "coef_unstd": coefs,
                "se_unstd": ses,
                "t": ts,
                "p": ps,
            }
        )

        # Standardized betas from the same sample and in the same order
        betas = standardize_betas(coefs, model_df[terms], y)
        std_table = pd.DataFrame(
            {
                "term": terms,
                "term_label": [term_labels.get(t, t) for t in terms],
                "beta_std": betas,
                "stars": [star(p) for p in ps],
            }
        )

        # Model info
        intercept = float(model.params.get("const", np.nan))
        intercept_se = float(model.bse.get("const", np.nan)) if "const" in model.bse.index else np.nan
        intercept_p = float(model.pvalues.get("const", np.nan)) if "const" in model.pvalues.index else np.nan

        info = {
            "constant_unstd": intercept,
            "constant_se": intercept_se,
            "constant_p": intercept_p,
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "n": int(model.nobs),
            "dv_mean": float(y.mean()),
            "dv_sd": float(y.std(ddof=1)),
            "predictors_included": [term_labels.get(t, t) for t in terms],
        }

        return std_table, unstd_table, info, model_df

    # Read data
    if sep is None:
        sep = ","
    if na_values is None:
        na_values = ["", " ", "NA", "NaN", "nan", "None"]
    df = pd.read_csv(data_source, sep=sep, na_values=na_values)

    # Prepare music items
    minority_genres = ["rap", "reggae", "blues", "jazz", "gospel", "latin"]
    remaining_genres = [
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

    # Verify required columns exist
    missing_music_A = [c for c in minority_genres if c not in df.columns]
    missing_music_B = [c for c in remaining_genres if c not in df.columns]
    if missing_music_A or missing_music_B:
        raise ValueError(f"Missing music item columns: {missing_music_A + missing_music_B}")

    # Build dislike indicators for all 18 items
    for col in minority_genres + remaining_genres:
        df[f"{col}_dis"] = dislike_indicator(df[col])

    # Build DVs: require complete data on the respective genre sets
    df["count_minority_dislikes"] = df[[f"{c}_dis" for c in minority_genres]].sum(axis=1, min_count=1)
    df["count_remaining_dislikes"] = df[[f"{c}_dis" for c in remaining_genres]].sum(axis=1, min_count=1)

    # Enforce complete responses on genre items per model (listwise on DV components)
    mask_A_items = df[[f"{c}_dis" for c in minority_genres]].notna().all(axis=1)
    mask_B_items = df[[f"{c}_dis" for c in remaining_genres]].notna().all(axis=1)

    # Build covariates
    df["racism_score"] = build_racism(df)
    df["educ"] = to_num(df["educ"]) if "educ" in df.columns else np.nan
    df["income_pc"] = build_income_pc(df)
    df["prestg80"] = to_num(df["prestg80"]) if "prestg80" in df.columns else np.nan
    df["female"] = build_female(df)
    df["age"] = to_num(df["age"]) if "age" in df.columns else np.nan
    black, other = build_race_dummies(df)
    df["black"] = black
    df["other_race"] = other
    hispanic_series = build_hispanic(df)
    if hispanic_series is not None:
        df["hispanic"] = hispanic_series
    cons_prot_series = build_cons_prot(df)
    if cons_prot_series is not None:
        df["cons_prot"] = cons_prot_series
    df["norelig"] = build_norelig(df)
    df["south"] = build_south(df)

    # Define predictor order and labels
    predictors_order = [
        "racism_score",
        "educ",
        "income_pc",
        "prestg80",
        "female",
        "age",
        "black",
        "hispanic",      # include if present
        "other_race",
        "cons_prot",     # include if present
        "norelig",
        "south",
    ]
    term_labels = {
        "racism_score": "Racism score",
        "educ": "Education (years)",
        "income_pc": "Household income per capita",
        "prestg80": "Occupational prestige",
        "female": "Female",
        "age": "Age",
        "black": "Black",
        "hispanic": "Hispanic",
        "other_race": "Other race",
        "cons_prot": "Conservative Protestant",
        "norelig": "No religion",
        "south": "Southern",
    }

    # Only include predictors present in the DataFrame
    available_predictors = [p for p in predictors_order if p in df.columns]

    # Assemble results
    results = {}

    # Model 1 (minority-associated genres: 6)
    predictors_A = available_predictors.copy()
    # DV-specific complete items mask
    df_A = df.loc[mask_A_items].copy()
    dv_A = "count_minority_dislikes"

    # Fit Model 1
    std_A, unstd_A, info_A, model_df_A = fit_model(df_A, dv_A, predictors_A, term_labels)

    # Model 2 (remaining 12 genres)
    predictors_B = available_predictors.copy()
    df_B = df.loc[mask_B_items].copy()
    dv_B = "count_remaining_dislikes"

    # Fit Model 2
    std_B, unstd_B, info_B, model_df_B = fit_model(df_B, dv_B, predictors_B, term_labels)

    # Attach model labels and DV labels
    results["Model 1"] = {
        "dv": dv_A,
        "dv_label": "Dislike count: Rap, Reggae, Blues/R&B, Jazz, Gospel, Latin",
        "std_table": std_A.reset_index(drop=True),
        "unstd_table": unstd_A.reset_index(drop=True),
        "model_info": info_A,
    }
    results["Model 2"] = {
        "dv": dv_B,
        "dv_label": "Dislike count: remaining 12 genres",
        "std_table": std_B.reset_index(drop=True),
        "unstd_table": unstd_B.reset_index(drop=True),
        "model_info": info_B,
    }

    # Notes about missing covariates that affect exact replication
    missing_covs = []
    if "hispanic" not in predictors_A or "hispanic" not in predictors_B:
        missing_covs.append("Hispanic")
    if "cons_prot" not in predictors_A or "cons_prot" not in predictors_B:
        missing_covs.append("Conservative Protestant")
    if missing_covs:
        results["notes"] = {
            "omitted_predictors_unavailable": missing_covs,
            "message": "Some predictors required by the published table are not available in the input data and were omitted. Results will differ from the published table.",
        }

    return results