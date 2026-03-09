def run_analysis(data_source):
    import os
    import re
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore")

    # --- Helpers ---
    def norm_col(s):
        return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

    def find_column(df, candidates, required=True):
        cols = list(df.columns)
        norm_map = {norm_col(c): c for c in cols}

        # direct / normalized exact matches
        for cand in candidates:
            nc = norm_col(cand)
            if nc in norm_map:
                return norm_map[nc]

        # contains match (fallback)
        for cand in candidates:
            nc = norm_col(cand)
            for k, orig in norm_map.items():
                if nc and (nc in k or k in nc):
                    return orig

        if required:
            raise KeyError(f"Could not find required column among candidates: {candidates}. Available: {list(df.columns)}")
        return None

    def to_numeric_series(s):
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        return pd.to_numeric(s, errors="coerce")

    def recode_like_dislike_to_dislike_indicator(series):
        """
        Returns: dislike (1/0) with NaN for DK/NA/invalid.
        Expected original scale: 1..5 (Like very much ... Dislike very much)
        Treat DK/NA as missing.
        """
        x = series.copy()

        # If numeric, use numeric rules
        x_num = to_numeric_series(x)
        if x_num.notna().sum() > 0:
            # Common GSS coding: 1 like very much, 2 like it, 3 mixed, 4 dislike, 5 dislike very much
            # Treat out-of-range and nonpositive as missing
            dislike = pd.Series(np.nan, index=x_num.index, dtype=float)
            valid = x_num.between(1, 5)
            dislike.loc[valid] = (x_num.loc[valid] >= 4).astype(float)
            return dislike

        # If string categories, parse (fallback)
        x_str = x.astype(str).str.strip().str.lower()
        x_str = x_str.replace({"": np.nan, "nan": np.nan, "none": np.nan})
        # Mark DK/NA as missing
        dk_na = x_str.str.contains(r"don.?t know|dont know|dk|no answer|refus|na\b|n/a", regex=True, na=False)
        out = pd.Series(np.nan, index=x_str.index, dtype=float)
        # dislike conditions
        dis = x_str.str.contains(r"dislike", regex=True, na=False) & ~x_str.str.contains(r"don.?t know|dont know", regex=True, na=False)
        like_or_mixed = x_str.str.contains(r"like|mixed", regex=True, na=False)
        out.loc[dk_na] = np.nan
        out.loc[dis & ~dk_na] = 1.0
        out.loc[like_or_mixed & ~dk_na] = 0.0
        return out

    def fit_logit_irls(X, y, max_iter=100, tol=1e-8):
        """
        IRLS for logistic regression.
        X: (n,k) numpy array with intercept included.
        y: (n,) binary {0,1}
        Returns: beta (k,), converged (bool)
        """
        n, k = X.shape
        beta = np.zeros(k, dtype=float)

        for _ in range(max_iter):
            eta = X @ beta
            # stable sigmoid
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
            w = p * (1 - p)
            # Avoid zeros
            w = np.clip(w, 1e-8, None)
            z = eta + (y - p) / w

            # Weighted least squares: beta_new = (X'WX)^-1 X'W z
            WX = X * w[:, None]
            XtWX = X.T @ WX
            XtWz = X.T @ (w * z)

            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.pinv(XtWX) @ XtWz

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                return beta, True
            beta = beta_new

        return beta, False

    # --- Read data ---
    df = pd.read_csv(data_source)

    # --- Identify columns ---
    year_col = find_column(df, ["YEAR", "year"])
    educ_col = find_column(df, ["EDUC", "educ", "education", "years_education", "yrs_educ"])

    # Genre columns: use provided canonical names with robust fallback
    genre_candidates = {
        "Latin/Salsa": ["LATIN", "latin", "musiclat", "musiclatin", "latinmariachisalsa"],
        "Jazz": ["JAZZ", "jazz", "musicjazz"],
        "Blues/R&B": ["BLUES", "blues", "musicblu", "musicblues", "rhythmandblues", "rnb"],
        "Show Tunes": ["MUSICALS", "musicals", "showtunes", "musicmus", "musicmusicals"],
        "Oldies": ["OLDIES", "oldies", "musicold", "musicoldies"],
        "Classical/Chamber": ["CLASSICL", "classicl", "classical", "musiccla", "musicclassical"],
        "Swing/Big Band": ["BIGBAND", "bigband", "musicbig", "swing"],
        "New Age/Space": ["NEWAGE", "newage", "musicnew", "newage_space"],
        "Opera": ["OPERA", "opera", "musicopr", "musicopera"],
        "Bluegrass": ["BLUGRASS", "blugrass", "musicblg", "musicbluegrass"],
        "Folk": ["FOLK", "folk", "musicfol", "musicfolk"],
        "Reggae": ["REGGAE", "reggae", "musicreg", "musicreggae"],
        "Easy Listening": ["MOODEASY", "moodeasy", "easylis", "easylistening", "musicezl", "mood"],
        "Pop/Contemporary Rock": ["CONROCK", "conrock", "contemporaryrock", "poprock", "musicpop", "musicrok"],
        "Rap": ["RAP", "rap", "musicrap"],
        "Heavy Metal": ["HVYMETAL", "hvymetal", "heavymetal", "musicmet", "metal"],
        "Country/Western": ["COUNTRY", "country", "countrywestern", "musiccnt", "musiccountry"],
        "Gospel": ["GOSPEL", "gospel", "musicgos", "musicgospel"],
    }

    genre_cols = {}
    for label, cands in genre_candidates.items():
        genre_cols[label] = find_column(df, cands, required=True)

    # --- Filter year (1993) and clean education ---
    df = df.copy()
    df[year_col] = to_numeric_series(df[year_col])
    df = df.loc[df[year_col] == 1993].copy()

    df[educ_col] = to_numeric_series(df[educ_col])
    df.loc[~df[educ_col].between(0, 25), educ_col] = np.nan  # conservative bounds
    sample_mean_educ = float(df[educ_col].mean(skipna=True))

    # --- Construct dislike indicators for each genre ---
    dislike = {}
    for g, col in genre_cols.items():
        dislike[g] = recode_like_dislike_to_dislike_indicator(df[col])

    # --- Construct tolerance counts excluding dependent genre, and run 18 logits ---
    results = []
    for g in genre_cols.keys():
        # build tolerance T_-g: number of other genres NOT disliked (0), among nonmissing for those genres
        other_genres = [h for h in genre_cols.keys() if h != g]
        other_dislike_df = pd.DataFrame({h: dislike[h] for h in other_genres})

        # Not disliked indicator (1 if 0, 0 if 1, NaN if missing)
        not_disliked = (other_dislike_df == 0).astype(float)
        not_disliked = not_disliked.where(other_dislike_df.notna(), np.nan)

        T_minus_g = not_disliked.sum(axis=1, min_count=1)

        y = dislike[g]
        educ = df[educ_col]

        model_df = pd.DataFrame({
            "y": y,
            "T": T_minus_g,
            "educ": educ
        })

        # CRITICAL: drop missing before fitting
        model_df = model_df.dropna(subset=["y", "T", "educ"]).copy()

        # Ensure y binary 0/1
        model_df = model_df.loc[model_df["y"].isin([0.0, 1.0])].copy()

        # If insufficient variation, skip (but should not happen)
        if model_df.shape[0] < 50 or model_df["y"].nunique() < 2:
            beta_T = np.nan
        else:
            X = np.column_stack([
                np.ones(model_df.shape[0]),
                model_df["T"].to_numpy(dtype=float),
                model_df["educ"].to_numpy(dtype=float)
            ])
            yv = model_df["y"].to_numpy(dtype=float)
            beta, _ = fit_logit_irls(X, yv)
            beta_T = float(beta[1])

        # Compute mean education of genre audience (like very much / like it => 1 or 2)
        x_num = to_numeric_series(df[genre_cols[g]])
        like_mask = x_num.isin([1, 2])
        mean_educ_aud = float(df.loc[like_mask, educ_col].mean(skipna=True))

        results.append({
            "genre": g,
            "beta_tolerance": beta_T,
            "mean_educ_audience": mean_educ_aud
        })

    res = pd.DataFrame(results)

    # --- Order genres by tolerance coefficient (beta) ---
    # More negative => more rejected by tolerant respondents (right side)
    res = res.sort_values("beta_tolerance", ascending=False).reset_index(drop=True)

    # --- Plot: dual y-axis with connected lines ---
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260130_222138/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(res.shape[0])
    genres_ordered = res["genre"].tolist()

    fig, ax1 = plt.subplots(figsize=(13.5, 6.8))

    # Left axis: tolerance coefficients
    ax1.plot(
        x, res["beta_tolerance"].to_numpy(dtype=float),
        linestyle=(0, (2, 2)),
        color="black",
        linewidth=1.8
    )
    ax1.set_ylabel("Coefficients for Musical Tolerance in Affecting Odds (Probability) of Disliking Each Individual Genre")
    ax1.set_xlim(-0.5, len(x) - 0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(genres_ordered, rotation=45, ha="right")
    ax1.axhline(0, color="black", linewidth=0.6, alpha=0.6)

    # Right axis: mean education of genre audience
    ax2 = ax1.twinx()
    ax2.plot(
        x, res["mean_educ_audience"].to_numpy(dtype=float),
        linestyle="-",
        color="black",
        linewidth=2.0
    )
    ax2.set_ylabel("Mean Educational Level of Respondents Who Reported Liking Each Audience")

    # Sample mean education reference line (right axis)
    if np.isfinite(sample_mean_educ):
        ax2.axhline(sample_mean_educ, color="black", linestyle=":", linewidth=1.2)
        # label near right edge
        ax2.text(
            len(x) - 0.6, sample_mean_educ,
            "Sample Mean Education",
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8)
        )

    # Title
    fig.suptitle(
        "Figure 1. The Effect of Being Musically Tolerant on Disliking Each Music Genre Compared to the Educational Composition of Genre Audiences",
        y=0.98
    )

    # In-plot annotations (approximate placement)
    # Mean education line annotation
    try:
        idx_me = int(np.nanargmax(res["mean_educ_audience"].to_numpy(dtype=float)))
    except Exception:
        idx_me = max(0, len(x) // 3)
    ax2.annotate(
        "Mean Education of Genre Audience",
        xy=(idx_me, res.loc[idx_me, "mean_educ_audience"]),
        xytext=(max(-0.2, idx_me - 5), float(res["mean_educ_audience"].max()) if np.isfinite(res["mean_educ_audience"].max()) else 14.0),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.0),
        fontsize=9,
        ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8)
    )

    # Coefficient line annotation
    try:
        idx_bt = int(np.nanargmin(res["beta_tolerance"].to_numpy(dtype=float)))
    except Exception:
        idx_bt = max(0, (2 * len(x)) // 3)
    ax1.annotate(
        "Coefficient for Musical Tolerance",
        xy=(idx_bt, res.loc[idx_bt, "beta_tolerance"]),
        xytext=(max(-0.2, idx_bt - 6), float(res["beta_tolerance"].min()) + 0.08 if np.isfinite(res["beta_tolerance"].min()) else -0.3),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.0),
        fontsize=9,
        ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8)
    )

    # Layout
    ax1.grid(False)
    ax2.grid(False)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)