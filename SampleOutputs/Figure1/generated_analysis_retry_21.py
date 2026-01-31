def run_analysis(data_source):
    import os
    import re
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore")

    # -----------------------------
    # Helpers
    # -----------------------------
    def norm_col(s):
        return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

    def find_column(df, candidates, required=True):
        cols = list(df.columns)
        norm_map = {norm_col(c): c for c in cols}

        # exact normalized match
        for cand in candidates:
            nc = norm_col(cand)
            if nc in norm_map:
                return norm_map[nc]

        # contains match fallback
        for cand in candidates:
            nc = norm_col(cand)
            if not nc:
                continue
            for k, orig in norm_map.items():
                if nc in k or k in nc:
                    return orig

        if required:
            raise KeyError(
                f"Could not find required column among candidates={candidates}. "
                f"Available columns={list(df.columns)}"
            )
        return None

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def recode_music_item(series):
        """
        Return numeric 1..5 if possible; DK/NA -> NaN.
        Expected coding:
          1 Like very much
          2 Like it
          3 Mixed feelings
          4 Dislike
          5 Dislike very much
        """
        x = to_num(series)
        if x.notna().any():
            return x.where(x.between(1, 5), np.nan)

        s = series.astype(str).str.strip().str.lower()
        s = s.str.replace(r"\s+", " ", regex=True)
        s = s.str.replace("dont", "don't", regex=False).str.replace("do not", "don't", regex=False)

        mapping = {
            "like very much": 1,
            "like it": 2,
            "mixed feelings": 3,
            "dislike": 4,
            "dislike it": 4,
            "dislike very much": 5,
            "don't know much about it": np.nan,
            "dont know much about it": np.nan,
            "don't know much": np.nan,
            "dont know much": np.nan,
            "don't know": np.nan,
            "dont know": np.nan,
            "dk": np.nan,
            "no answer": np.nan,
            "na": np.nan,
            "n/a": np.nan,
            "refused": np.nan,
        }
        x2 = s.map(mapping)
        x2 = to_num(x2).where(to_num(x2).between(1, 5), np.nan)
        return x2

    def dislike_indicator(x_1to5):
        # 1 if 4/5, 0 if 1/2/3, NaN otherwise
        out = pd.Series(np.nan, index=x_1to5.index, dtype=float)
        valid = x_1to5.between(1, 5)
        out.loc[valid] = (x_1to5.loc[valid] >= 4).astype(float)
        return out

    def like_audience_mask(x_1to5):
        # Audience = Like very much (1) or Like it (2)
        return x_1to5.isin([1, 2])

    def fit_logit_irls(X, y, max_iter=200, tol=1e-10):
        """
        Logistic regression via IRLS / Newton-Raphson.
        X includes intercept.
        Returns beta vector.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        beta = np.zeros(X.shape[1], dtype=float)

        for _ in range(max_iter):
            eta = X @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
            w = p * (1.0 - p)
            w = np.clip(w, 1e-9, None)

            z = eta + (y - p) / w

            WX = X * w[:, None]
            XtWX = X.T @ WX
            XtWz = X.T @ (w * z)

            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.pinv(XtWX) @ XtWz

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        return beta

    # -----------------------------
    # Read data
    # -----------------------------
    df = pd.read_csv(data_source)

    year_col = find_column(df, ["YEAR", "year"], required=True)
    educ_col = find_column(df, ["EDUC", "educ", "education", "years_education", "yrs_educ"], required=True)

    # Canonical 1993 GSS music battery (with fallbacks)
    genre_col_candidates = {
        "Latin/Salsa": ["LATIN", "MUSICLAT", "MUSICLATIN", "musiclat", "musiclatin", "latin", "salsa"],
        "Jazz": ["JAZZ", "MUSICJAZ", "MUSICJAZZ", "musicjaz", "musicjazz", "jazz"],
        "Blues/R&B": ["BLUES", "MUSICBLU", "MUSICBLUES", "musicblu", "musicblues", "blues", "rnb", "rhythmandblues", "rhythmblues"],
        "Show Tunes": ["MUSICALS", "MUSICMUS", "MUSICMUSICALS", "musicmus", "musicmusicals", "musicals", "showtunes", "showtune"],
        "Oldies": ["OLDIES", "MUSICOLD", "MUSICOLDIES", "musicold", "musicoldies", "oldies"],
        "Classical": ["CLASSICL", "MUSICCLA", "MUSICCLASSICAL", "classicl", "musiccla", "musicclassical", "classical"],
        "Swing": ["BIGBAND", "MUSICBIG", "MUSICBIGBAND", "bigband", "musicbig", "musicbigband", "swing"],
        "New Age/Space": ["NEWAGE", "MUSICNEW", "MUSICNEWAGE", "newage", "musicnew", "musicnewage", "space"],
        "Opera": ["OPERA", "MUSICOPR", "MUSICOPERA", "musicopr", "musicopera", "opera"],
        "Bluegrass": ["BLUGRASS", "MUSICBLG", "MUSICBLUEGRASS", "blugrass", "musicblg", "musicbluegrass", "bluegrass"],
        "Folk": ["FOLK", "MUSICFOL", "MUSICFOLK", "musicfol", "musicfolk", "folk"],
        "Reggae": ["REGGAE", "MUSICREG", "MUSICREGGAE", "musicreg", "musicreggae", "reggae"],
        "Easy Listening": ["MOODEASY", "MUSICEZL", "musicezl", "moodeasy", "easylistening", "mood"],
        "Pop/Rock": ["CONROCK", "MUSICPOP", "MUSICROK", "MUSICROCK", "conrock", "musicpop", "musicrok", "musicrock", "poprock", "contemporaryrock"],
        "Rap": ["RAP", "MUSICRAP", "musicrap", "rap"],
        "Heavy Metal": ["HVYMETAL", "MUSICMET", "hvymetal", "musicmet", "heavymetal", "metal"],
        "Country": ["COUNTRY", "MUSICCNT", "MUSICCOUNTRY", "countrywestern", "musiccnt", "musiccountry", "country"],
        "Gospel": ["GOSPEL", "MUSICGOS", "MUSICGOSPEL", "musicgos", "musicgospel", "gospel"],
    }
    genre_cols = {g: find_column(df, cands, required=True) for g, cands in genre_col_candidates.items()}

    # -----------------------------
    # Filter to 1993 + clean education
    # -----------------------------
    df = df.copy()
    df[year_col] = to_num(df[year_col])
    df = df.loc[df[year_col] == 1993].copy()

    df[educ_col] = to_num(df[educ_col])
    df.loc[~df[educ_col].between(0, 25), educ_col] = np.nan

    sample_mean_educ = float(df[educ_col].mean(skipna=True)) if df[educ_col].notna().any() else np.nan

    # -----------------------------
    # Recode all music items to 1..5, then dislike indicators
    # -----------------------------
    music_1to5 = {g: recode_music_item(df[col]) for g, col in genre_cols.items()}
    dislike = {g: dislike_indicator(music_1to5[g]) for g in genre_cols.keys()}

    # -----------------------------
    # Compute per-genre: tolerance coefficient + mean education among likers
    # -----------------------------
    genre_list = list(genre_cols.keys())
    rows = []

    for g in genre_list:
        others = [h for h in genre_list if h != g]
        other_dislike = pd.DataFrame({h: dislike[h] for h in others})

        # T_-g: number of OTHER genres NOT disliked; require complete info across those 17
        complete_other = other_dislike.notna().all(axis=1)
        not_disliked = (other_dislike == 0.0).astype(float).where(other_dislike.notna(), np.nan)
        T_minus_g = not_disliked.sum(axis=1).where(complete_other, np.nan)

        y = dislike[g]
        educ = df[educ_col]

        model_df = pd.DataFrame({"y": y, "T": T_minus_g, "educ": educ}).dropna()
        model_df = model_df.loc[model_df["y"].isin([0.0, 1.0])].copy()

        beta_T = np.nan
        if (
            model_df.shape[0] >= 50
            and model_df["y"].nunique() == 2
            and model_df["T"].nunique() >= 2
            and model_df["educ"].nunique() >= 2
        ):
            X = np.column_stack(
                [
                    np.ones(model_df.shape[0], dtype=float),
                    model_df["T"].to_numpy(dtype=float),
                    model_df["educ"].to_numpy(dtype=float),
                ]
            )
            yv = model_df["y"].to_numpy(dtype=float)
            beta = fit_logit_irls(X, yv)
            beta_T = float(beta[1])

        like_mask = like_audience_mask(music_1to5[g])
        mean_edu = float(df.loc[like_mask, educ_col].mean(skipna=True)) if like_mask.any() else np.nan

        rows.append({"genre": g, "coef_tolerance": beta_T, "mean_edu": mean_edu})

    res = pd.DataFrame(rows).drop_duplicates(subset=["genre"], keep="first").copy()

    # -----------------------------
    # Enforce EXACT Figure 1 x-axis order (as in the paper)
    # -----------------------------
    genre_order = [
        "Latin/Salsa",
        "Jazz",
        "Blues/R&B",
        "Show Tunes",
        "Oldies",
        "Classical",
        "Swing",
        "New Age/Space",
        "Opera",
        "Bluegrass",
        "Folk",
        "Reggae",
        "Easy Listening",
        "Pop/Rock",
        "Rap",
        "Heavy Metal",
        "Country",
        "Gospel",
    ]

    # Align strictly by genre (prevents accidental mismatching)
    res = res.set_index("genre").reindex(genre_order).reset_index()

    coef = res["coef_tolerance"].to_numpy(dtype=float)
    edu = res["mean_edu"].to_numpy(dtype=float)

    # -----------------------------
    # Sanity checks (avoid swapped series / wrong alignment)
    # -----------------------------
    if len(coef) != len(genre_order) or len(edu) != len(genre_order):
        raise RuntimeError("Series length mismatch after reindexing to paper genre order.")

    # Education should be in plausible range for years of schooling (if present)
    edu_finite = edu[np.isfinite(edu)]
    if edu_finite.size > 0:
        if edu_finite.min() < 8 or edu_finite.max() > 22:
            raise RuntimeError("Mean education series out of plausible bounds; check EDUC coding/input.")

    # Coefs should be negative in the paper; allow some slack but detect obvious swap
    coef_finite = coef[np.isfinite(coef)]
    if coef_finite.size > 0:
        if coef_finite.max() > 1 or coef_finite.min() < -5:
            raise RuntimeError("Tolerance coefficient series out of plausible bounds; model may have failed.")
        # If coefficients are mostly positive, likely a construction error; do not hard-fail, but warn via exception
        if np.nanmean(coef_finite) > 0.05:
            raise RuntimeError("Tolerance coefficients are mostly positive; check dislike coding/tolerance construction.")

    # -----------------------------
    # Plot (matplotlib only)
    # -----------------------------
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260130_222138/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(genre_order), dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=200)
    ax2 = ax.twinx()

    coef_lw = 2.0
    edu_lw = 2.0

    # LEFT axis: tolerance coefficient (SOLID)
    ax.plot(x, coef, color="black", linewidth=coef_lw, linestyle="-", marker=None, zorder=3)

    # RIGHT axis: mean education (DASH-DOT)
    ax2.plot(x, edu, color="black", linewidth=edu_lw, linestyle="-.", marker=None, zorder=2)

    # Sample mean education reference line (RIGHT axis), thin dotted
    if np.isfinite(sample_mean_educ):
        ax2.axhline(sample_mean_educ, color="black", linewidth=1.0, linestyle=(0, (1, 2)), zorder=1)

    # X axis
    ax.set_xticks(x)
    ax.set_xticklabels(genre_order, rotation=55, ha="right")
    ax.set_xlabel("Type of Music", fontweight="bold")

    # Left y-axis (coefficients)
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax.set_yticklabels(["-.5", "-.4", "-.3", "-.2", "-.1"])
    ax.set_ylabel(
        "Coefficients for Musical Tolerance in Affecting Odds (Probability)\n"
        "of Disliking Each Individual Genre",
        fontweight="bold",
    )

    # Right y-axis (education)
    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.set_ylabel(
        "Mean Educational Level of Respondents Who\nReported Liking Each Music Genre",
        fontweight="bold",
        rotation=270,
        labelpad=30,
    )

    # Title/caption
    ax.set_title(
        "Figure 1. The Effect of Being Musically Tolerant on Disliking Each Music Genre\n"
        "Compared to the Educational Composition of Genre Audiences",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # Style
    ax.grid(False)
    ax2.grid(False)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    for a in (ax, ax2):
        a.tick_params(axis="both", which="major", width=2.0, length=9, color="black", labelsize=11)

    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax2.spines["right"].set_linewidth(2.0)

    # -----------------------------
    # Annotations (point to correct axes/series)
    # -----------------------------
    # Education label on ax2 (dash-dot series)
    idx_edu = genre_order.index("Blues/R&B") if "Blues/R&B" in genre_order else 2
    if 0 <= idx_edu < len(edu) and np.isfinite(edu[idx_edu]):
        ax2.annotate(
            "Mean Education of Genre Audience",
            xy=(idx_edu, float(edu[idx_edu])),
            xycoords=("data", "data"),
            xytext=(max(idx_edu - 2.4, 0.0), 14.25),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

    # Coefficient label on ax (solid series)
    idx_coef = genre_order.index("Jazz") if "Jazz" in genre_order else 1
    if 0 <= idx_coef < len(coef) and np.isfinite(coef[idx_coef]):
        ax.annotate(
            "Coefficient for Musical Tolerance",
            xy=(idx_coef, float(coef[idx_coef])),
            xycoords=("data", "data"),
            xytext=(min(idx_coef + 2.0, len(genre_order) - 1), -0.47),
            textcoords=("data", "data"),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

    # Sample mean education label on ax2 (dotted reference line)
    if np.isfinite(sample_mean_educ):
        idx_anchor = genre_order.index("Reggae") if "Reggae" in genre_order else 11
        ax2.annotate(
            "Sample Mean Education",
            xy=(idx_anchor, sample_mean_educ),
            xycoords=("data", "data"),
            xytext=(idx_anchor, sample_mean_educ - 0.35),
            textcoords=("data", "data"),
            ha="center",
            va="top",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, shrinkA=0, shrinkB=0),
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)