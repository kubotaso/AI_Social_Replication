def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import statsmodels.api as sm

    # ----------------------------
    # Helpers
    # ----------------------------
    def find_col(df, preferred_names, contains_any=None):
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}

        for name in preferred_names:
            if name is None:
                continue
            key = str(name).strip().lower()
            if key in lower_map:
                return lower_map[key]

        if contains_any:
            tokens = [str(t).strip().lower() for t in contains_any if t is not None and str(t).strip() != ""]
            hits = []
            for c in cols:
                cl = c.lower()
                if any(tok in cl for tok in tokens):
                    hits.append(c)
            if len(hits) == 1:
                return hits[0]

        # unique substring match
        for name in preferred_names:
            if name is None:
                continue
            n = str(name).strip().lower()
            matches = [c for c in cols if n in c.lower()]
            if len(matches) == 1:
                return matches[0]

        return None

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def recode_dislike(series):
        # 1 if 4/5, 0 if 1/2/3, NaN otherwise
        x = to_num(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x >= 1) & (x <= 3)] = 0.0
        out[(x == 4) | (x == 5)] = 1.0
        return out

    def recode_like(series):
        # 1 if 1/2, 0 if 3/4/5, NaN otherwise
        x = to_num(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x == 1) | (x == 2)] = 1.0
        out[(x >= 3) & (x <= 5)] = 0.0
        return out

    # ----------------------------
    # Load + filter
    # ----------------------------
    df = pd.read_csv(data_source)

    year_col = find_col(df, ["YEAR", "year"])
    if year_col is not None:
        df[year_col] = to_num(df[year_col])
        df = df.loc[df[year_col] == 1993].copy()

    educ_col = find_col(df, ["EDUC", "educ", "education", "years_education", "schooling"])
    if educ_col is None:
        raise ValueError("Could not find education column (EDUC/educ).")
    df[educ_col] = to_num(df[educ_col])

    # CRITICAL: drop missing education up-front (used in every model + sample mean line)
    df = df.dropna(subset=[educ_col]).copy()

    # ----------------------------
    # Canonical Figure-1 genres (paper order)
    # ----------------------------
    genre_order = [
        "Latin/Salsa", "Jazz", "Blues/R&B", "Show Tunes", "Oldies", "Classical",
        "Reggae", "Swing", "New Age/Space", "Opera", "Bluegrass", "Folk",
        "Easy Listening", "Pop/Rock", "Rap", "Heavy Metal", "Country", "Gospel"
    ]

    genre_candidates = {
        "Latin/Salsa": ["LATIN", "latin"],
        "Jazz": ["JAZZ", "jazz"],
        "Blues/R&B": ["BLUES", "blues"],
        "Show Tunes": ["MUSICALS", "musicals", "SHOWTUNES", "showtunes"],
        "Oldies": ["OLDIES", "oldies"],
        "Classical": ["CLASSICL", "classicl", "CLASSICAL", "classical"],
        "Reggae": ["REGGAE", "reggae"],
        "Swing": ["BIGBAND", "bigband", "SWING", "swing"],
        "New Age/Space": ["NEWAGE", "newage", "NEW_AGE", "new_age"],
        "Opera": ["OPERA", "opera"],
        "Bluegrass": ["BLUGRASS", "blugrass", "BLUEGRASS", "bluegrass"],
        "Folk": ["FOLK", "folk"],
        "Easy Listening": ["MOODEASY", "moodeasy", "EASYLIST", "easylist", "EASY", "easy"],
        "Pop/Rock": ["CONROCK", "conrock", "POPROCK", "poprock", "ROCK", "rock"],
        "Rap": ["RAP", "rap", "HIPHOP", "hiphop", "HIP_HOP", "hip_hop"],
        "Heavy Metal": ["HVYMETAL", "hvymetal", "HEAVYMETAL", "heavymetal", "HEAVY_METAL", "heavy_metal"],
        "Country": ["COUNTRY", "country", "COUNTRYWESTERN", "countrywestern", "COUNTRY_WESTERN", "country_western"],
        "Gospel": ["GOSPEL", "gospel"],
    }

    resolved_cols = {}
    for label in genre_order:
        col = find_col(df, genre_candidates.get(label, []))
        if col is None:
            tokens = label.lower().replace("/", " ").replace("&", " ").replace("-", " ").split()
            col = find_col(df, [], contains_any=tokens)
        if col is None:
            raise ValueError(f"Could not find column for genre '{label}'.")
        resolved_cols[label] = col

    # ----------------------------
    # Recode dislikes/likes
    # ----------------------------
    dislike = pd.DataFrame(index=df.index)
    like = pd.DataFrame(index=df.index)

    for g in genre_order:
        raw = df[resolved_cols[g]]
        dislike[g] = recode_dislike(raw)
        like[g] = recode_like(raw)

    # Compute sample mean education using a consistent analysis sample:
    # respondents with non-missing education AND at least one non-missing genre response
    any_genre_nonmissing = dislike.notna().any(axis=1)
    df_mean = df.loc[any_genre_nonmissing].copy()
    sample_mean_edu = float(df_mean[educ_col].mean()) if len(df_mean) else np.nan

    # ----------------------------
    # Fit per-genre logistic models + mean education among "likers"
    # ----------------------------
    rows = []
    for g in genre_order:
        y = dislike[g]
        educ = df[educ_col]

        others = [h for h in genre_order if h != g]
        tol_minus_g = (1.0 - dislike[others]).sum(axis=1, min_count=len(others))  # NaN if any of 17 missing

        # CRITICAL: drop missing before fitting
        model_df = pd.DataFrame({"y": y, "tolerance": tol_minus_g, "educ": educ}).dropna()

        coef_tol = np.nan
        pval_tol = np.nan
        if len(model_df) >= 50 and model_df["y"].nunique() >= 2 and model_df["tolerance"].nunique() >= 2:
            X = sm.add_constant(model_df[["tolerance", "educ"]], has_constant="add")
            try:
                fit = sm.Logit(model_df["y"], X).fit(disp=False, maxiter=500)
                coef_tol = float(fit.params["tolerance"])
                pval_tol = float(fit.pvalues["tolerance"])
            except Exception:
                # fallback: regularized (keeps pipeline from crashing)
                try:
                    fit = sm.Logit(model_df["y"], X).fit_regularized(disp=False, maxiter=5000)
                    coef_tol = float(fit.params["tolerance"])
                    pval_tol = np.nan
                except Exception:
                    coef_tol = np.nan
                    pval_tol = np.nan

        # Mean education among likers (1/2 only)
        aud_df = pd.DataFrame({"like": like[g], "educ": educ}).dropna()
        if len(aud_df) and (aud_df["like"] == 1.0).any():
            mean_edu = float(aud_df.loc[aud_df["like"] == 1.0, "educ"].mean())
        else:
            mean_edu = np.nan

        rows.append({"genre": g, "coef_tolerance": coef_tol, "pval": pval_tol, "mean_edu": mean_edu})

    res = pd.DataFrame(rows).set_index("genre").reindex(genre_order).reset_index()

    # ----------------------------
    # Plot (match Figure 1: left=coef solid, right=education dash-dot, dotted mean edu)
    # ----------------------------
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260119_074740/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Global styling closer to print
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
    })

    x = np.arange(len(genre_order), dtype=int)
    coef_vals = res["coef_tolerance"].to_numpy(dtype="float64")
    edu_vals = res["mean_edu"].to_numpy(dtype="float64")

    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=150)
    ax2 = ax.twinx()

    # LEFT axis: coefficient (SOLID)
    ax.plot(x, coef_vals, color="black", lw=2.5, ls="-", zorder=3)

    # RIGHT axis: mean education (DASH-DOT)
    ax2.plot(x, edu_vals, color="black", lw=2.5, ls="-.", zorder=2)

    # Sample mean education line on RIGHT axis (DOTTED)
    if np.isfinite(sample_mean_edu):
        ax2.axhline(sample_mean_edu, color="black", lw=1.5, ls=":", zorder=1)

    # Labels (comparable to article)
    ax.set_xlabel("Type of Music", fontweight="bold")
    ax.set_ylabel(
        "Coefficients for Musical Tolerance as It Affects\n"
        "One’s Probability of Disliking Each Music Genre"
    )
    ax2.set_ylabel(
        "Mean Educational Level of Respondents Who\n"
        "Reported Liking Each Music Genre",
        rotation=270,
        labelpad=28
    )

    # X ticks
    ax.set_xticks(x)
    ax.set_xticklabels(genre_order, rotation=45, ha="right")

    # Axis ranges/ticks (match GT)
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax.set_yticklabels(["-.5", "-.4", "-.3", "-.2", "-.1"])

    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # Clean look
    ax.grid(False)
    ax2.grid(False)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=8, width=1.5)
    ax2.tick_params(direction="out", length=8, width=1.5)

    # ----------------------------
    # Annotations / callouts (axis-specific targets)
    # ----------------------------
    # Choose a stable early index that is likely finite
    def first_finite(arr, fallback=2):
        for i, v in enumerate(arr):
            if np.isfinite(v):
                return i
        return min(max(fallback, 0), len(arr) - 1)

    i = first_finite(edu_vals, fallback=2)
    j = first_finite(coef_vals, fallback=2)

    # Mean education callout (points to dash-dot on ax2)
    if np.isfinite(edu_vals[i]):
        ax2.annotate(
            "Mean Education of Genre Audience",
            xy=(x[i], float(edu_vals[i])),
            xycoords="data",
            xytext=(x[min(i + 1, len(x) - 1)] + 0.2, 14.2),
            textcoords="data",
            arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="center",
        )

    # Coefficient callout (points to solid on ax)
    if np.isfinite(coef_vals[j]):
        ax.annotate(
            "Coefficient for Musical Tolerance",
            xy=(x[j], float(coef_vals[j])),
            xycoords="data",
            xytext=(x[min(j + 1, len(x) - 1)] + 0.2, -0.47),
            textcoords="data",
            arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="center",
        )

    # Sample mean education arrow (points to dotted line on ax2)
    if np.isfinite(sample_mean_edu):
        k = min(6, len(x) - 1)
        ax2.annotate(
            "Sample Mean Education",
            xy=(x[k], sample_mean_edu),
            xycoords="data",
            xytext=(x[k], sample_mean_edu - 0.35),
            textcoords="data",
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
            fontsize=14,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)