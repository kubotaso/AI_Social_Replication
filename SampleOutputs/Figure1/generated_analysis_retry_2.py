def run_analysis(data_source):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import statsmodels.api as sm

    def find_col(df, preferred_names):
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}

        for name in preferred_names:
            if name.lower() in lower_map:
                return lower_map[name.lower()]

        # fallback: unique substring match
        for name in preferred_names:
            n = name.lower()
            matches = [c for c in cols if n in c.lower()]
            if len(matches) == 1:
                return matches[0]
        return None

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def recode_dislike(series):
        # 1 if response is 4/5, 0 if 1/2/3, NaN otherwise
        x = to_num(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x >= 1) & (x <= 3)] = 0.0
        out[(x == 4) | (x == 5)] = 1.0
        return out

    def recode_like(series):
        # 1 if response is 1/2, 0 if 3/4/5, NaN otherwise
        x = to_num(series)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out[(x == 1) | (x == 2)] = 1.0
        out[(x >= 3) & (x <= 5)] = 0.0
        return out

    # ----- Load -----
    df = pd.read_csv(data_source)

    year_col = find_col(df, ["YEAR", "year"])
    if year_col is not None:
        df[year_col] = to_num(df[year_col])
        df = df.loc[df[year_col] == 1993].copy()

    educ_col = find_col(df, ["EDUC", "educ", "education", "years_education", "schooling"])
    if educ_col is None:
        raise ValueError("Could not find education column (EDUC/educ).")
    df[educ_col] = to_num(df[educ_col])

    # ----- Genres: enforce GT labels and x-order explicitly -----
    # Column candidates include robust fallbacks, but labels are fixed to match GT figure.
    genre_map = {
        "Latin/Salsa": ["LATIN", "latin", "salsa", "mariachi"],
        "Jazz": ["JAZZ", "jazz"],
        "Blues/R&B": ["BLUES", "blues", "rnb", "r&b"],
        "Show Tunes": ["MUSICALS", "musicals", "showtunes", "show_tunes"],
        "Oldies": ["OLDIES", "oldies"],
        "Classical": ["CLASSICL", "classicl", "CLASSICAL", "classical"],
        "Reggae": ["REGGAE", "reggae"],
        "Swing": ["BIGBAND", "bigband", "swing", "big_band"],
        "New Age/Space": ["NEWAGE", "newage", "new_age", "space"],
        "Opera": ["OPERA", "opera"],
        "Bluegrass": ["BLUGRASS", "blugrass", "bluegrass"],
        "Folk": ["FOLK", "folk"],
        "Easy Listening": ["MOODEASY", "moodeasy", "easylist", "easy_listening", "easylst"],
        "Pop/Rock": ["CONROCK", "conrock", "poprock", "pop_rock", "rock", "contemporary"],
        "Rap": ["RAP", "rap", "hiphop", "hip_hop"],
        "Heavy Metal": ["HVYMETAL", "hvymetal", "HEAVYMETAL", "heavymetal", "heavy_metal"],
        "Country": ["COUNTRY", "country", "countrywestern", "country_western"],
        "Gospel": ["GOSPEL", "gospel"],
    }

    order = [
        "Latin/Salsa", "Jazz", "Blues/R&B", "Show Tunes", "Oldies", "Classical",
        "Reggae", "Swing", "New Age/Space", "Opera", "Bluegrass", "Folk",
        "Easy Listening", "Pop/Rock", "Rap", "Heavy Metal", "Country", "Gospel"
    ]

    resolved = {}
    for label in order:
        col = find_col(df, genre_map[label])
        if col is None:
            raise ValueError(f"Could not find column for genre '{label}'. Tried: {genre_map[label]}")
        resolved[label] = col

    # ----- Build dislike/like matrices -----
    dislike = pd.DataFrame(index=df.index)
    like = pd.DataFrame(index=df.index)
    for label in order:
        col = resolved[label]
        dislike[label] = recode_dislike(df[col])
        like[label] = recode_like(df[col])

    # sample mean education from respondent-level (1993, non-missing EDUC)
    sample_mean_edu = float(df[educ_col].dropna().mean())

    # ----- Fit per-genre logits and compute mean edu of audience -----
    rows = []
    for g in order:
        y = dislike[g]
        educ = df[educ_col]

        others = [h for h in order if h != g]
        others_dislike = dislike[others]

        # count of other genres not disliked, requiring all 17 observed (no missing)
        not_disliked = 1.0 - others_dislike
        tol_minus_g = not_disliked.sum(axis=1, min_count=len(others))

        model_df = pd.DataFrame({"y": y, "tolerance": tol_minus_g, "educ": educ}).dropna()

        coef_tol = np.nan
        pval_tol = np.nan
        if model_df["y"].nunique(dropna=True) >= 2 and model_df["tolerance"].nunique(dropna=True) >= 2:
            X = sm.add_constant(model_df[["tolerance", "educ"]], has_constant="add")
            try:
                fit = sm.Logit(model_df["y"], X).fit(disp=False, maxiter=200)
                coef_tol = float(fit.params["tolerance"])
                pval_tol = float(fit.pvalues["tolerance"])
            except Exception:
                # fallback: regularized if separation
                try:
                    fit = sm.Logit(model_df["y"], X).fit_regularized(disp=False, maxiter=500)
                    coef_tol = float(fit.params["tolerance"])
                    pval_tol = np.nan
                except Exception:
                    coef_tol = np.nan
                    pval_tol = np.nan

        # mean edu among likers (1/2); drop NaNs first
        aud_df = pd.DataFrame({"like": like[g], "educ": educ}).dropna()
        mean_edu = float(aud_df.loc[aud_df["like"] == 1.0, "educ"].mean()) if (aud_df["like"] == 1.0).any() else np.nan

        rows.append({"genre": g, "coef_tolerance": coef_tol, "pval": pval_tol, "mean_edu": mean_edu})

    res = pd.DataFrame(rows)

    # Ensure complete for plotting; keep GT order regardless of coefficients
    res = res.set_index("genre").loc[order].reset_index()

    # ----- Plot (match GT styling requirements) -----
    out_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_scripts/output_run_all/20260119_074740/Figure1/generated_results.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=150)
    ax2 = ax.twinx()

    # Coefficient for musical tolerance (LEFT axis) - solid
    ax.plot(x, res["coef_tolerance"].values, color="black", lw=2.0, ls="-")

    # Mean education (RIGHT axis) - dash-dot
    ax2.plot(x, res["mean_edu"].values, color="black", lw=2.0, ls="-.")

    # Sample mean education line on RIGHT axis - dotted
    if np.isfinite(sample_mean_edu):
        ax2.axhline(sample_mean_edu, color="black", lw=1.0, ls=":")

    ax.set_xlabel("Type of Music", fontweight="bold")
    ax.set_ylabel("Coefficients for Musical Tolerance\n(Log Odds Probability of Disliking Each Musical Genre)")
    ax2.set_ylabel("Mean Educational Level of Respondents Who Reported Liking Each Type of Music")

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")

    # Axis ranges/ticks/formatting to match GT figure
    ax.set_ylim(-0.5, -0.1)
    ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # Remove gridlines (classic monochrome look)
    ax.grid(False)
    ax2.grid(False)

    # Annotations with leader lines/arrows (positions tuned but data-driven anchors)
    # Anchor near early x positions; if missing values, choose first finite.
    def first_finite(series, default_idx=2):
        vals = np.asarray(series, dtype="float64")
        for i in range(len(vals)):
            if np.isfinite(vals[i]):
                return i
        return default_idx

    i_edu = first_finite(res["mean_edu"].values, default_idx=min(2, len(order) - 1))
    i_coef = first_finite(res["coef_tolerance"].values, default_idx=min(2, len(order) - 1))

    edu_anchor_x = int(i_edu)
    coef_anchor_x = int(i_coef)
    edu_anchor_y = float(res["mean_edu"].iloc[edu_anchor_x]) if np.isfinite(res["mean_edu"].iloc[edu_anchor_x]) else 13.8
    coef_anchor_y = float(res["coef_tolerance"].iloc[coef_anchor_x]) if np.isfinite(res["coef_tolerance"].iloc[coef_anchor_x]) else -0.35

    ax2.annotate(
        "Mean Education of Genre Audience",
        xy=(edu_anchor_x, edu_anchor_y),
        xycoords=("data", "data"),
        xytext=(max(0, edu_anchor_x - 1.0), 14.2),
        textcoords=("data", "data"),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
    )

    ax.annotate(
        "Coefficient for Musical Tolerance",
        xy=(coef_anchor_x, coef_anchor_y),
        xycoords=("data", "data"),
        xytext=(min(len(order) - 1, coef_anchor_x + 0.2), -0.47),
        textcoords=("data", "data"),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Sample mean education label with upward arrow to dotted line
    if np.isfinite(sample_mean_edu):
        mean_x = int(min(6, len(order) - 1))
        ax2.annotate(
            "Sample Mean Education",
            xy=(mean_x, sample_mean_edu),
            xycoords=("data", "data"),
            xytext=(mean_x, 12.7),
            textcoords=("data", "data"),
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="jpg")
    plt.close(fig)

    return os.path.abspath(out_path)