import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def run_analysis(data_source):
    df = pd.read_csv(data_source)

    genre_cols = ['latin', 'jazz', 'blues', 'musicals', 'oldies', 'classicl',
                  'reggae', 'bigband', 'newage', 'opera', 'blugrass', 'folk',
                  'moodeasy', 'conrock', 'rap', 'hvymetal', 'country', 'gospel']

    display_labels = {
        'latin': 'Latin/Salsa', 'jazz': 'Jazz', 'blues': 'Blues/R&B',
        'musicals': 'Show Tunes', 'oldies': 'Oldies', 'classicl': 'Classical/Chamber',
        'reggae': 'Reggae', 'bigband': 'Swing/Big Band', 'newage': 'New Age/Space',
        'opera': 'Opera', 'blugrass': 'Bluegrass', 'folk': 'Folk',
        'moodeasy': 'Easy Listening', 'conrock': 'Pop/Contemporary Rock',
        'rap': 'Rap', 'hvymetal': 'Heavy Metal', 'country': 'Country/Western',
        'gospel': 'Gospel'
    }

    # Step 1: Listwise deletion - keep only respondents with all 18 genres rated 1-5
    df_analysis = df.copy()
    for col in genre_cols:
        df_analysis = df_analysis[df_analysis[col].between(1, 5)]

    # Also require valid education
    df_analysis = df_analysis[df_analysis['educ'].notna()]
    df_analysis = df_analysis.reset_index(drop=True)

    print(f"Analysis sample size (all 18 genres valid + educ): {len(df_analysis)}")

    # Step 2: For each genre, run logistic regression
    tolerance_coeffs = {}
    tolerance_pvals = {}

    for genre in genre_cols:
        # DV: 1 if rating >= 4, 0 otherwise
        dv = (df_analysis[genre] >= 4).astype(int)

        # IV1: Musical tolerance = count of OTHER 17 genres where rating < 4
        other_genres = [g for g in genre_cols if g != genre]
        tolerance = sum((df_analysis[g] < 4).astype(int) for g in other_genres)

        # IV2: Education
        educ = df_analysis['educ']

        # Run logistic regression
        X = pd.DataFrame({'tolerance': tolerance, 'educ': educ})
        X = sm.add_constant(X)

        try:
            model = sm.Logit(dv, X)
            result = model.fit(disp=0)
            tolerance_coeffs[genre] = result.params['tolerance']
            tolerance_pvals[genre] = result.pvalues['tolerance']
        except Exception as e:
            print(f"Error for {genre}: {e}")
            tolerance_coeffs[genre] = np.nan
            tolerance_pvals[genre] = np.nan

    # Step 3: Mean education of fans (rating 1 or 2)
    # Use full sample with valid educ, not restricted to the 912
    df_fans = df.copy()
    df_fans = df_fans[df_fans['educ'].notna()]

    mean_educ_fans = {}
    for genre in genre_cols:
        fans = df_fans[df_fans[genre].isin([1, 2])]
        if len(fans) > 0:
            mean_educ_fans[genre] = fans['educ'].mean()
        else:
            mean_educ_fans[genre] = np.nan

    # Sample mean education
    sample_mean_educ = df_analysis['educ'].mean()
    print(f"Sample mean education: {sample_mean_educ:.2f}")

    # Step 4: Sort genres by tolerance coefficient (ascending)
    sorted_genres = sorted(genre_cols, key=lambda g: tolerance_coeffs[g])

    # Print results
    print("\n=== TOLERANCE COEFFICIENTS (sorted) ===")
    print(f"{'Genre':<25} {'Coeff':>8} {'p-value':>10}")
    print("-" * 45)
    for g in sorted_genres:
        label = display_labels[g]
        print(f"{label:<25} {tolerance_coeffs[g]:>8.4f} {tolerance_pvals[g]:>10.6f}")

    print("\n=== MEAN EDUCATION OF FANS ===")
    print(f"{'Genre':<25} {'Mean Educ':>10}")
    print("-" * 37)
    for g in sorted_genres:
        label = display_labels[g]
        print(f"{label:<25} {mean_educ_fans[g]:>10.2f}")

    # Step 5: Create the figure
    fig, ax1 = plt.subplots(figsize=(12, 7))

    x_pos = range(len(sorted_genres))
    labels = [display_labels[g] for g in sorted_genres]
    coeffs = [tolerance_coeffs[g] for g in sorted_genres]
    educ_vals = [mean_educ_fans[g] for g in sorted_genres]

    # Left axis: tolerance coefficients (solid line with markers)
    color1 = 'black'
    ax1.set_xlabel('Type of Music', fontsize=11)
    ax1.set_ylabel('Coefficients for Musical Tolerance as It Affects\nOne\'s Chances of Disliking Each Music Genre',
                    fontsize=9, color=color1)
    line1, = ax1.plot(x_pos, coeffs, 'o-', color=color1, linewidth=1.5, markersize=5, label='Coefficient of Musical Tolerance')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    # Right axis: mean education of fans (dashed line)
    ax2 = ax1.twinx()
    color2 = 'black'
    ax2.set_ylabel('Mean Educational Level of Respondents\nWho Reported Liking Each Music Genre',
                    fontsize=9, color=color2)
    line2, = ax2.plot(x_pos, educ_vals, 's--', color=color2, linewidth=1.5, markersize=5, label='Mean Education of Genre Audience')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Horizontal reference line at sample mean education
    hline = ax2.axhline(y=sample_mean_educ, color='gray', linestyle=':', linewidth=1.5, label='Sample Mean Education')

    # Set axis ranges
    ax1.set_ylim(-0.55, -0.05)
    ax2.set_ylim(11.5, 15.5)

    # Legend
    lines = [line1, line2, hline]
    labels_legend = [l.get_label() for l in lines]
    ax1.legend(lines, labels_legend, loc='lower left', fontsize=8)

    plt.title('Figure 1. The Effect of Being Musically Tolerant on Disliking\nEach Music Genre Compared to the Educational Composition of Genre Audiences',
              fontsize=10, pad=15)

    plt.tight_layout()

    # Save figure
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, 'generated_results_attempt_1.jpg')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', format='jpeg')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    return {
        'tolerance_coeffs': {display_labels[g]: tolerance_coeffs[g] for g in sorted_genres},
        'mean_educ_fans': {display_labels[g]: mean_educ_fans[g] for g in sorted_genres},
        'genre_order': [display_labels[g] for g in sorted_genres],
        'sample_mean_educ': sample_mean_educ,
        'n': len(df_analysis)
    }


def score_against_ground_truth():
    """Compare generated results against paper's approximate values."""
    # Ground truth from figure (approximate)
    paper_order = [
        'Latin/Salsa', 'Jazz', 'Blues/R&B', 'Show Tunes', 'Oldies',
        'Classical/Chamber', 'Reggae', 'New Age/Space', 'Swing/Big Band', 'Opera',
        'Bluegrass', 'Folk', 'Easy Listening', 'Pop/Contemporary Rock',
        'Rap', 'Heavy Metal', 'Country/Western', 'Gospel'
    ]

    paper_coeffs = {
        'Latin/Salsa': -0.13, 'Jazz': -0.15, 'Blues/R&B': -0.17,
        'Show Tunes': -0.18, 'Oldies': -0.22, 'Classical/Chamber': -0.22,
        'Reggae': -0.23, 'New Age/Space': -0.24, 'Swing/Big Band': -0.25,
        'Opera': -0.25, 'Bluegrass': -0.28, 'Folk': -0.30,
        'Easy Listening': -0.32, 'Pop/Contemporary Rock': -0.35,
        'Rap': -0.37, 'Heavy Metal': -0.42, 'Country/Western': -0.45,
        'Gospel': -0.48
    }

    paper_educ = {
        'Latin/Salsa': 14.0, 'Jazz': 14.3, 'Blues/R&B': 13.3,
        'Show Tunes': 14.0, 'Oldies': 12.7, 'Classical/Chamber': 14.7,
        'Reggae': 13.5, 'New Age/Space': 13.7, 'Swing/Big Band': 12.7,
        'Opera': 14.7, 'Bluegrass': 12.6, 'Folk': 14.0,
        'Easy Listening': 12.5, 'Pop/Contemporary Rock': 13.0,
        'Rap': 12.8, 'Heavy Metal': 12.2, 'Country/Western': 12.5,
        'Gospel': 12.7
    }

    print("\n=== SCORING AGAINST GROUND TRUTH ===")
    print(f"\nPaper genre order: {paper_order}")


if __name__ == "__main__":
    result = run_analysis("gss1993_clean.csv")

    print("\n=== SUMMARY ===")
    print(f"N = {result['n']}")
    print(f"Sample mean education = {result['sample_mean_educ']:.2f}")
    print(f"Genre order: {result['genre_order']}")

    score_against_ground_truth()
