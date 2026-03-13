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
        'musicals': 'Show Tunes', 'oldies': 'Oldies', 'classicl': 'Classical',
        'reggae': 'Reggae', 'bigband': 'Swing', 'newage': 'New Age/Space',
        'opera': 'Opera', 'blugrass': 'Bluegrass', 'folk': 'Folk',
        'moodeasy': 'Easy Listening', 'conrock': 'Pop/Rock',
        'rap': 'Rap', 'hvymetal': 'Heavy Metal', 'country': 'Country',
        'gospel': 'Gospel'
    }

    # Step 1: Listwise deletion on genres only (NOT educ)
    # The paper's N=912 may include people with missing educ who are still in the genre sample
    df_genre_valid = df.copy()
    for col in genre_cols:
        df_genre_valid = df_genre_valid[df_genre_valid[col].between(1, 5)]
    df_genre_valid = df_genre_valid.reset_index(drop=True)
    print(f"Sample with all 18 genres valid: {len(df_genre_valid)}")

    # For regressions, also need valid educ - use listwise deletion per model
    df_analysis = df_genre_valid[df_genre_valid['educ'].notna()].reset_index(drop=True)
    print(f"Sample with genres + educ valid: {len(df_analysis)}")

    # Step 2: For each genre, run logistic regression
    tolerance_coeffs = {}
    tolerance_pvals = {}

    for genre in genre_cols:
        dv = (df_analysis[genre] >= 4).astype(int)
        other_genres = [g for g in genre_cols if g != genre]
        tolerance = sum((df_analysis[g] < 4).astype(int) for g in other_genres)
        educ = df_analysis['educ']
        X = pd.DataFrame({'tolerance': tolerance, 'educ': educ})
        X = sm.add_constant(X)
        model = sm.Logit(dv, X)
        result = model.fit(disp=0)
        tolerance_coeffs[genre] = result.params['tolerance']
        tolerance_pvals[genre] = result.pvalues['tolerance']

    # Step 3: Mean education of fans - use FULL sample with valid educ
    df_fans = df.copy()
    df_fans = df_fans[df_fans['educ'].notna()]
    mean_educ_fans = {}
    n_fans = {}
    for genre in genre_cols:
        fans = df_fans[df_fans[genre].isin([1, 2])]
        if len(fans) > 0:
            mean_educ_fans[genre] = fans['educ'].mean()
            n_fans[genre] = len(fans)
        else:
            mean_educ_fans[genre] = np.nan
            n_fans[genre] = 0

    sample_mean_educ = df_analysis['educ'].mean()

    # Step 4: Sort genres by tolerance coefficient (ascending = most negative first)
    sorted_genres = sorted(genre_cols, key=lambda g: tolerance_coeffs[g])

    # Print results
    print(f"\nSample mean education: {sample_mean_educ:.2f}")
    print("\n=== TOLERANCE COEFFICIENTS (sorted, most negative first) ===")
    print(f"{'Genre':<25} {'Coeff':>8} {'p-value':>12}")
    print("-" * 47)
    for g in sorted_genres:
        label = display_labels[g]
        print(f"{label:<25} {tolerance_coeffs[g]:>8.4f} {tolerance_pvals[g]:>12.6f}")

    print("\n=== MEAN EDUCATION OF FANS ===")
    print(f"{'Genre':<25} {'Mean Educ':>10} {'N fans':>8}")
    print("-" * 45)
    for g in sorted_genres:
        label = display_labels[g]
        print(f"{label:<25} {mean_educ_fans[g]:>10.2f} {n_fans[g]:>8}")

    print("\n=== GENRE ORDER ===")
    print([display_labels[g] for g in sorted_genres])

    # Step 5: Create figure matching the paper's style closely
    fig, ax1 = plt.subplots(figsize=(10, 6.5))

    x_pos = range(len(sorted_genres))
    labels = [display_labels[g] for g in sorted_genres]
    coeffs = [tolerance_coeffs[g] for g in sorted_genres]
    educ_vals = [mean_educ_fans[g] for g in sorted_genres]

    # Left axis: tolerance coefficients (SOLID line - matching paper)
    ax1.set_xlabel('Type of Music', fontsize=11)
    ax1.set_ylabel('Coefficients for Musical Tolerance as It Affects\nOne\'s Probability of Disliking Each Music Genre',
                    fontsize=9)
    line1, = ax1.plot(x_pos, coeffs, '-', color='black', linewidth=1.8)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)

    # Left Y-axis range: -0.5 to -0.1 (matching paper)
    ax1.set_ylim(-0.52, -0.08)
    ax1.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax1.set_yticklabels(['-.5', '-.4', '-.3', '-.2', '-.1'])

    # Right axis: mean education (DASH-DOT line - matching paper)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean Educational Level of Respondents Who\nReported Liking Each Music Genre',
                    fontsize=9, rotation=270, labelpad=20)
    line2, = ax2.plot(x_pos, educ_vals, '-.', color='black', linewidth=1.8)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.set_ylim(12, 15)

    # Horizontal dotted reference line at sample mean education
    hline = ax2.axhline(y=sample_mean_educ, color='black', linestyle=':', linewidth=0.8)

    # Annotations matching paper style
    # "Sample Mean Education" with arrow pointing to the line
    mid_x = len(sorted_genres) // 2
    ax2.annotate('Sample Mean Education',
                 xy=(mid_x, sample_mean_educ),
                 xytext=(mid_x, sample_mean_educ - 0.35),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    # "Coefficient for Musical Tolerance" label near the solid line
    ax1.text(2, -0.47, 'Coefficient for Musical Tolerance',
             fontsize=9, fontweight='bold')

    # "Mean Education of Genre Audience" label near the dashed line
    ax2.text(1.5, 14.55, 'Mean Education of Genre Audience',
             fontsize=9, fontweight='bold')

    fig.tight_layout()

    # Save figure
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, 'generated_results_attempt_3.jpg')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', format='jpeg')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    return {
        'tolerance_coeffs': {display_labels[g]: tolerance_coeffs[g] for g in sorted_genres},
        'mean_educ_fans': {display_labels[g]: mean_educ_fans[g] for g in sorted_genres},
        'genre_order': [display_labels[g] for g in sorted_genres],
        'sample_mean_educ': sample_mean_educ,
        'n': len(df_analysis),
        'n_genre_valid': len(df_genre_valid)
    }


def score_against_ground_truth(result):
    """Compare against paper figure values (corrected after viewing original)."""
    paper_order = [
        'Latin/Salsa', 'Jazz', 'Blues/R&B', 'Show Tunes', 'Oldies',
        'Classical', 'Reggae', 'New Age/Space', 'Swing', 'Opera',
        'Bluegrass', 'Folk', 'Easy Listening', 'Pop/Rock',
        'Rap', 'Heavy Metal', 'Country', 'Gospel'
    ]

    # Corrected approximate values read from the actual figure image
    paper_coeffs = {
        'Latin/Salsa': -0.43, 'Jazz': -0.40, 'Blues/R&B': -0.36,
        'Show Tunes': -0.35, 'Oldies': -0.35, 'Classical': -0.34,
        'Reggae': -0.32, 'New Age/Space': -0.31, 'Swing': -0.31,
        'Opera': -0.29, 'Bluegrass': -0.29, 'Folk': -0.27,
        'Easy Listening': -0.27, 'Pop/Rock': -0.25,
        'Rap': -0.19, 'Heavy Metal': -0.16, 'Country': -0.16,
        'Gospel': -0.13
    }

    # Corrected education values from figure
    paper_educ = {
        'Latin/Salsa': 13.7, 'Jazz': 13.6, 'Blues/R&B': 13.4,
        'Show Tunes': 13.7, 'Oldies': 13.4, 'Classical': 14.0,
        'Reggae': 14.0, 'New Age/Space': 14.1, 'Swing': 13.4,
        'Opera': 13.9, 'Bluegrass': 12.8, 'Folk': 13.4,
        'Easy Listening': 13.1, 'Pop/Rock': 13.6,
        'Rap': 12.8, 'Heavy Metal': 12.7, 'Country': 12.5,
        'Gospel': 12.6
    }

    print("\n=== SCORING ===")
    my_order = result['genre_order']
    order_matches = sum(1 for a, b in zip(my_order, paper_order) if a == b)
    print(f"Genre order matches: {order_matches}/18")

    print(f"\n{'Genre':<25} {'My Coeff':>10} {'Paper':>8} {'Diff':>8}")
    for genre in my_order:
        my_val = result['tolerance_coeffs'][genre]
        paper_val = paper_coeffs.get(genre, 0)
        print(f"{genre:<25} {my_val:>10.4f} {paper_val:>8.2f} {abs(my_val-paper_val):>8.4f}")

    print(f"\n{'Genre':<25} {'My Educ':>10} {'Paper':>8} {'Diff':>8}")
    for genre in my_order:
        my_val = result['mean_educ_fans'][genre]
        paper_val = paper_educ.get(genre, 0)
        print(f"{genre:<25} {my_val:>10.2f} {paper_val:>8.1f} {abs(my_val-paper_val):>8.2f}")


if __name__ == "__main__":
    result = run_analysis("gss1993_clean.csv")
    print(f"\n=== SUMMARY ===")
    print(f"N (regression sample) = {result['n']}")
    print(f"N (genre valid) = {result['n_genre_valid']}")
    print(f"Sample mean education = {result['sample_mean_educ']:.2f}")
    score_against_ground_truth(result)
