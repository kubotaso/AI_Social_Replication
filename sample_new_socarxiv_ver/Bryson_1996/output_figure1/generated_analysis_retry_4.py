import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    # Step 1: Listwise deletion on all 18 genres (1-5 range) + educ
    df_analysis = df.copy()
    for col in genre_cols:
        df_analysis = df_analysis[df_analysis[col].between(1, 5)]
    df_analysis = df_analysis[df_analysis['educ'].notna()]
    df_analysis = df_analysis.reset_index(drop=True)
    n = len(df_analysis)
    print(f"Analysis sample size: {n}")

    # Step 2: For each genre, run logistic regression
    tolerance_coeffs = {}
    tolerance_pvals = {}
    tolerance_se = {}

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
        tolerance_se[genre] = result.bse['tolerance']

    # Step 3: Mean education of fans (rating 1 or 2) - full sample
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
    print("\n=== TOLERANCE COEFFICIENTS (sorted) ===")
    print(f"{'Genre':<25} {'Coeff':>8} {'SE':>8} {'p-value':>12}")
    print("-" * 55)
    for g in sorted_genres:
        label = display_labels[g]
        print(f"{label:<25} {tolerance_coeffs[g]:>8.4f} {tolerance_se[g]:>8.4f} {tolerance_pvals[g]:>12.6f}")

    print("\n=== MEAN EDUCATION OF FANS ===")
    print(f"{'Genre':<25} {'Mean Educ':>10} {'N fans':>8}")
    print("-" * 45)
    for g in sorted_genres:
        label = display_labels[g]
        print(f"{label:<25} {mean_educ_fans[g]:>10.2f} {n_fans[g]:>8}")

    # Step 5: Create figure - closely matching paper's visual style
    # Paper figure characteristics from original:
    # - Solid line for tolerance coefficients (thin, no markers)
    # - Dash-dot line for education (thick, no markers)
    # - Dotted horizontal line for sample mean education
    # - Y-axis labels: -.1 through -.5 (left), 12-15 (right)
    # - Bold text labels for each line directly on the plot
    # - Arrow annotation for "Sample Mean Education"
    # - X-axis labels rotated ~60 degrees

    fig, ax1 = plt.subplots(figsize=(10, 6.5))

    x_pos = list(range(len(sorted_genres)))
    labels_list = [display_labels[g] for g in sorted_genres]
    coeffs = [tolerance_coeffs[g] for g in sorted_genres]
    educ_vals = [mean_educ_fans[g] for g in sorted_genres]

    # Left Y-axis: Tolerance coefficients (SOLID line)
    ax1.plot(x_pos, coeffs, '-', color='black', linewidth=1.5)
    ax1.set_ylabel('Coefficients for Musical Tolerance as It Affects\nOne\'s Probability of Disliking Each Music Genre',
                    fontsize=9.5)
    ax1.set_ylim(-0.52, -0.08)
    ax1.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1])
    ax1.set_yticklabels(['-.5', '-.4', '-.3', '-.2', '-.1'], fontsize=9)
    ax1.set_xlabel('Type of Music', fontsize=11)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels_list, rotation=60, ha='right', fontsize=8.5)

    # Right Y-axis: Mean education (DASH-DOT line)
    ax2 = ax1.twinx()
    ax2.plot(x_pos, educ_vals, '-.', color='black', linewidth=2.0)
    ax2.set_ylabel('Mean Educational Level of Respondents Who\nReported Liking Each Music Genre',
                    fontsize=9.5, rotation=270, labelpad=18)
    ax2.set_ylim(12, 15)
    ax2.set_yticks([12, 13, 14, 15])
    ax2.tick_params(axis='y', labelsize=9)

    # Horizontal dotted reference line at sample mean education
    ax2.axhline(y=sample_mean_educ, color='black', linestyle=':', linewidth=0.8)

    # Text annotations matching paper placement
    # "Mean Education of Genre Audience" - upper left area
    ax2.text(2.5, 14.55, 'Mean Education of Genre Audience',
             fontsize=10, fontweight='bold')

    # "Coefficient for Musical Tolerance" - lower left area
    ax1.text(1.5, -0.465, 'Coefficient for Musical Tolerance',
             fontsize=10, fontweight='bold')

    # "Sample Mean Education" with arrow - middle area
    ax2.annotate('Sample Mean Education',
                 xy=(7.5, sample_mean_educ),
                 xytext=(7.5, sample_mean_educ - 0.35),
                 fontsize=9.5, ha='center',
                 arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    fig.tight_layout()

    # Save figure
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, 'generated_results_attempt_4.jpg')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', format='jpeg')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    return {
        'tolerance_coeffs': {display_labels[g]: tolerance_coeffs[g] for g in sorted_genres},
        'mean_educ_fans': {display_labels[g]: mean_educ_fans[g] for g in sorted_genres},
        'genre_order': [display_labels[g] for g in sorted_genres],
        'sample_mean_educ': sample_mean_educ,
        'n': n
    }


def score_against_ground_truth(result):
    """Compare against paper figure values."""
    # Paper order (from viewing original figure)
    paper_order = [
        'Latin/Salsa', 'Jazz', 'Blues/R&B', 'Show Tunes', 'Oldies',
        'Classical', 'Reggae', 'New Age/Space', 'Swing', 'Opera',
        'Bluegrass', 'Folk', 'Easy Listening', 'Pop/Rock',
        'Rap', 'Heavy Metal', 'Country', 'Gospel'
    ]

    # Corrected approximate values from original figure
    paper_coeffs = {
        'Latin/Salsa': -0.43, 'Jazz': -0.40, 'Blues/R&B': -0.36,
        'Show Tunes': -0.35, 'Oldies': -0.35, 'Classical': -0.34,
        'Reggae': -0.32, 'New Age/Space': -0.31, 'Swing': -0.31,
        'Opera': -0.29, 'Bluegrass': -0.29, 'Folk': -0.27,
        'Easy Listening': -0.27, 'Pop/Rock': -0.25,
        'Rap': -0.19, 'Heavy Metal': -0.16, 'Country': -0.16,
        'Gospel': -0.13
    }

    print("\n=== SCORING ===")
    my_order = result['genre_order']
    order_matches = sum(1 for a, b in zip(my_order, paper_order) if a == b)
    print(f"Genre order matches: {order_matches}/18")
    if order_matches < 18:
        for i, (a, b) in enumerate(zip(my_order, paper_order)):
            if a != b:
                print(f"  Position {i+1}: mine={a}, paper={b}")

    print(f"\nCoefficient comparison:")
    total_diff = 0
    for genre in my_order:
        my_val = result['tolerance_coeffs'][genre]
        paper_val = paper_coeffs.get(genre, 0)
        diff = abs(my_val - paper_val)
        total_diff += diff
        if diff > 0.01:
            print(f"  {genre}: mine={my_val:.4f}, paper={paper_val:.2f}, diff={diff:.4f}")
    print(f"  Average coefficient diff: {total_diff/18:.4f}")


if __name__ == "__main__":
    result = run_analysis("gss1993_clean.csv")
    print(f"\n=== SUMMARY ===")
    print(f"N = {result['n']}")
    print(f"Sample mean education = {result['sample_mean_educ']:.2f}")
    score_against_ground_truth(result)
