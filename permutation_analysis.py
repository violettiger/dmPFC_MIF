#!/usr/bin/env python3
"""
Permutation Testing for Functional Connectivity Threshold Determination
========================================================================

This script performs permutation testing to determine the optimal threshold
for binarizing functional connectivity graphs based on Spearman correlation.

FEATURES:
- Permutation testing for null distribution
- Bootstrap statistical comparison
- Comprehensive visualization of results

Author: Moisés dos Santos Corrêa
Date: February 2026
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, ks_2samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ---------------------------------------------------------------------------
# GROUP DEFINITIONS
# ---------------------------------------------------------------------------
# Each entry is (group_name, mask_function). mask_function takes the (possibly
# shuffled) Timepoint/Treatment arrays wrapped in a small dict-like object and
# returns a boolean mask. Because the homecage group is not part of the
# Timepoint x Treatment cross, define it with its own condition (edit the
# right-hand side to match how homecage rows are actually coded in your file,
# e.g. Timepoint == 'hc', or a dedicated 'Group' column, etc.).
#
# IMPORTANT: adjust the 'homecage' condition below to match your data coding.
# Two common cases are shown; keep only the one that matches your df:
#   (a) Timepoint has a distinct value, e.g. df['Timepoint'] == 'hc'
#   (b) Treatment has a distinct value, e.g. df['Treatment'] == 'hc'
# ---------------------------------------------------------------------------

def get_group_definitions():
    """Return the list of (group_name, mask_fn) pairs defining all 5 groups.

    mask_fn takes a dict with keys 'Timepoint' and 'Treatment' (numpy arrays,
    which may be real or shuffled) and returns a boolean mask.
    """
    return [
        ("rec_vei", lambda a: (a['Timepoint'] == 'rec') & (a['Treatment'] == 'vei')),
        ("rec_mif", lambda a: (a['Timepoint'] == 'rec') & (a['Treatment'] == 'mif')),
        ("rem_vei", lambda a: (a['Timepoint'] == 'rem') & (a['Treatment'] == 'vei')),
        ("rem_mif", lambda a: (a['Timepoint'] == 'rem') & (a['Treatment'] == 'mif')),
        # Homecage: neither rec/rem nor mif/vei. Edit this condition to match
        # your actual coding (e.g. Timepoint == 'hc', or Treatment == 'hc').
        ("homecage", lambda a: (a['Timepoint'] == 'hc')),
    ]


def compute_spearman_matrix(data, brain_regions):
    """
    Compute Spearman correlation matrix with pairwise deletion of NaNs.

    Parameters:
    -----------
    data : ndarray
        Data matrix (samples x brain regions)
    brain_regions : list
        Names of brain regions (for reference)

    Returns:
    --------
    corr_matrix : ndarray
        Spearman correlation matrix
    """
    n_regions = data.shape[1]
    corr_matrix = np.ones((n_regions, n_regions))

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            x = data[:, i]
            y = data[:, j]

            # Pairwise deletion: use only valid pairs
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            n_valid = np.sum(valid_mask)

            if n_valid > 2:  # Need at least 3 observations
                rho, _ = spearmanr(x[valid_mask], y[valid_mask])
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho
            else:
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan

    return corr_matrix


def extract_upper_triangle_corrs(corr_matrix):
    """Extract upper triangle correlations (excluding diagonal), removing NaNs."""
    n = corr_matrix.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    corrs = corr_matrix[upper_idx]
    return corrs[~np.isnan(corrs)]


def compute_real_matrices(df, brain_regions, group_definitions):
    """Compute correlation matrices for all real groups (now 5 groups)."""
    print("\n" + "=" * 70)
    print("COMPUTING REAL CORRELATION MATRICES")
    print("=" * 70)

    real_matrices = {}
    all_real_corrs = []

    arrays = {'Timepoint': df['Timepoint'].values, 'Treatment': df['Treatment'].values}

    for group_key, mask_fn in group_definitions:
        mask = mask_fn(arrays)
        group_data = df.loc[mask, brain_regions].values

        # Compute correlation matrix
        corr_matrix = compute_spearman_matrix(group_data, brain_regions)
        real_matrices[group_key] = corr_matrix

        # Extract correlations
        corrs = extract_upper_triangle_corrs(corr_matrix)
        all_real_corrs.extend(corrs)

        print(f"\n{group_key}: n={np.sum(mask)}")
        if len(corrs) > 0:
            print(f"  Valid correlations: {len(corrs)}")
            print(f"  Range: [{np.min(corrs):.3f}, {np.max(corrs):.3f}]")
        else:
            print("  Valid correlations: 0 (check group size / NaNs)")

    all_real_corrs = np.array(all_real_corrs)
    print(f"\nTotal real correlations: {len(all_real_corrs)}")
    print(f"Statistics:")
    print(f"  Mean: {np.mean(all_real_corrs):.4f}")
    print(f"  Median: {np.median(all_real_corrs):.4f}")
    print(f"  70th percentile: {np.percentile(all_real_corrs, 70):.4f}")

    return real_matrices, all_real_corrs


def permutation_test(df, brain_regions, group_definitions, n_iterations=1000, random_seed=42):
    """
    Perform permutation testing by shuffling group labels.

    Parameters:
    -----------
    df : DataFrame
        Full dataset
    brain_regions : list
        Brain region column names
    group_definitions : list
        List of (group_name, mask_fn) pairs, e.g. from get_group_definitions()
    n_iterations : int
        Number of permutations
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    all_random_corrs : ndarray
        All correlation coefficients from random matrices
    percentiles : dict
        Percentile values of the null distribution
    """
    np.random.seed(random_seed)

    print("\n" + "=" * 70)
    print(f"PERMUTATION TESTING ({n_iterations} iterations)")
    print("=" * 70)
    print(f"Random seed: {random_seed}")
    print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Pre-extract brain region data
    brain_data = df[brain_regions].values
    n_samples = len(df)

    all_random_corrs = []

    start_time = time.time()

    for iteration in range(n_iterations):
        # Progress update every 50 iterations
        if (iteration + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (iteration + 1) / elapsed
            remaining = (n_iterations - iteration - 1) / rate
            print(f"  Iteration {iteration + 1}/{n_iterations} "
                  f"({100*(iteration+1)/n_iterations:.1f}%) - "
                  f"Est. remaining: {remaining/60:.1f} min")

        # Shuffle group labels (Timepoint and Treatment shuffled together,
        # same permutation index, so co-occurrence structure across the two
        # label columns for real (non-homecage) rows is preserved under H0)
        shuffle_idx = np.random.permutation(n_samples)
        shuffled_arrays = {
            'Timepoint': df['Timepoint'].values[shuffle_idx],
            'Treatment': df['Treatment'].values[shuffle_idx],
        }

        # Compute matrices for all 5 random groups
        for group_key, mask_fn in group_definitions:
            mask = mask_fn(shuffled_arrays)
            group_data = brain_data[mask, :]

            # Compute correlation matrix
            corr_matrix = compute_spearman_matrix(group_data, brain_regions)

            # Extract and store correlations
            corrs = extract_upper_triangle_corrs(corr_matrix)
            all_random_corrs.extend(corrs)

    elapsed_total = time.time() - start_time
    print(f"\n✓ Permutation testing completed in {elapsed_total/60:.2f} minutes")

    all_random_corrs = np.array(all_random_corrs)

    # Calculate percentiles
    percentiles = {
        '25th': np.percentile(all_random_corrs, 25),
        '50th': np.percentile(all_random_corrs, 50),
        '70th': np.percentile(all_random_corrs, 70),
        '75th': np.percentile(all_random_corrs, 75),
        '90th': np.percentile(all_random_corrs, 90),
        '95th': np.percentile(all_random_corrs, 95),
        '99th': np.percentile(all_random_corrs, 99),
    }

    print(f"\nNull distribution statistics:")
    print(f"  Total correlations: {len(all_random_corrs)}")
    print(f"  Mean: {np.mean(all_random_corrs):.4f}")
    print(f"  Std: {np.std(all_random_corrs):.4f}")
    print(f"  Range: [{np.min(all_random_corrs):.3f}, {np.max(all_random_corrs):.3f}]")
    print(f"\n  Percentiles:")
    for pct_name, pct_value in percentiles.items():
        print(f"    {pct_name}: {pct_value:.4f}")

    return all_random_corrs, percentiles


def bootstrap_comparison_test(real_corrs, random_corrs, n_bootstrap=1000, random_seed=42):
    """
    Perform statistical comparison between real and random correlations
    using bootstrap subsampling and two-sample tests.
    (Unchanged from the original 4-group version — operates on the pooled
    real_corrs / random_corrs arrays regardless of how many groups fed them.)
    """
    np.random.seed(random_seed)

    print("\n" + "=" * 70)
    print(f"STATISTICAL COMPARISON (Bootstrap n={n_bootstrap})")
    print("=" * 70)
    print(f"Real correlations: n={len(real_corrs)}")
    print(f"Random correlations: n={len(random_corrs)}")

    n_real = len(real_corrs)

    bootstrap_mean_diffs = []
    bootstrap_median_diffs = []
    bootstrap_mw_pvals = []
    bootstrap_ks_pvals = []
    bootstrap_ttest_pvals = []

    start_time = time.time()

    for iteration in range(n_bootstrap):
        if (iteration + 1) % 100 == 0:
            print(f"  Bootstrap iteration {iteration + 1}/{n_bootstrap}...")

        random_subsample = np.random.choice(random_corrs, size=n_real, replace=True)

        mean_diff = np.mean(real_corrs) - np.mean(random_subsample)
        median_diff = np.median(real_corrs) - np.median(random_subsample)

        bootstrap_mean_diffs.append(mean_diff)
        bootstrap_median_diffs.append(median_diff)

        mw_stat, mw_pval = mannwhitneyu(real_corrs, random_subsample, alternative='two-sided')
        bootstrap_mw_pvals.append(mw_pval)

        ks_stat, ks_pval = ks_2samp(real_corrs, random_subsample)
        bootstrap_ks_pvals.append(ks_pval)

        t_stat, t_pval = ttest_ind(real_corrs, random_subsample)
        bootstrap_ttest_pvals.append(t_pval)

    elapsed = time.time() - start_time
    print(f"\n✓ Bootstrap comparison completed in {elapsed:.2f} seconds")

    bootstrap_mean_diffs = np.array(bootstrap_mean_diffs)
    bootstrap_median_diffs = np.array(bootstrap_median_diffs)
    bootstrap_mw_pvals = np.array(bootstrap_mw_pvals)
    bootstrap_ks_pvals = np.array(bootstrap_ks_pvals)
    bootstrap_ttest_pvals = np.array(bootstrap_ttest_pvals)

    test_results = {
        'mean_diff': {
            'observed': np.mean(real_corrs) - np.mean(random_corrs),
            'bootstrap_mean': np.mean(bootstrap_mean_diffs),
            'bootstrap_std': np.std(bootstrap_mean_diffs),
            'ci_95': np.percentile(bootstrap_mean_diffs, [2.5, 97.5]),
        },
        'median_diff': {
            'observed': np.median(real_corrs) - np.median(random_corrs),
            'bootstrap_mean': np.mean(bootstrap_median_diffs),
            'bootstrap_std': np.std(bootstrap_median_diffs),
            'ci_95': np.percentile(bootstrap_median_diffs, [2.5, 97.5]),
        },
        'mann_whitney': {
            'median_pval': np.median(bootstrap_mw_pvals),
            'mean_pval': np.mean(bootstrap_mw_pvals),
            'pval_ci_95': np.percentile(bootstrap_mw_pvals, [2.5, 97.5]),
            'prop_significant': np.mean(bootstrap_mw_pvals < 0.05),
        },
        'kolmogorov_smirnov': {
            'median_pval': np.median(bootstrap_ks_pvals),
            'mean_pval': np.mean(bootstrap_ks_pvals),
            'pval_ci_95': np.percentile(bootstrap_ks_pvals, [2.5, 97.5]),
            'prop_significant': np.mean(bootstrap_ks_pvals < 0.05),
        },
        'ttest': {
            'median_pval': np.median(bootstrap_ttest_pvals),
            'mean_pval': np.mean(bootstrap_ttest_pvals),
            'pval_ci_95': np.percentile(bootstrap_ttest_pvals, [2.5, 97.5]),
            'prop_significant': np.mean(bootstrap_ttest_pvals < 0.05),
        },
    }

    bootstrap_data = {
        'mean_diffs': bootstrap_mean_diffs,
        'median_diffs': bootstrap_median_diffs,
        'mw_pvals': bootstrap_mw_pvals,
        'ks_pvals': bootstrap_ks_pvals,
        'ttest_pvals': bootstrap_ttest_pvals,
    }

    print("\n" + "-" * 70)
    print("STATISTICAL TEST RESULTS")
    print("-" * 70)

    print(f"\nMean Difference (Real - Random):")
    print(f"  Observed: {test_results['mean_diff']['observed']:.4f}")
    print(f"  Bootstrap mean: {test_results['mean_diff']['bootstrap_mean']:.4f}")
    print(f"  Bootstrap std: {test_results['mean_diff']['bootstrap_std']:.4f}")
    print(f"  95% CI: [{test_results['mean_diff']['ci_95'][0]:.4f}, {test_results['mean_diff']['ci_95'][1]:.4f}]")

    print(f"\nMedian Difference (Real - Random):")
    print(f"  Observed: {test_results['median_diff']['observed']:.4f}")
    print(f"  Bootstrap mean: {test_results['median_diff']['bootstrap_mean']:.4f}")
    print(f"  Bootstrap std: {test_results['median_diff']['bootstrap_std']:.4f}")
    print(f"  95% CI: [{test_results['median_diff']['ci_95'][0]:.4f}, {test_results['median_diff']['ci_95'][1]:.4f}]")

    print(f"\nMann-Whitney U Test (non-parametric):")
    print(f"  Median p-value: {test_results['mann_whitney']['median_pval']:.4f}")
    print(f"  Proportion significant (p<0.05): {test_results['mann_whitney']['prop_significant']:.2%}")

    print(f"\nKolmogorov-Smirnov Test (distribution):")
    print(f"  Median p-value: {test_results['kolmogorov_smirnov']['median_pval']:.4f}")
    print(f"  Proportion significant (p<0.05): {test_results['kolmogorov_smirnov']['prop_significant']:.2%}")

    print(f"\nIndependent t-test (parametric):")
    print(f"  Median p-value: {test_results['ttest']['median_pval']:.4f}")
    print(f"  Proportion significant (p<0.05): {test_results['ttest']['prop_significant']:.2%}")

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if test_results['mean_diff']['ci_95'][0] > 0:
        print("✓ Real correlations are SIGNIFICANTLY HIGHER than random (mean)")
    elif test_results['mean_diff']['ci_95'][1] < 0:
        print("✓ Real correlations are SIGNIFICANTLY LOWER than random (mean)")
    else:
        print("  No significant difference in means (CI includes 0)")

    if test_results['mann_whitney']['prop_significant'] > 0.95:
        print("✓ Strong evidence of difference (Mann-Whitney: >95% significant)")
    elif test_results['mann_whitney']['prop_significant'] > 0.80:
        print("  Moderate evidence of difference (Mann-Whitney: >80% significant)")
    else:
        print("  Weak evidence of difference (Mann-Whitney)")

    if test_results['kolmogorov_smirnov']['prop_significant'] > 0.95:
        print("✓ Distributions are significantly different (KS test: >95% significant)")

    return test_results, bootstrap_data


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Permutation testing for functional connectivity threshold'
    )

    parser.add_argument('--input', type=str, default='allv2.xlsx',
                         help='Input Excel file (default: allv1.xlsx)')
    parser.add_argument('--iterations', type=int, default=10000,
                         help='Number of permutation iterations (default: 10000)')
    parser.add_argument('--bootstrap', type=int, default=10000,
                         help='Number of bootstrap iterations for statistical tests (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                         help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='thresholding/permutation_results.txt',
                         help='Output results file (default: permutation_results.txt)')
    parser.add_argument('--no-stats', action='store_true',
                         help='Skip statistical comparison (faster)')
    parser.add_argument('--no-plots', action='store_true',
                         help='Skip generating plots')

    args = parser.parse_args()

    print("=" * 70)
    print("FUNCTIONAL CONNECTIVITY PERMUTATION ANALYSIS")
    print("=" * 70)
    print(f"\nInput file: {args.input}")
    print(f"Permutation iterations: {args.iterations}")
    print(f"Bootstrap iterations: {args.bootstrap}")
    print(f"Random seed: {args.seed}")
    print(f"Output file: {args.output}")

    # Load data
    print(f"\nLoading data from {args.input}...")
    df = pd.read_excel(args.input)

    brain_regions = df.columns[2:].tolist()

    print(f"Loaded: {len(df)} samples, {len(brain_regions)} brain regions")
    print(f"Brain regions: {brain_regions}")

    # Group definitions now include the homecage group as a 5th entry
    group_definitions = get_group_definitions()

    # Get group info (works for any number of groups)
    arrays = {'Timepoint': df['Timepoint'].values, 'Treatment': df['Treatment'].values}
    group_sizes = {}
    for group_key, mask_fn in group_definitions:
        group_sizes[group_key] = int(np.sum(mask_fn(arrays)))

    print("\nGroup sizes:")
    for group, size in group_sizes.items():
        print(f"  {group}: {size}")

    # Compute real matrices (5 groups)
    real_matrices, real_corrs = compute_real_matrices(df, brain_regions, group_definitions)

    # Perform permutation testing (5 groups per iteration)
    random_corrs, percentiles = permutation_test(
        df, brain_regions, group_definitions,
        n_iterations=args.iterations,
        random_seed=args.seed
    )

    # Perform statistical comparison
    if not args.no_stats:
        test_results, bootstrap_data = bootstrap_comparison_test(
            real_corrs, random_corrs,
            n_bootstrap=args.bootstrap,
            random_seed=args.seed
        )
    else:
        test_results = None
        bootstrap_data = None
        print("\n⊘ Skipping statistical comparison (--no-stats flag)")

    df_info = {
        'n_samples': len(df),
        'n_regions': len(brain_regions),
        'brain_regions': brain_regions,
        'group_sizes': group_sizes,
        'n_iterations': args.iterations,
        'n_bootstrap': args.bootstrap if not args.no_stats else 0,
    }

    # Save correlation distributions to CSV (real_corrs now pools all 5 groups)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        'real_correlations': pd.Series(real_corrs),
        'random_correlations': pd.Series(random_corrs),
    })
    csv_file = args.output.replace('.txt', '_distributions.csv')
    results_df.to_csv(csv_file, index=False)
    print(f"✓ Correlation distributions saved to: {csv_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n*** RECOMMENDED THRESHOLD: {percentiles['70th']:.4f} ***\n")


if __name__ == "__main__":
    main()
