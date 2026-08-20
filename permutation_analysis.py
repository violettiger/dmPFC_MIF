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

Author: Generated for neuroscience postdoc research
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


def compute_spearman_matrix(data, brain_regions):
    """
    Compute Spearman correlation matrix with pairwise deletion of NaNs.
    
    Parameters:
    -----------
    data : ndarray
        Data matrix (samples × brain regions)
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


def compute_real_matrices(df, brain_regions):
    """Compute correlation matrices for all real groups."""
    print("\n" + "="*70)
    print("COMPUTING REAL CORRELATION MATRICES")
    print("="*70)
    
    real_matrices = {}
    all_real_corrs = []
    
    for timepoint in ['rec', 'rem']:
        for treatment in ['vei', 'mif']:
            group_key = f"{timepoint}_{treatment}"
            
            # Extract group data
            mask = (df['Timepoint'] == timepoint) & (df['Treatment'] == treatment)
            group_data = df.loc[mask, brain_regions].values
            
            # Compute correlation matrix
            corr_matrix = compute_spearman_matrix(group_data, brain_regions)
            real_matrices[group_key] = corr_matrix
            
            # Extract correlations
            corrs = extract_upper_triangle_corrs(corr_matrix)
            all_real_corrs.extend(corrs)
            
            print(f"\n{group_key}: n={np.sum(mask)}")
            print(f"  Valid correlations: {len(corrs)}")
            print(f"  Range: [{np.min(corrs):.3f}, {np.max(corrs):.3f}]")
    
    all_real_corrs = np.array(all_real_corrs)
    print(f"\nTotal real correlations: {len(all_real_corrs)}")
    print(f"Statistics:")
    print(f"  Mean: {np.mean(all_real_corrs):.4f}")
    print(f"  Median: {np.median(all_real_corrs):.4f}")
    print(f"  70th percentile: {np.percentile(all_real_corrs, 70):.4f}")
    
    return real_matrices, all_real_corrs


def permutation_test(df, brain_regions, n_iterations=1000, random_seed=42):
    """
    Perform permutation testing by shuffling group labels.
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    brain_regions : list
        Brain region column names
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
    
    print("\n" + "="*70)
    print(f"PERMUTATION TESTING ({n_iterations} iterations)")
    print("="*70)
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
        
        # Shuffle group labels
        shuffle_idx = np.random.permutation(n_samples)
        timepoints_shuffled = df['Timepoint'].values[shuffle_idx]
        treatments_shuffled = df['Treatment'].values[shuffle_idx]
        
        # Compute matrices for all 4 random groups
        for timepoint in ['rec', 'rem']:
            for treatment in ['vei', 'mif']:
                # Get shuffled group data
                mask = (timepoints_shuffled == timepoint) & (treatments_shuffled == treatment)
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
        '99th': np.percentile(all_random_corrs, 99)
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
    
    Parameters:
    -----------
    real_corrs : ndarray
        Real correlation coefficients
    random_corrs : ndarray
        Random correlation coefficients from permutation
    n_bootstrap : int
        Number of bootstrap iterations
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    test_results : dict
        Dictionary containing test statistics and p-values
    bootstrap_data : dict
        Raw bootstrap data for plotting
    """
    np.random.seed(random_seed)
    
    print("\n" + "="*70)
    print(f"STATISTICAL COMPARISON (Bootstrap n={n_bootstrap})")
    print("="*70)
    print(f"Real correlations: n={len(real_corrs)}")
    print(f"Random correlations: n={len(random_corrs)}")
    
    # Sample sizes
    n_real = len(real_corrs)
    
    # Store bootstrap results
    bootstrap_mean_diffs = []
    bootstrap_median_diffs = []
    bootstrap_mw_pvals = []
    bootstrap_ks_pvals = []
    bootstrap_ttest_pvals = []
    
    start_time = time.time()
    
    for iteration in range(n_bootstrap):
        if (iteration + 1) % 100 == 0:
            print(f"  Bootstrap iteration {iteration + 1}/{n_bootstrap}...")
        
        # Subsample from random correlations (same size as real)
        random_subsample = np.random.choice(random_corrs, size=n_real, replace=True)
        
        # Compute statistics
        mean_diff = np.mean(real_corrs) - np.mean(random_subsample)
        median_diff = np.median(real_corrs) - np.median(random_subsample)
        
        bootstrap_mean_diffs.append(mean_diff)
        bootstrap_median_diffs.append(median_diff)
        
        # Mann-Whitney U test (non-parametric)
        mw_stat, mw_pval = mannwhitneyu(real_corrs, random_subsample, alternative='two-sided')
        bootstrap_mw_pvals.append(mw_pval)
        
        # Kolmogorov-Smirnov test (distribution comparison)
        ks_stat, ks_pval = ks_2samp(real_corrs, random_subsample)
        bootstrap_ks_pvals.append(ks_pval)
        
        # Independent t-test (parametric, for comparison)
        t_stat, t_pval = ttest_ind(real_corrs, random_subsample)
        bootstrap_ttest_pvals.append(t_pval)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Bootstrap comparison completed in {elapsed:.2f} seconds")
    
    # Convert to arrays
    bootstrap_mean_diffs = np.array(bootstrap_mean_diffs)
    bootstrap_median_diffs = np.array(bootstrap_median_diffs)
    bootstrap_mw_pvals = np.array(bootstrap_mw_pvals)
    bootstrap_ks_pvals = np.array(bootstrap_ks_pvals)
    bootstrap_ttest_pvals = np.array(bootstrap_ttest_pvals)
    
    # Compile results
    test_results = {
        'mean_diff': {
            'observed': np.mean(real_corrs) - np.mean(random_corrs),
            'bootstrap_mean': np.mean(bootstrap_mean_diffs),
            'bootstrap_std': np.std(bootstrap_mean_diffs),
            'ci_95': np.percentile(bootstrap_mean_diffs, [2.5, 97.5])
        },
        'median_diff': {
            'observed': np.median(real_corrs) - np.median(random_corrs),
            'bootstrap_mean': np.mean(bootstrap_median_diffs),
            'bootstrap_std': np.std(bootstrap_median_diffs),
            'ci_95': np.percentile(bootstrap_median_diffs, [2.5, 97.5])
        },
        'mann_whitney': {
            'median_pval': np.median(bootstrap_mw_pvals),
            'mean_pval': np.mean(bootstrap_mw_pvals),
            'pval_ci_95': np.percentile(bootstrap_mw_pvals, [2.5, 97.5]),
            'prop_significant': np.mean(bootstrap_mw_pvals < 0.05)
        },
        'kolmogorov_smirnov': {
            'median_pval': np.median(bootstrap_ks_pvals),
            'mean_pval': np.mean(bootstrap_ks_pvals),
            'pval_ci_95': np.percentile(bootstrap_ks_pvals, [2.5, 97.5]),
            'prop_significant': np.mean(bootstrap_ks_pvals < 0.05)
        },
        'ttest': {
            'median_pval': np.median(bootstrap_ttest_pvals),
            'mean_pval': np.mean(bootstrap_ttest_pvals),
            'pval_ci_95': np.percentile(bootstrap_ttest_pvals, [2.5, 97.5]),
            'prop_significant': np.mean(bootstrap_ttest_pvals < 0.05)
        }
    }
    
    # Store bootstrap data for plotting
    bootstrap_data = {
        'mean_diffs': bootstrap_mean_diffs,
        'median_diffs': bootstrap_median_diffs,
        'mw_pvals': bootstrap_mw_pvals,
        'ks_pvals': bootstrap_ks_pvals,
        'ttest_pvals': bootstrap_ttest_pvals
    }
    
    # Print summary
    print("\n" + "-"*70)
    print("STATISTICAL TEST RESULTS")
    print("-"*70)
    
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
    print(f"  Mean p-value: {test_results['mann_whitney']['mean_pval']:.4f}")
    print(f"  95% CI: [{test_results['mann_whitney']['pval_ci_95'][0]:.4f}, {test_results['mann_whitney']['pval_ci_95'][1]:.4f}]")
    print(f"  Proportion significant (p<0.05): {test_results['mann_whitney']['prop_significant']:.2%}")
    
    print(f"\nKolmogorov-Smirnov Test (distribution):")
    print(f"  Median p-value: {test_results['kolmogorov_smirnov']['median_pval']:.4f}")
    print(f"  Mean p-value: {test_results['kolmogorov_smirnov']['mean_pval']:.4f}")
    print(f"  95% CI: [{test_results['kolmogorov_smirnov']['pval_ci_95'][0]:.4f}, {test_results['kolmogorov_smirnov']['pval_ci_95'][1]:.4f}]")
    print(f"  Proportion significant (p<0.05): {test_results['kolmogorov_smirnov']['prop_significant']:.2%}")
    
    print(f"\nIndependent t-test (parametric):")
    print(f"  Median p-value: {test_results['ttest']['median_pval']:.4f}")
    print(f"  Mean p-value: {test_results['ttest']['mean_pval']:.4f}")
    print(f"  95% CI: [{test_results['ttest']['pval_ci_95'][0]:.4f}, {test_results['ttest']['pval_ci_95'][1]:.4f}]")
    print(f"  Proportion significant (p<0.05): {test_results['ttest']['prop_significant']:.2%}")
    
    # Overall interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION")
    print("-"*70)
    
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


def plot_results(real_corrs, random_corrs, percentiles, test_results, bootstrap_data, output_prefix):
    """
    Create comprehensive visualization of permutation and statistical test results.
    
    Parameters:
    -----------
    real_corrs : ndarray
        Real correlation coefficients
    random_corrs : ndarray
        Random correlation coefficients
    percentiles : dict
        Percentile values from permutation test
    test_results : dict
        Statistical test results
    bootstrap_data : dict
        Bootstrap raw data
    output_prefix : str
        Prefix for output filenames
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Distribution comparison (main plot - spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot histograms
    ax1.hist(random_corrs, bins=100, alpha=0.5, label='Null (Random)', 
             color='gray', density=True, edgecolor='black', linewidth=0.5)
    ax1.hist(real_corrs, bins=50, alpha=0.7, label='Real Data', 
             color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for percentiles
    ax1.axvline(percentiles['70th'], color='blue', linestyle='--', linewidth=2, 
                label=f"70th percentile: {percentiles['70th']:.3f}")
    ax1.axvline(np.mean(real_corrs), color='red', linestyle='-', linewidth=2, 
                label=f"Real mean: {np.mean(real_corrs):.3f}")
    ax1.axvline(np.mean(random_corrs), color='black', linestyle='-', linewidth=2, 
                label=f"Random mean: {np.mean(random_corrs):.3f}")
    
    ax1.set_xlabel('Spearman Correlation Coefficient (ρ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution Comparison: Real vs Random Correlations', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative distributions
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Sort data for CDF
    real_sorted = np.sort(real_corrs)
    random_sorted = np.sort(random_corrs)
    real_cdf = np.arange(1, len(real_sorted)+1) / len(real_sorted)
    random_cdf = np.arange(1, len(random_sorted)+1) / len(random_sorted)
    
    ax2.plot(random_sorted, random_cdf, color='gray', linewidth=2, label='Null (Random)', alpha=0.7)
    ax2.plot(real_sorted, real_cdf, color='#e74c3c', linewidth=2, label='Real Data')
    ax2.axvline(percentiles['70th'], color='blue', linestyle='--', linewidth=1.5)
    
    ax2.set_xlabel('Correlation Coefficient', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Distributions', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Bootstrap mean differences
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.hist(bootstrap_data['mean_diffs'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax3.axvline(test_results['mean_diff']['observed'], color='red', linestyle='-', 
                linewidth=2, label=f"Observed: {test_results['mean_diff']['observed']:.4f}")
    ax3.axvline(test_results['mean_diff']['ci_95'][0], color='orange', linestyle='--', 
                linewidth=1.5, label='95% CI')
    ax3.axvline(test_results['mean_diff']['ci_95'][1], color='orange', linestyle='--', linewidth=1.5)
    ax3.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Mean Difference (Real - Random)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax3.set_title('Bootstrap: Mean Differences', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Bootstrap median differences
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.hist(bootstrap_data['median_diffs'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.axvline(test_results['median_diff']['observed'], color='red', linestyle='-', 
                linewidth=2, label=f"Observed: {test_results['median_diff']['observed']:.4f}")
    ax4.axvline(test_results['median_diff']['ci_95'][0], color='orange', linestyle='--', 
                linewidth=1.5, label='95% CI')
    ax4.axvline(test_results['median_diff']['ci_95'][1], color='orange', linestyle='--', linewidth=1.5)
    ax4.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax4.set_xlabel('Median Difference (Real - Random)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax4.set_title('Bootstrap: Median Differences', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Mann-Whitney p-values
    ax5 = fig.add_subplot(gs[1, 2])
    
    ax5.hist(bootstrap_data['mw_pvals'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax5.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax5.axvline(test_results['mann_whitney']['median_pval'], color='blue', linestyle='-', 
                linewidth=2, label=f"Median: {test_results['mann_whitney']['median_pval']:.4f}")
    
    ax5.set_xlabel('p-value', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax5.set_title(f"Mann-Whitney U Test\n({test_results['mann_whitney']['prop_significant']:.1%} significant)", 
                  fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Kolmogorov-Smirnov p-values
    ax6 = fig.add_subplot(gs[2, 0])
    
    ax6.hist(bootstrap_data['ks_pvals'], bins=50, color='#e67e22', alpha=0.7, edgecolor='black')
    ax6.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax6.axvline(test_results['kolmogorov_smirnov']['median_pval'], color='blue', linestyle='-', 
                linewidth=2, label=f"Median: {test_results['kolmogorov_smirnov']['median_pval']:.4f}")
    
    ax6.set_xlabel('p-value', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax6.set_title(f"Kolmogorov-Smirnov Test\n({test_results['kolmogorov_smirnov']['prop_significant']:.1%} significant)", 
                  fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. t-test p-values
    ax7 = fig.add_subplot(gs[2, 1])
    
    ax7.hist(bootstrap_data['ttest_pvals'], bins=50, color='#f39c12', alpha=0.7, edgecolor='black')
    ax7.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax7.axvline(test_results['ttest']['median_pval'], color='blue', linestyle='-', 
                linewidth=2, label=f"Median: {test_results['ttest']['median_pval']:.4f}")
    
    ax7.set_xlabel('p-value', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax7.set_title(f"Independent t-test\n({test_results['ttest']['prop_significant']:.1%} significant)", 
                  fontsize=11, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary statistics table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    
    Real Correlations:
      n = {len(real_corrs)}
      Mean = {np.mean(real_corrs):.4f}
      Median = {np.median(real_corrs):.4f}
      SD = {np.std(real_corrs):.4f}
    
    Random Correlations:
      n = {len(random_corrs)}
      Mean = {np.mean(random_corrs):.4f}
      Median = {np.median(random_corrs):.4f}
      SD = {np.std(random_corrs):.4f}
    
    Threshold (70th %ile): {percentiles['70th']:.4f}
    
    Statistical Tests:
      Mann-Whitney: {test_results['mann_whitney']['prop_significant']:.1%} sig
      KS Test: {test_results['kolmogorov_smirnov']['prop_significant']:.1%} sig
      t-test: {test_results['ttest']['prop_significant']:.1%} sig
    """
    
    ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle('Functional Connectivity Permutation Analysis: Statistical Validation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_file1 = f"{output_prefix}_comprehensive_analysis.png"
    output_file_eps1 = f"{output_prefix}_comprehensive_analysis.eps"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file_eps1, bbox_inches='tight', facecolor='white')
    print(f"✓ Comprehensive analysis plot saved: {output_file1}")
    print(f"✓ Comprehensive analysis plot (EPS) saved: {output_file_eps1}")
    
    plt.close()
    
    # Create second figure: Detailed p-value comparisons
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    tests = ['Mann-Whitney U', 'Kolmogorov-Smirnov', 'Independent t-test']
    pval_data = [bootstrap_data['mw_pvals'], bootstrap_data['ks_pvals'], bootstrap_data['ttest_pvals']]
    colors = ['#2ecc71', '#e67e22', '#f39c12']
    
    for idx, (ax, test_name, pvals, color) in enumerate(zip(axes, tests, pval_data, colors)):
        # Box plot
        bp = ax.boxplot([pvals], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Add significance line
        ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05', alpha=0.7)
        
        # Add scatter of individual points (subsample for visibility)
        if len(pvals) > 100:
            sample_idx = np.random.choice(len(pvals), 100, replace=False)
            sample_pvals = pvals[sample_idx]
        else:
            sample_pvals = pvals
        
        ax.scatter(np.ones(len(sample_pvals)) + np.random.normal(0, 0.02, len(sample_pvals)), 
                   sample_pvals, alpha=0.3, s=20, color=color)
        
        ax.set_ylabel('p-value', fontsize=12, fontweight='bold')
        ax.set_title(f'{test_name}\n{np.mean(pvals < 0.05):.1%} significant', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(-0.05, 1.05)
        
        if idx == 0:
            ax.legend(fontsize=10)
    
    fig2.suptitle('Bootstrap P-value Distributions Across Statistical Tests', 
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file2 = f"{output_prefix}_pvalue_comparison.png"
    output_file_eps2 = f"{output_prefix}_pvalue_comparison.eps"

    plt.savefig(output_file2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_file_eps2, bbox_inches="tight", facecolor="white")
    print(f"✓ P-value comparison plot saved: {output_file2}")
    print(f"✓ P-value comparison plot (EPS) saved: {output_file_eps2}")
    
    plt.close()
    
    print("✓ All visualizations generated successfully!")


def save_results(output_file, real_corrs, random_corrs, percentiles, test_results, df_info):
    """Save results to text file."""
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FUNCTIONAL CONNECTIVITY PERMUTATION ANALYSIS RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset: {df_info['n_samples']} samples, {df_info['n_regions']} brain regions\n")
        f.write(f"Brain regions: {df_info['brain_regions']}\n\n")
        
        f.write("Group sizes:\n")
        for group, size in df_info['group_sizes'].items():
            f.write(f"  {group}: {size}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("REAL DATA CORRELATIONS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total correlations: {len(real_corrs)}\n")
        f.write(f"Mean: {np.mean(real_corrs):.4f}\n")
        f.write(f"Median: {np.median(real_corrs):.4f}\n")
        f.write(f"Std: {np.std(real_corrs):.4f}\n")
        f.write(f"25th percentile: {np.percentile(real_corrs, 25):.4f}\n")
        f.write(f"75th percentile: {np.percentile(real_corrs, 75):.4f}\n")
        f.write(f"Range: [{np.min(real_corrs):.4f}, {np.max(real_corrs):.4f}]\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("NULL DISTRIBUTION (PERMUTED DATA)\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of permutations: {df_info['n_iterations']}\n")
        f.write(f"Total random correlations: {len(random_corrs)}\n")
        f.write(f"Mean: {np.mean(random_corrs):.4f}\n")
        f.write(f"Median: {np.median(random_corrs):.4f}\n")
        f.write(f"Std: {np.std(random_corrs):.4f}\n")
        f.write(f"Range: [{np.min(random_corrs):.4f}, {np.max(random_corrs):.4f}]\n")
        f.write("\nPercentiles:\n")
        for pct_name, pct_value in percentiles.items():
            f.write(f"  {pct_name}: {pct_value:.4f}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("STATISTICAL COMPARISON (Bootstrap Test Results)\n")
        f.write("-"*70 + "\n")
        f.write(f"Bootstrap iterations: {df_info['n_bootstrap']}\n\n")
        
        f.write("Mean Difference (Real - Random):\n")
        f.write(f"  Observed: {test_results['mean_diff']['observed']:.4f}\n")
        f.write(f"  Bootstrap mean: {test_results['mean_diff']['bootstrap_mean']:.4f}\n")
        f.write(f"  Bootstrap std: {test_results['mean_diff']['bootstrap_std']:.4f}\n")
        f.write(f"  95% CI: [{test_results['mean_diff']['ci_95'][0]:.4f}, {test_results['mean_diff']['ci_95'][1]:.4f}]\n\n")
        
        f.write("Median Difference (Real - Random):\n")
        f.write(f"  Observed: {test_results['median_diff']['observed']:.4f}\n")
        f.write(f"  Bootstrap mean: {test_results['median_diff']['bootstrap_mean']:.4f}\n")
        f.write(f"  Bootstrap std: {test_results['median_diff']['bootstrap_std']:.4f}\n")
        f.write(f"  95% CI: [{test_results['median_diff']['ci_95'][0]:.4f}, {test_results['median_diff']['ci_95'][1]:.4f}]\n\n")
        
        f.write("Mann-Whitney U Test:\n")
        f.write(f"  Median p-value: {test_results['mann_whitney']['median_pval']:.4f}\n")
        f.write(f"  Mean p-value: {test_results['mann_whitney']['mean_pval']:.4f}\n")
        f.write(f"  Proportion significant (p<0.05): {test_results['mann_whitney']['prop_significant']:.2%}\n\n")
        
        f.write("Kolmogorov-Smirnov Test:\n")
        f.write(f"  Median p-value: {test_results['kolmogorov_smirnov']['median_pval']:.4f}\n")
        f.write(f"  Mean p-value: {test_results['kolmogorov_smirnov']['mean_pval']:.4f}\n")
        f.write(f"  Proportion significant (p<0.05): {test_results['kolmogorov_smirnov']['prop_significant']:.2%}\n\n")
        
        f.write("Independent t-test:\n")
        f.write(f"  Median p-value: {test_results['ttest']['median_pval']:.4f}\n")
        f.write(f"  Mean p-value: {test_results['ttest']['mean_pval']:.4f}\n")
        f.write(f"  Proportion significant (p<0.05): {test_results['ttest']['prop_significant']:.2%}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDED THRESHOLD FOR GRAPH BINARIZATION\n")
        f.write("="*70 + "\n")
        f.write(f"70th percentile of null distribution: {percentiles['70th']:.4f}\n\n")
        f.write("This threshold represents the value above which only 30% of random\n")
        f.write("correlations occur by chance. Edges with correlations above this\n")
        f.write("threshold can be considered statistically meaningful.\n")
    
    print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Permutation testing for functional connectivity threshold'
    )
    parser.add_argument('--input', type=str, default='allv1.xlsx',
                        help='Input Excel file (default: allv1.xlsx)')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of permutation iterations (default: 1000)')
    parser.add_argument('--bootstrap', type=int, default=10000,
                        help='Number of bootstrap iterations for statistical tests (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='thresholding/permutation_results.txt',
                        help='Output results file (default: permutation_results.txt)')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip statistical comparison (faster)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FUNCTIONAL CONNECTIVITY PERMUTATION ANALYSIS")
    print("="*70)
    print(f"\nInput file: {args.input}")
    print(f"Permutation iterations: {args.iterations}")
    print(f"Bootstrap iterations: {args.bootstrap}")
    print(f"Random seed: {args.seed}")
    print(f"Output file: {args.output}")
    print(f"Statistical comparison: {'No' if args.no_stats else 'Yes'}")
    print(f"Generate plots: {'No' if args.no_plots else 'Yes'}")
    
    # Load data
    print(f"\nLoading data from {args.input}...")
    df = pd.read_excel(args.input)
    
    brain_regions = df.columns[2:].tolist()
    
    print(f"Loaded: {len(df)} samples, {len(brain_regions)} brain regions")
    print(f"Brain regions: {brain_regions}")
    
    # Get group info
    group_sizes = {}
    for tp in ['rec', 'rem']:
        for tr in ['vei', 'mif']:
            key = f"{tp}_{tr}"
            size = len(df[(df['Timepoint'] == tp) & (df['Treatment'] == tr)])
            group_sizes[key] = size
    
    print("\nGroup sizes:")
    for group, size in group_sizes.items():
        print(f"  {group}: {size}")
    
    # Compute real matrices
    real_matrices, real_corrs = compute_real_matrices(df, brain_regions)
    
    # Perform permutation testing
    random_corrs, percentiles = permutation_test(
        df, brain_regions, 
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
    
    # Prepare info for saving
    df_info = {
        'n_samples': len(df),
        'n_regions': len(brain_regions),
        'brain_regions': brain_regions,
        'group_sizes': group_sizes,
        'n_iterations': args.iterations,
        'n_bootstrap': args.bootstrap if not args.no_stats else 0
    }
    
    # Save results
    if test_results is not None:
        save_results(args.output, real_corrs, random_corrs, percentiles, test_results, df_info)
    else:
        # Save without test results
        with open(args.output, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FUNCTIONAL CONNECTIVITY PERMUTATION ANALYSIS RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write(f"70th percentile threshold: {percentiles['70th']:.4f}\n")
            f.write("\nFull results available by running without --no-stats flag.\n")
        print(f"\n✓ Results saved to: {args.output}")
    
    # Save correlation distributions to CSV
    results_df = pd.DataFrame({
        'real_correlations': pd.Series(real_corrs),
        'random_correlations': pd.Series(random_corrs)
    })
    csv_file = args.output.replace('.txt', '_distributions.csv')
    results_df.to_csv(csv_file, index=False)
    print(f"✓ Correlation distributions saved to: {csv_file}")
    
    # Save bootstrap test results if available
    if test_results is not None:
        test_summary_df = pd.DataFrame({
            'Test': ['Mean Difference', 'Median Difference', 'Mann-Whitney U', 
                     'Kolmogorov-Smirnov', 'Independent t-test'],
            'Statistic': [
                test_results['mean_diff']['observed'],
                test_results['median_diff']['observed'],
                test_results['mann_whitney']['median_pval'],
                test_results['kolmogorov_smirnov']['median_pval'],
                test_results['ttest']['median_pval']
            ],
            'CI_Lower': [
                test_results['mean_diff']['ci_95'][0],
                test_results['median_diff']['ci_95'][0],
                test_results['mann_whitney']['pval_ci_95'][0],
                test_results['kolmogorov_smirnov']['pval_ci_95'][0],
                test_results['ttest']['pval_ci_95'][0]
            ],
            'CI_Upper': [
                test_results['mean_diff']['ci_95'][1],
                test_results['median_diff']['ci_95'][1],
                test_results['mann_whitney']['pval_ci_95'][1],
                test_results['kolmogorov_smirnov']['pval_ci_95'][1],
                test_results['ttest']['pval_ci_95'][1]
            ],
            'Prop_Significant': [
                np.nan,
                np.nan,
                test_results['mann_whitney']['prop_significant'],
                test_results['kolmogorov_smirnov']['prop_significant'],
                test_results['ttest']['prop_significant']
            ]
        })
        test_csv = args.output.replace('.txt', '_statistical_tests.csv')
        test_summary_df.to_csv(test_csv, index=False)
        print(f"✓ Statistical test summary saved to: {test_csv}")
    
    # Generate plots
    if not args.no_plots and test_results is not None:
        output_prefix = args.output.replace('.txt', '')
        plot_results(real_corrs, random_corrs, percentiles, test_results, 
                    bootstrap_data, output_prefix)
    elif args.no_plots:
        print("\n⊘ Skipping plot generation (--no-plots flag)")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n*** RECOMMENDED THRESHOLD: {percentiles['70th']:.4f} ***\n")
    print("Use this threshold to binarize your functional connectivity graphs.")
    print("Edges with |correlation| >= threshold are considered significant.\n")


if __name__ == "__main__":
    main()
