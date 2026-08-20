"""
BRAIN FUNCTIONAL CONNECTIVITY ANALYSIS - COMPLETE VERSION WITH FISHER R-TO-Z TRANSFORMATION

Complete features:
1. Animal-level permutation tests (proper statistical inference)
2. Hub analysis with participation coefficient and within-module z-score
3. Community detection with modularity optimization
4. Statistical comparison of network metrics using real animal resampling
5. SPEARMAN CORRELATION (rank-based, more robust)
6. Multiple comparison correction with FDR
7. ORIGINAL network visualization style (with detailed legends)
8. P-VALUE THRESHOLDING (p < 0.05 for Spearman correlations)
9. FLEXIBLE THRESHOLDING MODE (coefficient-based OR p-value-based OR combined)
10. CORRELATION MATRIX PLOTTING (both groups)
11. FIXED THRESHOLDING & COMMUNITY DETECTION
12. COMPREHENSIVE NODE-LEVEL METRICS PLOTTING
13. FISHER R-TO-Z TRANSFORMATION FOR CORRELATION MATRIX COMPARISON
14. GLOBAL METRICS 
15. Z-SCORE BASED HUB CLASSIFICATION WITH Z-SCORE PLOTTING
"""

import numpy as np
print(np.__version__)
import pandas as pd
print(pd.__version__)
import networkx as nx
print(nx.__version__)
import igraph as ig
print(ig.__version__)
import matplotlib
print(matplotlib.__version__)   
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
print(sns.__version__)
import scipy
print(scipy.__version__)
from scipy import stats
from scipy.stats import spearmanr, ttest_ind, ttest_1samp, norm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')
import os
from collections import defaultdict
print("collections version:", defaultdict.__module__)
import community.community_louvain as community_louvain

# Set random seed for reproducible results
np.random.seed(42)

# ORIGINAL VISUALIZATION PARAMETERS AND FUNCTIONS
R_MIN, R_MAX = 0.50, 0.90  # correlation thresholds for width scaling
W_MIN, W_MAX = 1.0, 5.0     # line widths for min/max |rho|
ABS_THRESHOLD = 0.5         # do not plot edges with |rho| < this

# Fixed node order for consistent positioning across all graphs
FIXED_NODE_ORDER = ['aIC', 'CeA', 'BLA', 'dCA1', 'dDG', 'dCA3', 'vCA1', 'vDG', 'vCA3', 'ACC', 'aRSC', 'PrL']

def width_from_r(r, rmin=R_MIN, rmax=R_MAX, wmin=W_MIN, wmax=W_MAX):
    """Map |r| to a line width with linear scaling and clamping."""
    a = abs(r)
    a = max(rmin, min(a, rmax))
    return wmin + (a - rmin) * (wmax - wmin) / (rmax - rmin + 1e-9)

def create_fixed_positions(node_list, fixed_order=FIXED_NODE_ORDER):
    """Create fixed circular positions for nodes to ensure consistent layout"""
    # Use the fixed order if all nodes are present, otherwise use sorted order
    available_nodes = [node for node in fixed_order if node in node_list]
    remaining_nodes = sorted([node for node in node_list if node not in fixed_order])
    ordered_nodes = available_nodes + remaining_nodes
    
    # Create circular layout with fixed positions
    n = len(ordered_nodes)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    positions = {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(ordered_nodes, angles)}
    
    return positions, ordered_nodes

def fisher_r_to_z(r):
    """
    Convert Pearson/Spearman correlation coefficient to z-score using Fisher transformation.
    
    Formula: z = 0.5 * ln((1+r)/(1-r))
    
    Parameters:
    -----------
    r : float or array
        Correlation coefficient(s) between -1 and 1
    
    Returns:
    --------
    z : float or array
        Fisher z-transformed value(s)
    """
    # Clip to prevent numerical issues
    r_clipped = np.clip(r, -0.9999, 0.9999)
    z = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
    return z

def calculate_z_score_matrix(corr_matrix_1, corr_matrix_2, n1, n2):
    """
    Calculate z-scores comparing two correlation matrices using Fisher transformation.
    
    For each correlation pair (i,j), computes:
        z = (z1 - z2) / sqrt(1/(n1-3) + 1/(n2-3))
        p-value is two-tailed: p = 2 * (1 - norm.cdf(|z|))
    
    Parameters:
    -----------
    corr_matrix_1 : 2D array
        Correlation matrix from group 1
    corr_matrix_2 : 2D array
        Correlation matrix from group 2
    n1 : int
        Sample size of group 1
    n2 : int
        Sample size of group 2
    
    Returns:
    --------
    z_scores : 2D array
        Z-scores for each correlation pair comparison
    p_values : 2D array
        Two-tailed p-values for each z-score
    sig_mask : 2D boolean array
        True where p < 0.05, False otherwise
    """
    # Transform correlations to z-scores
    z1 = fisher_r_to_z(corr_matrix_1)
    z2 = fisher_r_to_z(corr_matrix_2)
    
    # Calculate difference in z-scores
    z_diff = z1 - z2
    
    # Calculate standard error
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    
    # Calculate z-statistic for the difference
    z_scores = z_diff / se
    
    # Calculate two-tailed p-values
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
    
    # Create significance mask (p < 0.05)
    sig_mask = p_values < 0.05
    
    return z_scores, p_values, sig_mask

def plot_z_score_matrix(z_scores, p_values, sig_mask, brain_regions,
                        group1_name, group2_name, output_dir="."):
    """
    Plot Fisher z-score matrix as heatmap with significance markers.
    
    Parameters:
    -----------
    z_scores : 2D array
        Z-scores from calculate_z_score_matrix()
    p_values : 2D array
        P-values from calculate_z_score_matrix()
    sig_mask : 2D boolean array
        Significance mask from calculate_z_score_matrix()
    brain_regions : list
        List of region names
    group1_name : str
        Name of group 1
    group2_name : str
        Name of group 2
    output_dir : str
        Directory for saving figures
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(z_scores, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(z_scores,
                mask=~mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-3, vmax=3,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                xticklabels=brain_regions,
                yticklabels=brain_regions,
                ax=ax)
    
    # Mark significant comparisons with black boxes
    for i in range(len(brain_regions)):
        for j in range(i+1, len(brain_regions)):
            if sig_mask[i, j]:  # p < 0.05
                rect = plt.Rectangle((j, i), 1, 1, fill=False,
                                   edgecolor='black', linewidth=3)
                ax.add_patch(rect)
    
    ax.set_title(f'Fisher Z-Score Matrix: {group2_name.upper()} vs {group1_name.upper()}\n'
                f'(Black boxes: p < 0.05)')
    plt.tight_layout()
    
    # Save figures
    plt.savefig(os.path.join(output_dir, 'fisher_z_score_matrix.png'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fisher_z_score_matrix.eps'),
               format='eps', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(corr_matrix, brain_regions, group_name, output_dir,
                           threshold=0.4, significance_mask=None, threshold_type="coefficient"):
    """Plot correlation matrix with optional significance highlighting based on threshold type"""
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(corr_matrix,
                mask=~mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                xticklabels=brain_regions,
                yticklabels=brain_regions,
                ax=ax)
    
    # Highlight correlations above threshold
    for i in range(len(brain_regions)):
        for j in range(i+1, len(brain_regions)):
            highlight = False
            
            # Determine if this correlation should be highlighted
            if threshold_type == "pvalue" and significance_mask is not None:
                highlight = significance_mask[i, j]
            elif threshold_type == "coefficient":
                highlight = corr_matrix[i, j] >= threshold
            elif threshold_type == "both" and significance_mask is not None:
                # BOTH: OR logic (pass if EITHER criterion met)
                coeff_passes = corr_matrix[i, j] >= threshold
                pval_passes = significance_mask[i, j]
                highlight = coeff_passes or pval_passes
            elif threshold_type == "strict" and significance_mask is not None:
                # STRICT: AND logic (pass only if BOTH criteria met)
                coeff_passes = corr_matrix[i, j] >= threshold
                pval_passes = significance_mask[i, j]
                highlight = coeff_passes and pval_passes
            
            if highlight:
                rect = plt.Rectangle((j, i), 1, 1, fill=False,
                                   edgecolor='black', linewidth=3)
                ax.add_patch(rect)
    
    threshold_label = f"p < {threshold}" if threshold_type == "pvalue" else f"|ρ| ≥ {threshold}"
    ax.set_title(f'{group_name.upper()} Group Correlation Matrix\nSpearman Correlation ({threshold_label} highlighted)')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{group_name}_correlation_matrix.png'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{group_name}_correlation_matrix.eps'),
               format='eps', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_graph_by_coeff(G, partition=None, title='', filename='graph_coeffs',
                            output_dir=".", width_params=(R_MIN, R_MAX, W_MIN, W_MAX),
                            pos_color='red', neg_color='blue'):
    """Visualize graph with edge widths based on correlation coefficients and colors for positive/negative correlations."""
    rmin, rmax, wmin, wmax = width_params
    
    # Convert partition to a dictionary if it's not already
    if partition is not None and not isinstance(partition, dict):
        partition = {node: community for node, community in zip(G.nodes(), partition)}
    
    # Create node name mapping
    node_names = {}
    node_list = []
    for node in G.nodes():
        if hasattr(G.nodes[node], 'get') and 'name' in G.nodes[node]:
            node_names[node] = G.nodes[node]['name']
            node_list.append(G.nodes[node]['name'])
        else:
            node_names[node] = str(node)
            node_list.append(str(node))
    
    # Create fixed positions for consistent layout
    positions, ordered_nodes = create_fixed_positions(node_list)
    
    # Map positions back to actual node indices
    node_positions = {}
    for node in G.nodes():
        node_name = node_names[node]
        if node_name in positions:
            node_positions[node] = positions[node_name]
        else:
            node_positions[node] = (0, 0)
    
    # Node sizes based on degree
    degrees = dict(G.degree())
    min_degree = 0
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = []
    for node in G.nodes():
        degree = degrees.get(node, 0)
        if max_degree == min_degree:
            size = 800
        else:
            size = 400 + (degree - min_degree) * (1200 - 400) / (max_degree - min_degree)
        node_sizes.append(size)
    
    # Separate positive and negative edges
    pos_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('weight', 0) >= 0]
    neg_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('weight', 0) < 0]
    
    def widths(edata):
        return [width_from_r(d.get('weight', 0), rmin, rmax, wmin, wmax) for _, _, d in edata]
    
    # Create figure with ORIGINAL layout proportions
    fig = plt.figure(figsize=(14, 10))
    ax_main = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    
    # Draw nodes with community colors if partition is provided
    if partition is not None:
        node_colors = []
        for node in G.nodes():
            node_name = node_names[node]
            if node_name in partition:
                node_colors.append(partition[node_name])
            else:
                node_colors.append(0)
        nx.draw_networkx_nodes(G, node_positions,
                             node_size=node_sizes, node_color=node_colors,
                             cmap=plt.cm.tab20, edgecolors='k', linewidths=0.5, ax=ax_main)
    else:
        nx.draw_networkx_nodes(G, node_positions,
                             node_size=node_sizes, node_color='skyblue',
                             edgecolors='k', linewidths=0.5, ax=ax_main)
    
    # Draw node labels
    labels = {node: node_names[node] for node in G.nodes()}
    nx.draw_networkx_labels(G, node_positions, labels, font_size=12, ax=ax_main)
    
    # Draw edges
    if pos_edges:
        nx.draw_networkx_edges(G, node_positions,
                             edgelist=[(u, v) for u, v, _ in pos_edges],
                             width=widths(pos_edges), edge_color=pos_color, alpha=0.85, ax=ax_main)
    
    if neg_edges:
        nx.draw_networkx_edges(G, node_positions,
                             edgelist=[(u, v) for u, v, _ in neg_edges],
                             width=widths(neg_edges), edge_color=neg_color, alpha=0.85, ax=ax_main)
    
    ax_main.set_title(title, fontsize=14)
    ax_main.axis('off')
    
    # Legend subplot (right side) - ORIGINAL format
    ax_legend = plt.subplot2grid((1, 3), (0, 2))
    ax_legend.axis('off')
    
    legend_y_pos = 0.95
    
    # Edge thickness legend
    ax_legend.text(0.1, legend_y_pos, 'Edge Thickness (|ρ|):', fontweight='bold', fontsize=12, transform=ax_legend.transAxes)
    legend_y_pos -= 0.08
    
    thickness_examples = [0.50, 0.70, 0.90]
    for i, rho in enumerate(thickness_examples):
        width = width_from_r(rho, rmin, rmax, wmin, wmax)
        ax_legend.plot([0.15, 0.45], [legend_y_pos - i*0.05, legend_y_pos - i*0.05],
                      color='black', linewidth=width, alpha=0.8)
        ax_legend.text(0.5, legend_y_pos - i*0.05, f'ρ = {rho}',
                      va='center', fontsize=10, transform=ax_legend.transAxes)
    
    legend_y_pos -= 0.2
    
    # Edge color legend
    ax_legend.text(0.1, legend_y_pos, 'Edge Colors:', fontweight='bold', fontsize=12, transform=ax_legend.transAxes)
    legend_y_pos -= 0.05
    ax_legend.plot([0.15, 0.35], [legend_y_pos, legend_y_pos],
                  color=pos_color, linewidth=3, alpha=0.8)
    ax_legend.text(0.4, legend_y_pos, 'Positive correlation',
                  va='center', fontsize=10, transform=ax_legend.transAxes)
    legend_y_pos -= 0.05
    ax_legend.plot([0.15, 0.35], [legend_y_pos, legend_y_pos],
                  color=neg_color, linewidth=3, alpha=0.8)
    ax_legend.text(0.4, legend_y_pos, 'Negative correlation',
                  va='center', fontsize=10, transform=ax_legend.transAxes)
    
    legend_y_pos -= 0.15
    
    # Node size legend
    ax_legend.text(0.1, legend_y_pos, 'Node Size (# edges):', fontweight='bold', fontsize=12, transform=ax_legend.transAxes)
    legend_y_pos -= 0.08
    
    if max_degree > 0:
        size_examples = [0, max_degree//2 if max_degree > 1 else 1, max_degree]
        size_labels = ['Low connectivity', 'Medium connectivity', 'High connectivity']
        
        for i, (degree, label) in enumerate(zip(size_examples, size_labels)):
            if max_degree == min_degree:
                size = 800
            else:
                size = 400 + (degree - min_degree) * (1200 - 400) / (max_degree - min_degree)
            display_size = size / 20
            circle = plt.Circle((0.25, legend_y_pos - i*0.08), display_size/1000,
                              color='lightblue', ec='black', linewidth=0.5)
            ax_legend.add_patch(circle)
            ax_legend.text(0.35, legend_y_pos - i*0.08, f'{label} ({degree} edges)',
                         va='center', fontsize=9, transform=ax_legend.transAxes)
    
    legend_y_pos -= 0.25
    
    # Community colors legend
    if partition is not None:
        unique_communities = sorted(set(partition.values()))
        if len(unique_communities) > 1:
            ax_legend.text(0.1, legend_y_pos, 'Communities:', fontweight='bold', fontsize=12, transform=ax_legend.transAxes)
            legend_y_pos -= 0.05
            
            cmap = plt.cm.tab20
            for i, community in enumerate(unique_communities):
                if i >= 10:
                    ax_legend.text(0.15, legend_y_pos - i*0.04, f'... and more',
                                 fontsize=9, transform=ax_legend.transAxes)
                    break
                
                color = cmap(community / max(unique_communities)) if max(unique_communities) > 0 else cmap(0)
                circle = plt.Circle((0.2, legend_y_pos - i*0.04), 0.015,
                                  color=color, ec='black', linewidth=0.5)
                ax_legend.add_patch(circle)
                
                community_nodes = [node_name for node_name, comm in partition.items() if comm == community]
                ax_legend.text(0.25, legend_y_pos - i*0.04, f'Community {community+1}: {", ".join(community_nodes)}',
                             va='center', fontsize=8, transform=ax_legend.transAxes)
    
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{filename}.eps'), format='eps', dpi=300, bbox_inches='tight')
    plt.close()

# ======== HELPER FUNCTIONS FOR EXTENDED METRICS ========

def calculate_small_worldness(G):
    """
    Calculate small-world propensity following Telesford et al. (2011)
    Small-worldness combines clustering and path length relative to random networks
    """
    if G.number_of_edges() < 3 or G.number_of_nodes() < 3:
        return np.nan
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # Calculate observed metrics
    C_obs = nx.average_clustering(G, weight='abs_weight')
    if nx.is_connected(G):
        L_obs = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) > 1:
            G_cc = G.subgraph(largest_cc)
            L_obs = nx.average_shortest_path_length(G_cc)
        else:
            return np.nan
    
    # Generate random network with same degree distribution
    try:
        G_random = nx.configuration_model([d for n, d in G.degree()])
        G_random = nx.Graph(G_random)  # simple undirected graph
        G_random.remove_edges_from(nx.selfloop_edges(G_random))
        
        C_random = nx.average_clustering(G_random)
        for u, v in G_random.edges():
            G_random[u][v]['abs_weight'] = 1.0
        C_random = nx.average_clustering(G_random, weight='abs_weight')
        
        if nx.is_connected(G_random):
            L_random = nx.average_shortest_path_length(G_random)
        else:
            largest_cc_random = max(nx.connected_components(G_random), key=len)
            if len(largest_cc_random) > 1:
                G_random_cc = G_random.subgraph(largest_cc_random)
                L_random = nx.average_shortest_path_length(G_random_cc)
            else:
                return np.nan
        
        # Small-world propensity: SW = (gamma / lambda)
        if L_random > 0 and C_random > 0:
            gamma = C_obs / C_random
            lambda_ = L_obs / L_random
            small_worldness = gamma / lambda_
            return small_worldness
        else:
            return np.nan
    except:
        return np.nan

class BrainNetworkAnalyzer:
    """
    Comprehensive brain network analysis following established methodology
    WITH SPEARMAN CORRELATION, ABSOLUTE & P-VALUE THRESHOLDING, COMPLETE PLOTTING,
    AND FISHER R-TO-Z TRANSFORMATION
    
    UPDATED: 6 global metrics (removed transitivity, characteristic_path_length, density; added modularity)
    UPDATED: Z-score based hub classification with z-score plotting
    """
    
    def __init__(self, data_file, groups=['vei', 'mif'],
                 correlation_threshold=0.4, p_value_threshold=0.05,
                 threshold_method="coefficient"):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        data_file : str
            Path to Excel file with expression data
        groups : list
            Group names (e.g., ['vei', 'mif'])
        correlation_threshold : float
            Absolute correlation coefficient threshold (e.g., 0.563)
        p_value_threshold : float
            P-value threshold for Spearman correlation (e.g., 0.05)
        threshold_method : str
            "coefficient" - use absolute correlation threshold only
            "pvalue" - use p-value threshold only
            "both" - use both (edge included if passes either criterion)
            "strict" - use both (edge included only if passes both criteria)
        """
        self.data_file = data_file
        self.groups = groups
        self.correlation_threshold = correlation_threshold
        self.p_value_threshold = p_value_threshold
        self.threshold_method = threshold_method
        self.brain_regions = None
        self.data = None
        self.individual_data = {}
        self.group_correlations = {}
        self.group_pvalues = {}  # NEW: Store p-values
        self.group_significance_masks = {}  # NEW: Store significance masks
        self.networks = {}
        self.network_metrics = {}
        self.community_assignments = {}
        self.community_modularity = {}  # Store modularity for each group
        self.hub_classifications = {}
        self.fisher_z_scores = None
        self.fisher_p_values = None
        self.fisher_sig_mask = None
        
        print(f"\nThreshold method: {threshold_method}")
        print(f"Correlation threshold: {correlation_threshold}")
        print(f"P-value threshold: {p_value_threshold}")
    
    def load_data(self):
        """Load and preprocess data"""
        print("Loading data...")
        self.data = pd.read_excel(self.data_file)
        
        # Identify brain regions
        self.brain_regions = [col for col in self.data.columns
                            if col not in ['Treatment', 'Animal']]
        
        # Extract individual animal data for each group
        for group in self.groups:
            group_data = self.data[self.data['Treatment'] == group]
            self.individual_data[group] = group_data[self.brain_regions].values
            print(f"{group.upper()}: {len(group_data)} subjects, {len(self.brain_regions)} regions")
    
    def compute_functional_connectivity(self):
        """
        Compute functional connectivity matrices using SPEARMAN correlations
        NEW: Also compute p-values for each correlation
        """
        print("\nComputing functional connectivity matrices using SPEARMAN correlation...")
        
        for group in self.groups:
            data_matrix = self.individual_data[group]
            n_regions = len(self.brain_regions)
            n_subjects = data_matrix.shape[0]
            
            print(f"  {group.upper()}: Computing correlations for {n_subjects} subjects x {n_regions} regions...")
            
            # Initialize correlation and p-value matrices
            corr_matrix = np.zeros((n_regions, n_regions))
            pval_matrix = np.ones((n_regions, n_regions))
            
            # Compute Spearman correlations pairwise
            for i in range(n_regions):
                for j in range(n_regions):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                        pval_matrix[i, j] = 1.0
                    else:
                        region_i_data = data_matrix[:, i]
                        region_j_data = data_matrix[:, j]
                        valid_indices = ~(np.isnan(region_i_data) | np.isnan(region_j_data))
                        
                        if np.sum(valid_indices) >= 3:
                            r, p = spearmanr(region_i_data[valid_indices],
                                           region_j_data[valid_indices],
                                           alternative='greater')
                            corr_matrix[i, j] = r if not np.isnan(r) else 0.0
                            pval_matrix[i, j] = p if not np.isnan(p) else 1.0
                        else:
                            corr_matrix[i, j] = 0.0
                            pval_matrix[i, j] = 1.0
            
            self.group_correlations[group] = corr_matrix
            self.group_pvalues[group] = pval_matrix
            
            # Create significance mask (p < threshold)
            pval_for_mask = np.where(np.isnan(pval_matrix), 1.0, pval_matrix)
            significance_mask = pval_for_mask < self.p_value_threshold
            self.group_significance_masks[group] = significance_mask
            
            # Print diagnostics
            off_diagonal_indices = np.triu_indices(n_regions, k=1)
            off_diagonal = corr_matrix[off_diagonal_indices]
            pval_off_diagonal = pval_matrix[off_diagonal_indices]
            
            mean_abs_corr = np.mean(off_diagonal)
            above_coeff_threshold = np.sum(off_diagonal >= self.correlation_threshold)
            significant_pvals = np.sum(pval_off_diagonal < self.p_value_threshold)
            total_pairs = len(off_diagonal)
            
            print(f"  {group.upper()}: Spearman correlation matrix computed")
            print(f"    Mean |correlation|: {mean_abs_corr:.3f}")
            print(f"    Correlations ≥ {self.correlation_threshold}: {above_coeff_threshold}/{total_pairs} ({100*above_coeff_threshold/total_pairs:.1f}%)")
            print(f"    Significant correlations (p < {self.p_value_threshold}): {significant_pvals}/{total_pairs} ({100*significant_pvals/total_pairs:.1f}%)")
            print(f"    Range: [{np.min(off_diagonal):.3f}, {np.max(off_diagonal):.3f}]")
    
    def plot_correlation_matrices(self, output_dir):
        """Plot correlation matrices for both groups"""
        print("\nPlotting correlation matrices...")
        for group in self.groups:
            corr_matrix = self.group_correlations[group]
            significance_mask = self.group_significance_masks[group]
            
            plot_correlation_matrix(corr_matrix, self.brain_regions, group, output_dir,
                                  threshold=self.correlation_threshold,
                                  significance_mask=significance_mask,
                                  threshold_type=self.threshold_method)
            print(f"  {group.upper()} correlation matrix saved")
    
    def compare_correlation_matrices_fisher(self, output_dir='.'):
        """
        NEW: Compare correlation matrices using Fisher r-to-z transformation.
        Identifies which functional connections significantly differ between groups.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("FISHER R-TO-Z TRANSFORMATION: Comparing Correlation Matrices")
        print("="*80)
        
        # Get correlation matrices and sample sizes
        corr1 = self.group_correlations[self.groups[0]]
        corr2 = self.group_correlations[self.groups[1]]
        n1 = len(self.individual_data[self.groups[0]])
        n2 = len(self.individual_data[self.groups[1]])
        
        print(f"\nComparing: {self.groups[0].upper()} (n={n1}) vs {self.groups[1].upper()} (n={n2})")
        
        # Calculate z-scores
        z_scores, p_values, sig_mask = calculate_z_score_matrix(corr1, corr2, n1, n2)
        
        # Store results
        self.fisher_z_scores = z_scores
        self.fisher_p_values = p_values
        self.fisher_sig_mask = sig_mask
        
        # Print diagnostics
        off_diag_idx = np.triu_indices(len(self.brain_regions), k=1)
        sig_count = np.sum(sig_mask[off_diag_idx])
        total_pairs = len(off_diag_idx[0])
        
        print(f"\nSignificant differences (p < 0.05): {sig_count}/{total_pairs}")
        
        if sig_count > 0:
            # Extract significant pairs
            sig_pairs = []
            for i, j in zip(off_diag_idx[0], off_diag_idx[1]):
                if sig_mask[i, j]:
                    sig_pairs.append({
                        'Region_A': self.brain_regions[i],
                        'Region_B': self.brain_regions[j],
                        'Corr_Group1': corr1[i, j],
                        'Corr_Group2': corr2[i, j],
                        'Z_Score': z_scores[i, j],
                        'P_Value': p_values[i, j],
                        'Direction': 'Stronger in ' + self.groups[1].upper() if z_scores[i, j] < 0 else 'Stronger in ' + self.groups[0].upper()
                    })
            
            # Sort by absolute z-score
            sig_pairs.sort(key=lambda x: abs(x['Z_Score']), reverse=True)
            
            # Print top results
            print(f"\nTop 10 Most Significant Differences:")
            print(f"{'Region A':8} {'Region B':8} {self.groups[0].upper():8} {self.groups[1].upper():8} {'Z-Score':10} {'P-Value':10} {'Direction':30}")
            print("-" * 90)
            for pair in sig_pairs[:10]:
                print(f"{pair['Region_A']:8} {pair['Region_B']:8} "
                     f"{pair['Corr_Group1']:8.3f} {pair['Corr_Group2']:8.3f} "
                     f"{pair['Z_Score']:10.3f} {pair['P_Value']:10.4f} {pair['Direction']:30}")
            
            # Save results to CSV
            sig_df = pd.DataFrame(sig_pairs)
            sig_df.to_csv(os.path.join(output_dir, 'fisher_significant_pairs.csv'), index=False)
            print(f"\nSaved significant pairs to fisher_significant_pairs.csv")
        
        # Save full z-score matrix
        z_df = pd.DataFrame(z_scores, index=self.brain_regions, columns=self.brain_regions)
        z_df.to_csv(os.path.join(output_dir, 'fisher_z_score_matrix.csv'))
        
        # Save p-value matrix
        p_df = pd.DataFrame(p_values, index=self.brain_regions, columns=self.brain_regions)
        p_df.to_csv(os.path.join(output_dir, 'fisher_p_value_matrix.csv'))
        
        # Plot visualization
        plot_z_score_matrix(z_scores, p_values, sig_mask, self.brain_regions,
                          self.groups[0], self.groups[1], output_dir)
        
        print(f"Saved Fisher transformation results to CSV and PNG/EPS files")
        print("="*80)
        
        return z_scores, p_values, sig_mask
    
    def permutation_test_correlation_differences(self, n_permutations=1000, random_state=42, output_dir="."):
        """
        Permutation test for correlation differences between groups.
        Works with your DataFrame structure where:
        - Rows = individual animals
        - Columns = brain regions
        - No explicit animal IDs needed
        """
        np.random.seed(random_state)
        
        print("\n" + "="*80)
        print("STEP 8.5: PERMUTATION VALIDATION OF FISHER TRANSFORMATION")
        print("="*80)
        print(f"Testing correlation differences with {n_permutations} permutations")
        
        # Get basic info
        n_regions = len(self.brain_regions)
        
        # Get individual data for both groups
        arr_group1 = self.individual_data[self.groups[0]]
        df_group1 = pd.DataFrame(arr_group1, columns=self.brain_regions)
        
        arr_group2 = self.individual_data[self.groups[1]]
        df_group2 = pd.DataFrame(arr_group2, columns=self.brain_regions)
        
        n_group1 = len(df_group1)
        n_group2 = len(df_group2)
        n_total = n_group1 + n_group2
        
        print(f"\nSetup:")
        print(f"  Group 1 ({self.groups[0]}): n = {n_group1}")
        print(f"  Group 2 ({self.groups[1]}): n = {n_group2}")
        print(f"  Total animals: n = {n_total}")
        print(f"  Region pairs: {n_regions * (n_regions - 1) // 2}")
        
        # Initialize storage
        permutation_pvalues = np.zeros((n_regions, n_regions))
        permutation_results = {}
        
        # Get the observed correlation matrices
        corr1 = self.group_correlations[self.groups[0]]
        corr2 = self.group_correlations[self.groups[1]]
        
        # Pool all data from both groups
        pooled_data = pd.concat([df_group1, df_group2], axis=0, ignore_index=True)
        print(f"  Pooled animals for shuffling: {len(pooled_data)}")
        
        # For each region pair
        pair_count = 0
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                pair_count += 1
                region_pair = f"{self.brain_regions[i]}-{self.brain_regions[j]}"
                
                # Get observed correlation difference
                corr_group1 = corr1[i, j]
                corr_group2 = corr2[i, j]
                observed_deltar = abs(corr_group2 - corr_group1)
                
                # Generate permutation distribution
                permuted_deltar_values = []
                
                for perm in range(n_permutations):
                    # Randomly shuffle rows (animals) between groups
                    shuffled_indices = np.random.permutation(len(pooled_data))
                    
                    # Split into pseudo-groups
                    pseudo_group1_indices = shuffled_indices[:n_group1]
                    pseudo_group2_indices = shuffled_indices[n_group1:]
                    
                    # Get data for pseudo-groups
                    pseudo_df1 = pooled_data.iloc[pseudo_group1_indices]
                    pseudo_df2 = pooled_data.iloc[pseudo_group2_indices]
                    
                    # Calculate correlations for the region pair
                    pseudo_corr1 = self._calculate_correlation_from_dataframe(
                        pseudo_df1, self.brain_regions[i], self.brain_regions[j]
                    )
                    pseudo_corr2 = self._calculate_correlation_from_dataframe(
                        pseudo_df2, self.brain_regions[i], self.brain_regions[j]
                    )
                    
                    if not np.isnan(pseudo_corr1) and not np.isnan(pseudo_corr2):
                        permuted_deltar = abs(pseudo_corr2 - pseudo_corr1)
                        permuted_deltar_values.append(permuted_deltar)
                
                if len(permuted_deltar_values) == 0:
                    p_value = 1.0
                else:
                    permuted_deltar_values = np.array(permuted_deltar_values)
                    # Empirical p-value
                    p_value = (np.sum(permuted_deltar_values >= observed_deltar) + 1) / (len(permuted_deltar_values) + 1)
                
                # Store results
                permutation_pvalues[i, j] = p_value
                permutation_pvalues[j, i] = p_value
                
                permutation_results[region_pair] = {
                    'observed_deltar': observed_deltar,
                    'p_value': p_value,
                    'r_group1': corr_group1,
                    'r_group2': corr_group2
                }
                
                if pair_count % 5 == 0:
                    print(f"  Processed {pair_count}/{n_regions * (n_regions - 1) // 2} pairs...", end='\r')
        
        print(f"  Processed {pair_count}/{n_regions * (n_regions - 1) // 2} pairs... ✓\n")
        
        # Save results
        perm_pvalue_df = pd.DataFrame(
            permutation_pvalues,
            index=self.brain_regions,
            columns=self.brain_regions
        )
        output_file = os.path.join(output_dir, 'permutation_pvalue_matrix.csv')
        perm_pvalue_df.to_csv(output_file)
        print(f"✓ Saved: permutation_pvalue_matrix.csv")
        
        # Create comparison table
        comparison_results = []
        for pair_name in sorted(permutation_results.keys()):
            results = permutation_results[pair_name]
            
            # Find Fisher p-value
            regions = pair_name.split('-')
            i = self.brain_regions.index(regions[0])
            j = self.brain_regions.index(regions[1])
            
            # Try to get Fisher p-value if it exists
            fisher_pval = np.nan
            if hasattr(self, 'fisher_p_values'):
                fisher_pval = self.fisher_p_values[i, j]
            
            # Determine agreement
            perm_sig = results['p_value'] < 0.05
            fisher_sig = fisher_pval < 0.05 if not np.isnan(fisher_pval) else False
            agreement = "Yes" if perm_sig == fisher_sig else "No"
            
            comparison_results.append({
                'Region_Pair': pair_name,
                'r_Group1': f"{results['r_group1']:.4f}",
                'r_Group2': f"{results['r_group2']:.4f}",
                'Observed_Δr': f"{results['observed_deltar']:.4f}",
                'Permutation_p': f"{results['p_value']:.4f}",
                'Fisher_p': f"{fisher_pval:.4f}" if not np.isnan(fisher_pval) else "N/A",
                'Agreement_at_0.05': agreement
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_output = os.path.join(output_dir, 'fisher_vs_permutation_comparison.csv')
        comparison_df.to_csv(comparison_output, index=False)
        print(f"✓ Saved: fisher_vs_permutation_comparison.csv\n")
        
        # Print summary
        print("PERMUTATION TEST SUMMARY:")
        print("-" * 80)
        print(f"Pairs tested: {len(permutation_results)}")
        significant = sum(1 for r in permutation_results.values() if r['p_value'] < 0.05)
        print(f"Significant at p < 0.05: {significant}")
        
        if hasattr(self, 'fisher_p_values'):
            agreement_count = 0
            for pair_name in permutation_results.keys():
                regions = pair_name.split('-')
                i = self.brain_regions.index(regions[0])
                j = self.brain_regions.index(regions[1])
                fisher_p = self.fisher_p_values[i, j]
                perm_p = permutation_results[pair_name]['p_value']
                
                perm_sig = perm_p < 0.05
                fisher_sig = fisher_p < 0.05
                
                if perm_sig == fisher_sig:
                    agreement_count += 1
            
            print(f"\nAGREEMENT WITH FISHER TEST (p < 0.05 threshold):")
            print(f"  {agreement_count}/{len(permutation_results)} pairs agree " +
                 f"({agreement_count/len(permutation_results)*100:.1f}%)")
        
        print("\n" + "="*80 + "\n")
        
        return permutation_results, permutation_pvalues
    
    def _calculate_correlation_from_dataframe(self, df, region_i, region_j):
        """
        Calculate Spearman correlation between two regions from a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame where rows are animals, columns are brain regions
        region_i, region_j : str
            Names of brain regions to correlate
        
        Returns:
        --------
        correlation : float
            Spearman correlation coefficient (or NaN if insufficient data)
        """
        if region_i not in df.columns or region_j not in df.columns:
            return np.nan
        
        values_i = df[region_i].values
        values_j = df[region_j].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(values_i) | np.isnan(values_j))
        values_i = values_i[valid_mask]
        values_j = values_j[valid_mask]
        
        if len(values_i) < 3:  # Need at least 3 points for meaningful correlation
            return np.nan
        
        try:
            corr, _ = spearmanr(values_i, values_j, alternative='greater')
            return corr
        except:
            return np.nan
    
    def compare_correlation_matrices_fisher_with_validation(self, output_dir="."):
        """
        Enhanced Fisher comparison that includes permutation validation.
        
        Runs both:
        1. Fisher r-to-z transformation (parametric)
        2. Permutation test (non-parametric, empirical)
        
        Provides side-by-side comparison of both approaches.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory for output files
        
        Returns:
        --------
        permutation_results : dict
            Results from permutation test
        permutation_pvalues : np.ndarray
            Permutation p-value matrix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("FISHER TRANSFORMATION WITH PERMUTATION VALIDATION")
        print("="*80)
        
        # Step 1: Run original Fisher transformation
        print("\n[Step 1/2] Running Fisher r-to-z transformation...")
        self.compare_correlation_matrices_fisher(output_dir)
        
        # Step 2: Run permutation validation
        print("\n[Step 2/2] Running permutation validation...")
        permutation_results, permutation_pvalues = self.permutation_test_correlation_differences(
            n_permutations=1000,
            random_state=42,
            output_dir=output_dir
        )
        
        # Store for later reference
        self.permutation_results = permutation_results
        self.permutation_pvalues = permutation_pvalues
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("  [Fisher original]")
        print("    • fisher_z_score_matrix.csv")
        print("    • fisher_p_value_matrix.csv")
        print("    • fisher_significant_pairs.csv")
        print("    • fisher_z_score_matrix.png")
        print("\n  [Permutation validation - NEW]")
        print("    • permutation_pvalue_matrix.csv")
        print("    • fisher_vs_permutation_comparison.csv")
        print("\nCompare the two approaches in:")
        print("  → fisher_vs_permutation_comparison.csv")
        print("\n" + "="*80 + "\n")
        
        return permutation_results, permutation_pvalues
    
    def _should_include_edge(self, r, p):
        """
        Determine if edge should be included based on threshold method.
        Returns True if edge passes the criterion.
        """
        coeff_passes = r >= self.correlation_threshold
        pval_passes = p < self.p_value_threshold
        
        if self.threshold_method == "coefficient":
            return coeff_passes
        elif self.threshold_method == "pvalue":
            return pval_passes
        elif self.threshold_method == "both":  # OR logic
            return coeff_passes or pval_passes
        elif self.threshold_method == "strict":  # AND logic
            return coeff_passes and pval_passes
        else:
            return coeff_passes  # Default to coefficient
    
    def create_networks(self):
        """
        Create network graphs from correlation matrices.
        NEW: Uses flexible thresholding based on threshold_method parameter
        """
        print(f"\nCreating networks with {self.threshold_method.upper()} thresholding...")
        
        for group in self.groups:
            corr_matrix = self.group_correlations[group]
            pval_matrix = self.group_pvalues[group]
            n_regions = len(self.brain_regions)
            
            G = nx.Graph()
            
            # Add nodes with region names
            for i, region in enumerate(self.brain_regions):
                G.add_node(i, name=region)
            
            # Add edges based on threshold method
            edges_added = 0
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    r = corr_matrix[i, j]
                    p = pval_matrix[i, j]
                    
                    if self._should_include_edge(r, p):
                        G.add_edge(i, j, weight=r, abs_weight=abs(r))
                        edges_added += 1
            
            self.networks[group] = G
            density = nx.density(G)
            
            threshold_desc = self._get_threshold_description()
            print(f"  {group.upper()}: {G.number_of_nodes()} nodes, {edges_added} edges, density = {density:.3f}")
            print(f"    Threshold: {threshold_desc}")
    
    def _get_threshold_description(self):
        """Get human-readable description of threshold being used"""
        if self.threshold_method == "coefficient":
            return f"|ρ| ≥ {self.correlation_threshold}"
        elif self.threshold_method == "pvalue":
            return f"p < {self.p_value_threshold}"
        elif self.threshold_method == "both":
            return f"|ρ| ≥ {self.correlation_threshold} OR p < {self.p_value_threshold}"
        elif self.threshold_method == "strict":
            return f"|ρ| ≥ {self.correlation_threshold} AND p < {self.p_value_threshold}"
        else:
            return f"|ρ| ≥ {self.correlation_threshold}"
    
    def calculate_network_metrics(self):
        """Calculate comprehensive network metrics - UPDATED: 6 metrics + modularity from detect_communities"""
        print("\nCalculating network metrics...")
        
        for group in self.groups:
            G = self.networks[group]
            metrics = self.network_metrics.get(group, {})
            metrics['global'] = metrics.get('global', {})
            
            # Global metrics - UPDATED: Only 6 metrics
            metrics['global']['average_clustering'] = nx.average_clustering(G, weight='abs_weight')
            metrics['global']['small_worldness'] = calculate_small_worldness(G)
            
            # Mean degree
            degrees = [d for n, d in G.degree()]
            metrics['global']['mean_degree'] = np.mean(degrees) if len(degrees) > 0 else 0
            
            # Mean strength
            strengths = []
            for node in G.nodes():
                strength = sum([G[node][neighbor].get('abs_weight', 1.0) for neighbor in G.neighbors(node)])
                strengths.append(strength)
            metrics['global']['mean_strength'] = np.mean(strengths) if len(strengths) > 0 else 0          
            metrics['global']['global_efficiency'] = nx.global_efficiency(G)
           
            # ADD MODULARITY FROM detect_communities (will be set after detect_communities is called)
            # Initialize as NaN - will be updated after detect_communities
            metrics['global'].setdefault('modularity', np.nan)
            
            self.network_metrics[group] = metrics
            print(f"  {group.upper()}: Network metrics calculated (6 metrics)")
    
    def _detect_communities_from_graph(self, G, brain_regions=None):
        """
        Helper method to detect communities from a NetworkX graph.
        This implements the same multi-method approach as detect_communities()
        but works on a provided graph rather than self.networks.
        
        Parameters
        ----------
        G : nx.Graph
            NetworkX graph to detect communities in
        brain_regions : list, optional
            List of brain region names. If None, uses node indices.
        
        Returns
        -------
        tuple
            (community_assignment, modularity, method_used)
            - community_assignment: dict mapping node to community_id
            - modularity: float, modularity score
            - method_used: str, name of method that was used
        """
        if G.number_of_edges() == 0:
            if brain_regions is not None:
                community_assignment = {brain_regions[node]: node for node in G.nodes()}
            else:
                community_assignment = {node: node for node in G.nodes()}
            return community_assignment, 0.0, "no_edges"
        
        community_assignment = None
        modularity = -1
        method_used = "none"
        
        # Method 1: Try Leiden algorithm
        # try:
        #     edge_list = []
        #     weights = []
        #     node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        #     reverse_mapping = {idx: node for node, idx in node_mapping.items()}
            
        #     for u, v, data in G.edges(data=True):
        #         edge_list.append((node_mapping[u], node_mapping[v]))
        #         # Try multiple weight attribute names
        #         weight = data.get('abs_weight', data.get('weight', 1.0))
        #         weights.append(weight)
            
        #     if len(edge_list) > 0:
        #         g = ig.Graph()
        #         g.add_vertices(len(G.nodes()))
        #         g.add_edges(edge_list)
                
        #         for resolution in [1.0, 0.8, 0.6, 1.2]:
        #             try:
        #                 partition = g.community_leiden(weights=weights, resolution=resolution,
        #                                               beta=0.01, n_iterations=-2)
                        
        #                 if len(set(partition.membership)) > 1 and partition.modularity > modularity:
        #                     temp_assignment = {}
        #                     for ig_node, community in enumerate(partition.membership):
        #                         nx_node = reverse_mapping[ig_node]
        #                         if brain_regions is not None:
        #                             node_name = brain_regions[nx_node]
        #                             temp_assignment[node_name] = community
        #                         else:
        #                             temp_assignment[nx_node] = community
                            
        #                     community_assignment = temp_assignment
        #                     modularity = partition.modularity
        #                     method_used = f"Leiden (resolution={resolution})"
        #             except Exception:
        #                 continue
        # except Exception:
        #     pass
        
        # Method 2: Try Louvain
        if community_assignment is None or len(set(community_assignment.values())) <= 1:
            try:
                # Try both possible weight attribute names
                weight_attr = 'abs_weight' if 'abs_weight' in list(G.edges(data=True))[0][2] else 'weight'
                partition = community_louvain.best_partition(G, weight=weight_attr, resolution=0.6, random_state=42)
                
                if len(set(partition.values())) > 1:
                    temp_assignment = {}
                    for node, community in partition.items():
                        if brain_regions is not None:
                            node_name = brain_regions[node]
                            temp_assignment[node_name] = community
                        else:
                            temp_assignment[node] = community
                    
                    # Convert partition to community list for modularity calculation
                    community_sets = defaultdict(set)
                    for node, comm_id in partition.items():
                        community_sets[comm_id].add(node)
                    community_list = list(community_sets.values())
                    
                    temp_mod = nx.algorithms.community.modularity(G, community_list, weight=weight_attr)
                    
                    if temp_mod > modularity:
                        community_assignment = temp_assignment
                        modularity = temp_mod
                        method_used = "Louvain"
            except Exception:
                pass
        
        # Method 3: Greedy modularity
        if community_assignment is None or len(set(community_assignment.values())) <= 1:
            try:
                weight_attr = 'abs_weight' if G.number_of_edges() > 0 and 'abs_weight' in list(G.edges(data=True))[0][2] else 'weight'
                communities = nx.algorithms.community.greedy_modularity_communities(G, weight=weight_attr)
                
                if len(communities) > 1:
                    temp_assignment = {}
                    for comm_idx, community in enumerate(communities):
                        for node in community:
                            if brain_regions is not None:
                                node_name = brain_regions[node]
                                temp_assignment[node_name] = comm_idx
                            else:
                                temp_assignment[node] = comm_idx
                    
                    if len(set(temp_assignment.values())) > 1:
                        community_assignment = temp_assignment
                        modularity = nx.algorithms.community.modularity(G, communities, weight=weight_attr)
                        method_used = "Greedy Modularity"
            except Exception:
                pass
        
        # Fallback: each node is its own community
        if community_assignment is None:
            if brain_regions is not None:
                community_assignment = {brain_regions[node]: node for node in G.nodes()}
            else:
                community_assignment = {node: idx for idx, node in enumerate(G.nodes())}
            method_used = "fallback"
            modularity = 0.0
        
        return community_assignment, modularity, method_used
    
    def detect_communities(self):
        """Detect communities using multiple approaches for robustness and SAVE MODULARITY"""
        print("\nDetecting communities...")
        
        for group in self.groups:
            G = self.networks[group]
            print(f"  {group.upper()}: Network has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            if G.number_of_edges() == 0:
                community_assignment = {self.brain_regions[node]: node
                                      for node in G.nodes()}
                self.community_assignments[group] = community_assignment
                print(f"    No edges - each node is its own community")
                
                # Store modularity info
                self.community_modularity[group] = {
                    "modularity": 0.0,
                    "method": "no_edges",
                    "n_communities": len(community_assignment)
                }
                
                # UPDATE network_metrics with modularity
                if group in self.network_metrics:
                    self.network_metrics[group]['global']['modularity'] = 0.0
                
                continue
            
            # Try multiple community detection methods
            community_assignment = None
            modularity = -1
            method_used = "none"
            
            # Method 1: Try Leiden algorithm
            # try:
            #     print("    Trying Leiden...")
            #     edge_list = []
            #     weights = []
            #     node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            #     reverse_mapping = {idx: node for node, idx in node_mapping.items()}
                
            #     for u, v, data in G.edges(data=True):
            #         edge_list.append((node_mapping[u], node_mapping[v]))
            #         weights.append(data.get('abs_weight', 1.0))
                
            #     if len(edge_list) > 0:
            #         g = ig.Graph()
            #         g.add_vertices(len(G.nodes()))
            #         g.add_edges(edge_list)
                    
            #         for resolution in [1.0, 0.8, 0.6, 1.2]:
            #             try:
            #                 partition = g.community_leiden(weights=weights, resolution=resolution,
            #                                               beta=0.01, n_iterations=-2)
                            
            #                 if len(set(partition.membership)) > 1 and partition.modularity > modularity:
            #                     temp_assignment = {}
            #                     for ig_node, community in enumerate(partition.membership):
            #                         nx_node = reverse_mapping[ig_node]
            #                         node_name = self.brain_regions[nx_node]
            #                         temp_assignment[node_name] = community
                                
            #                     community_assignment = temp_assignment
            #                     modularity = partition.modularity
            #                     method_used = f"Leiden (resolution={resolution})"
            #             except Exception as e:
            #                 continue
            #     else:
            #         print("    Leiden skipped: no edges in edge_list")
            # except Exception as e:
            #     print(f"    Leiden setup failed: {e}")
            
            # Method 2: Try Louvain
            if community_assignment is None or len(set(community_assignment.values())) <= 1:
                try:
                    print("    Trying Louvain...")
                    partition = community_louvain.best_partition(G, weight='abs_weight', resolution=0.6, random_state=42)
                    print(f"    Louvain returned {len(set(partition.values()))} communities")
                    
                    if len(set(partition.values())) > 1:
                        temp_assignment = {}
                        for node, community in partition.items():
                            node_name = self.brain_regions[node]
                            temp_assignment[node_name] = community
                        
                        temp_mod = community_louvain.modularity(partition, G, weight='abs_weight')
                        
                        if temp_mod > modularity:
                            community_assignment = temp_assignment
                            modularity = temp_mod
                            method_used = "Louvain"
                except Exception as e:
                    print(f"    Louvain failed: {e}")
            
            # Method 3: Greedy modularity
            if community_assignment is None or len(set(community_assignment.values())) <= 1:
                try:
                    print("    Trying Greedy modularity...")
                    communities = nx.algorithms.community.greedy_modularity_communities(G, weight='abs_weight')
                    
                    if len(communities) > 1:
                        temp_assignment = {}
                        for comm_idx, community in enumerate(communities):
                            for node in community:
                                node_name = self.brain_regions[node]
                                temp_assignment[node_name] = comm_idx
                        
                        if len(set(temp_assignment.values())) > 1:
                            community_assignment = temp_assignment
                            modularity = nx.algorithms.community.modularity(G, communities, weight='abs_weight')
                            method_used = "Greedy Modularity"
                except:
                    pass
            
            # Fallback: correlation-based clustering
            if community_assignment is None or len(set(community_assignment.values())) <= 1:
                corr_matrix = self.group_correlations[group]
                pval_matrix = self.group_pvalues[group]
                
                community_assignment = {}
                community_id = 0
                assigned_nodes = set()
                
                for i, region_i in enumerate(self.brain_regions):
                    if region_i in assigned_nodes:
                        continue
                    
                    current_community = [region_i]
                    assigned_nodes.add(region_i)
                    
                    for j, region_j in enumerate(self.brain_regions):
                        if region_j in assigned_nodes:
                            continue
                        
                        if self._should_include_edge(corr_matrix[i, j], pval_matrix[i, j]):
                            current_community.append(region_j)
                            assigned_nodes.add(region_j)
                    
                    for node in current_community:
                        community_assignment[node] = community_id
                    
                    community_id += 1
                
                for region in self.brain_regions:
                    if region not in community_assignment:
                        community_assignment[region] = community_id
                        community_id += 1
                
                method_used = "Correlation-based clustering"
            
            self.community_assignments[group] = community_assignment
            n_communities = len(set(community_assignment.values()))
            
            print(f"    {n_communities} communities detected using {method_used}")
            if modularity > 0:
                print(f"    Modularity: {modularity:.3f}")
            
            # STORE modularity info
            self.community_modularity[group] = {
                "modularity": float(modularity),
                "method": method_used,
                "n_communities": n_communities
            }
            
            # UPDATE network_metrics with modularity
            if group not in self.network_metrics:
                self.network_metrics[group] = {}
            if 'global' not in self.network_metrics[group]:
                self.network_metrics[group]['global'] = {}

            self.network_metrics[group]['global']['modularity'] = float(modularity)
                
            # Print community composition
            community_composition = defaultdict(list)
            for node, community in community_assignment.items():
                community_composition[community].append(node)
            
            for comm_id, nodes in sorted(community_composition.items()):
                print(f"    Community {comm_id}: {', '.join(nodes)}")
    
    def classify_hubs(self):
        """
        Classify hubs using within-module z-score.
        
        Hub classification based on:
        - Within-module z-score (z)
        - Participation coefficient (P)
        
        Hub types:
        - Provincial hub: z ≥ 0.16, P < 0.30
        - Connector hub: z ≥ 0.16, 0.30 ≤ P < 0.75
        - Kinless hub: z ≥ 0.16, P ≥ 0.75
        - Non-hub connector: z < 0.16, P ≥ 0.30
        - Ultra-peripheral: z < 0.16, P < 0.30
        """
        print("\nClassifying hubs using within-module z-score...")
        
        for group in self.groups:
            G = self.networks[group]
            communities = self.community_assignments[group]
            hub_classification = {}
            
            for node in G.nodes():
                node_name = self.brain_regions[node]
                node_community = communities[node_name]
                
                # Within-module degree z-score
                same_community_nodes = [n for n in G.nodes()
                                      if communities[self.brain_regions[n]] == node_community]
                
                if len(same_community_nodes) > 1:
                    within_module_degrees = []
                    for n in same_community_nodes:
                        within_degree = sum(1 for neighbor in G.neighbors(n)
                                          if communities[self.brain_regions[neighbor]] == node_community)
                        within_module_degrees.append(within_degree)
                    
                    node_within_degree = sum(1 for neighbor in G.neighbors(node)
                                           if communities[self.brain_regions[neighbor]] == node_community)
                    
                    mean_within_degree = np.mean(within_module_degrees)
                    std_within_degree = np.std(within_module_degrees, ddof=1)
                    
                    if std_within_degree > 0:
                        z_score = (node_within_degree - mean_within_degree) / std_within_degree
                    else:
                        z_score = 0
                else:
                    z_score = 0
                    node_within_degree = 0
                
                # Participation coefficient
                total_degree = G.degree(node)
                if total_degree > 0:
                    community_degrees = defaultdict(int)
                    for neighbor in G.neighbors(node):
                        neighbor_community = communities[self.brain_regions[neighbor]]
                        community_degrees[neighbor_community] += 1
                    
                    participation_coeff = 1 - sum((deg / total_degree) ** 2
                                                 for deg in community_degrees.values())
                else:
                    participation_coeff = 0
                
                # Hub classification based on z-score and participation coefficient
                if z_score >= 0.16:
                    if participation_coeff < 0.3:
                        hub_type = 'provincial_hub'
                    elif participation_coeff < 0.75:
                        hub_type = 'connector_hub'
                    else:
                        hub_type = 'kinless_hub'
                else:
                    if participation_coeff >= 0.30:
                        hub_type = 'non-hub_connector'
                    else:
                        hub_type = 'ultra-peripheral'
                
                hub_classification[node_name] = {
                    'within_module_z_score': z_score,
                    'within_module_degree': node_within_degree,
                    'total_degree': total_degree,
                    'participation_coefficient': participation_coeff,
                    'hub_type': hub_type,
                    'community': node_community
                }
            
            self.hub_classifications[group] = hub_classification
            
            # Count hub types and print summary
            hub_counts = defaultdict(int)
            hub_summary = defaultdict(list)
            
            for node_name, classification in hub_classification.items():
                hub_type = classification['hub_type']
                hub_counts[hub_type] += 1
                hub_summary[hub_type].append({
                    'name': node_name,
                    'z_score': classification['within_module_z_score'],
                    'pc': classification['participation_coefficient'],
                    'degree': classification['total_degree']
                })
            
            print(f"\n  {group.upper()} Hub Classification Summary:")
            for hub_type in ['provincial_hub', 'connector_hub', 'kinless_hub', 'non-hub_connector', 'ultra-peripheral']:
                if hub_type in hub_counts:
                    count = hub_counts[hub_type]
                    print(f"    {hub_type}: {count}")
                    
                    # Print details for hubs (not ultra-peripheral)
                    if hub_type != 'ultra-peripheral':
                        for node_info in hub_summary[hub_type]:
                            print(f"      - {node_info['name']:6s} | z={node_info['z_score']:6.2f} | "
                                 f"PC={node_info['pc']:.3f} | degree={node_info['degree']}")
            print()
    
    def animal_level_permutation_test(self, metric_function, n_permutations=5000):
        """Perform animal-level permutation test"""
        print(f"\nPerforming animal-level permutation tests ({n_permutations} permutations)...")
        
        all_data = []
        all_labels = []
        
        for group in self.groups:
            group_data = self.individual_data[group]
            all_data.extend(group_data)
            all_labels.extend([group] * len(group_data))
        
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        
        # Calculate observed difference
        observed_metrics = {}
        for group in self.groups:
            group_indices = np.where(all_labels == group)[0]
            group_data = all_data[group_indices]
            observed_metrics[group] = metric_function(group_data)
        
        observed_diff = observed_metrics[self.groups[1]] - observed_metrics[self.groups[0]]
        
        # Permutation test
        permuted_diffs = []
        group_sizes = [len(self.individual_data[group]) for group in self.groups]
        
        for i in range(n_permutations):
            shuffled_indices = np.random.permutation(len(all_data))
            shuffled_data = all_data[shuffled_indices]
            
            split_point = group_sizes[0]
            pseudo_group1_data = shuffled_data[:split_point]
            pseudo_group2_data = shuffled_data[split_point:split_point + group_sizes[1]]
            
            pseudo_metric1 = metric_function(pseudo_group1_data)
            pseudo_metric2 = metric_function(pseudo_group2_data)
            permuted_diff = pseudo_metric2 - pseudo_metric1
            permuted_diffs.append(permuted_diff)
        
        permuted_diffs = np.array(permuted_diffs)
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        
        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'permuted_distribution': permuted_diffs,
            'observed_metrics': observed_metrics
        }
    
    def compare_network_metrics_statistically(self, n_permutations=5000):
        """Statistical comparison using animal-level permutations - UPDATED: 6 metrics only"""
        print("\nStatistical comparison of network metrics...")
        results = {}
        
        # Define metric functions for permutation testing - UPDATED: 6 metrics only
        def average_clustering(data):
            n_regions = data.shape[1]
            G = nx.Graph()
            G.add_nodes_from(range(n_regions))
            
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    region_i_data = data[:, i]
                    region_j_data = data[:, j]
                    valid_indices = ~(np.isnan(region_i_data) | np.isnan(region_j_data))
                    
                    if np.sum(valid_indices) >= 3:
                        r, p = spearmanr(region_i_data[valid_indices],
                                       region_j_data[valid_indices], alternative='greater')
                        if not np.isnan(r) and self._should_include_edge(r, p):
                            G.add_edge(i, j, weight=abs(r))
            
            return nx.average_clustering(G, weight='weight')
        
        def global_efficiency(data):
            n_regions = data.shape[1]
            G = nx.Graph()
            G.add_nodes_from(range(n_regions))
            
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    region_i_data = data[:, i]
                    region_j_data = data[:, j]
                    valid_indices = ~(np.isnan(region_i_data) | np.isnan(region_j_data))
                    
                    if np.sum(valid_indices) >= 3:
                        r, p = spearmanr(region_i_data[valid_indices],
                                       region_j_data[valid_indices], alternative='greater')
                        if not np.isnan(r) and self._should_include_edge(r, p):
                            G.add_edge(i, j, weight=abs(r))
            
            if G.number_of_edges() > 0:
                return nx.global_efficiency(G)
            else:
                return 0
        
        def small_worldness(data):
            n_regions = data.shape[1]
            G = nx.Graph()
            G.add_nodes_from(range(n_regions))
            
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    region_i_data = data[:, i]
                    region_j_data = data[:, j]
                    valid_indices = ~(np.isnan(region_i_data) | np.isnan(region_j_data))
                    
                    if np.sum(valid_indices) >= 3:
                        r, p = spearmanr(region_i_data[valid_indices],
                                       region_j_data[valid_indices], alternative='greater')
                        if not np.isnan(r) and self._should_include_edge(r, p):
                            G.add_edge(i, j, weight=abs(r))
            
            for u, v in G.edges():
                if 'abs_weight' not in G[u][v]:
                    G[u][v]['abs_weight'] = G[u][v].get('weight', 1.0)
            
            sw = calculate_small_worldness(G)
            return sw if not np.isnan(sw) else 0
        
        def mean_degree(data):
            n_regions = data.shape[1]
            G = nx.Graph()
            G.add_nodes_from(range(n_regions))
            
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    region_i_data = data[:, i]
                    region_j_data = data[:, j]
                    valid_indices = ~(np.isnan(region_i_data) | np.isnan(region_j_data))
                    
                    if np.sum(valid_indices) >= 3:
                        r, p = spearmanr(region_i_data[valid_indices],
                                       region_j_data[valid_indices], alternative='greater')
                        if not np.isnan(r) and self._should_include_edge(r, p):
                            G.add_edge(i, j)
            
            degrees = [d for n, d in G.degree()]
            return np.mean(degrees) if len(degrees) > 0 else 0
        
        def mean_strength(data):
            n_regions = data.shape[1]
            G = nx.Graph()
            G.add_nodes_from(range(n_regions))
            
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    region_i_data = data[:, i]
                    region_j_data = data[:, j]
                    valid_indices = ~(np.isnan(region_i_data) | np.isnan(region_j_data))
                    
                    if np.sum(valid_indices) >= 3:
                        r, p = spearmanr(region_i_data[valid_indices],
                                       region_j_data[valid_indices], alternative='greater')
                        if not np.isnan(r) and self._should_include_edge(r, p):
                            G.add_edge(i, j, weight=abs(r))
            
            strengths = []
            for node in G.nodes():
                strength = sum([G[node][neighbor].get('weight', 1.0) for neighbor in G.neighbors(node)])
                strengths.append(strength)
            return np.mean(strengths) if len(strengths) > 0 else 0
        
        def modularity_metric(data):
            """Calculate modularity for permutation test using detect_communities logic"""
            n_regions = data.shape[1]
            G = nx.Graph()
            G.add_nodes_from(range(n_regions))
            
            # Build graph from data
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    region_i_data = data[:, i]
                    region_j_data = data[:, j]
                    valid_indices = ~(np.isnan(region_i_data) | np.isnan(region_j_data))
                    
                    if np.sum(valid_indices) >= 3:
                        r, p = spearmanr(region_i_data[valid_indices],
                                       region_j_data[valid_indices], alternative='greater')
                        if not np.isnan(r) and self._should_include_edge(r, p):
                            G.add_edge(i, j, weight=abs(r))
            
            if G.number_of_edges() == 0:
                return 0.0
            
            # Use the shared community detection method
            community_assignment, modularity, method_used = self._detect_communities_from_graph(G, brain_regions=None)
            
            return modularity
        
        # Global metrics dictionary - UPDATED: Only 6 metrics
        global_metrics = {
            'mean_degree': mean_degree,
            'mean_strength': mean_strength,
            'average_clustering': average_clustering,
            'global_efficiency': global_efficiency,
            'small_worldness': small_worldness,
            'modularity': modularity_metric,
        }
        
        results['global_metrics'] = {}
        for metric_name, metric_func in global_metrics.items():
            print(f"  Testing {metric_name}...")
            result = self.animal_level_permutation_test(metric_func, n_permutations)
            results['global_metrics'][metric_name] = result
            
            if result['p_value'] < 0.05:
                print(f"    SIGNIFICANT: p = {result['p_value']:.4f}")
            else:
                print(f"    Not significant: p = {result['p_value']:.4f}")
        
        self.statistical_results = results
        return results
    
    def compare_individual_cfos_expression(self):
        """
        Statistical comparison of c-fos expression levels.
        
        For each region:
        - Welch's t-test between group1 and group2
        - One-sample t-tests vs baseline = 1.0 for each group
        - FDR correction applied separately to each family of tests
        """
        print("\nStatistical comparison of c-fos expression levels...")
        cfos_results = {}
        
        # --- 1. Compute all tests per region (raw p-values) ---
        for region in self.brain_regions:
            group1_vals = self.data[self.data['Treatment'] == self.groups[0]][region].dropna().values
            group2_vals = self.data[self.data['Treatment'] == self.groups[1]][region].dropna().values
            
            # Basic descriptive stats
            mean1, mean2 = np.mean(group1_vals), np.mean(group2_vals)
            std1, std2 = np.std(group1_vals, ddof=1), np.std(group2_vals, ddof=1)
            n1, n2 = len(group1_vals), len(group2_vals)
            
            # Welch's t-test between groups
            if n1 > 1 and n2 > 1:
                t_between, p_between = ttest_ind(group1_vals, group2_vals, equal_var=False)
            else:
                t_between, p_between = np.nan, 1.0
            
            # Effect size (Cohen's d for between-group)
            if n1 > 1 and n2 > 1:
                pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
                pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0
                effect_size = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
            else:
                effect_size = 0
            
            # One-sample t-tests vs baseline = 1.0
            if n1 > 1:
                t_g1_vs1, p_g1_vs1 = ttest_1samp(group1_vals, popmean=1.0)
            else:
                t_g1_vs1, p_g1_vs1 = np.nan, 1.0
            
            if n2 > 1:
                t_g2_vs1, p_g2_vs1 = ttest_1samp(group2_vals, popmean=1.0)
            else:
                t_g2_vs1, p_g2_vs1 = np.nan, 1.0
            
            cfos_results[region] = {
                # Descriptives
                'group1_mean': mean1,
                'group1_std': std1,
                'group1_n': n1,
                'group2_mean': mean2,
                'group2_std': std2,
                'group2_n': n2,
                # Between-group Welch test (raw)
                't_between': t_between,
                'p_between_raw': p_between,
                'effect_size': effect_size,
                'difference': mean2 - mean1,
                # One-sample vs 1.0 (raw)
                't_group1_vs1': t_g1_vs1,
                'p_group1_vs1_raw': p_g1_vs1,
                't_group2_vs1': t_g2_vs1,
                'p_group2_vs1_raw': p_g2_vs1,
            }
        
        # --- 2. FDR correction for each family of tests ---
        # (a) Between-group tests
        p_between_list = [res['p_between_raw'] for res in cfos_results.values()]
        rej_between, p_between_corr, _, _ = multipletests(p_between_list, method='fdr_bh')
        
        # (b) Group1 vs 1
        p_g1_list = [res['p_group1_vs1_raw'] for res in cfos_results.values()]
        rej_g1, p_g1_corr, _, _ = multipletests(p_g1_list, method='fdr_bh')
        
        # (c) Group2 vs 1
        p_g2_list = [res['p_group2_vs1_raw'] for res in cfos_results.values()]
        rej_g2, p_g2_corr, _, _ = multipletests(p_g2_list, method='fdr_bh')
        
        # Attach corrected p-values and significance flags
        for i, region in enumerate(cfos_results.keys()):
            res = cfos_results[region]
            res['p_between_fdr'] = p_between_corr[i]
            res['between_sig_fdr'] = rej_between[i]
            res['p_group1_vs1_fdr'] = p_g1_corr[i]
            res['group1_vs1_sig_fdr'] = rej_g1[i]
            res['p_group2_vs1_fdr'] = p_g2_corr[i]
            res['group2_vs1_sig_fdr'] = rej_g2[i]
        
        self.cfos_results = cfos_results
        
        # Between-group significant regions (FDR)
        sig_between = [r for r, res in cfos_results.items() if res['between_sig_fdr']]
        print(f"  Between-group c-fos differences (FDR corrected): {len(sig_between)}")
        for region in sig_between:
            res = cfos_results[region]
            direction = "↑" if res['difference'] > 0 else "↓"
            print(f"    {region}: {self.groups[1]} {direction} {self.groups[0]} "
                 f"(p_raw={res['p_between_raw']:.4f}, q_FDR={res['p_between_fdr']:.4f}, "
                 f"d={res['effect_size']:.3f})")
        
        # Example: report group1 vs 1.0 significant regions
        sig_g1 = [r for r, res in cfos_results.items() if res['group1_vs1_sig_fdr']]
        print(f"\n  {self.groups[0].upper()} vs baseline=1.0 (FDR): {len(sig_g1)} regions")
        for region in sig_g1:
            res = cfos_results[region]
            direction = "↑" if res['group1_mean'] > 1.0 else "↓"
            print(f"    {region}: {self.groups[0]} {direction} 1.0 "
                 f"(p_raw={res['p_group1_vs1_raw']:.4f}, q_FDR={res['p_group1_vs1_fdr']:.4f})")
        
        sig_g2 = [r for r, res in cfos_results.items() if res['group2_vs1_sig_fdr']]
        print(f"\n  {self.groups[1].upper()} vs baseline=1.0 (FDR): {len(sig_g2)} regions")
        for region in sig_g2:
            res = cfos_results[region]
            direction = "↑" if res['group2_mean'] > 1.0 else "↓"
            print(f"    {region}: {self.groups[1]} {direction} 1.0 "
                 f"(p_raw={res['p_group2_vs1_raw']:.4f}, q_FDR={res['p_group2_vs1_fdr']:.4f})")
        
        return cfos_results
    
    def create_comprehensive_plots(self, output_dir="brain_network_analysis_results"):
        """Create comprehensive visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCreating comprehensive plots in {output_dir}/...")
        
        # 1. Correlation matrices
        self.plot_correlation_matrices(output_dir)
        
        # 2. Network visualizations
        self._plot_networks_original_style(output_dir)
        
        # 3. c-fos expression comparison
        self._plot_cfos_comparison(output_dir)
        
        # 4. Network metrics comparison - UPDATED: 6 metrics only
        self._plot_network_metrics_permutation(output_dir)
        
        # 5. Hub analysis - UPDATED: z-score on y-axis
        self._plot_hub_analysis(output_dir)
        
        # 6. Community structure
        self._plot_community_structure(output_dir)
        
        print("  All plots created successfully!")
    
    def _plot_networks_original_style(self, output_dir):
        """Plot network visualizations"""
        for group in self.groups:
            G = self.networks[group]
            communities = self.community_assignments[group]
            
            threshold_desc = self._get_threshold_description()
            title = f"{group.upper()} Group Functional Connectivity ({threshold_desc})"
            filename = f"{group}_functional_connectivity"
            
            visualize_graph_by_coeff(G, communities, title, filename, output_dir)
            print(f"  Network plot saved: {filename}.png/eps")
    
    def _plot_cfos_comparison(self, output_dir):
        """Plot c-fos expression comparison"""
        if not hasattr(self, 'cfos_results'):
            return
        
        regions = list(self.cfos_results.keys())
        
        # Means and SEMs
        group1_means = [self.cfos_results[r]['group1_mean'] for r in regions]
        group1_sems = [self.cfos_results[r]['group1_std'] /
                      np.sqrt(self.cfos_results[r]['group1_n']) for r in regions]
        
        group2_means = [self.cfos_results[r]['group2_mean'] for r in regions]
        group2_sems = [self.cfos_results[r]['group2_std'] /
                      np.sqrt(self.cfos_results[r]['group2_n']) for r in regions]
        
        # Between-group p-values (raw and FDR)
        p_between_raw = [self.cfos_results[r]['p_between_raw'] for r in regions]
        p_between_fdr = [self.cfos_results[r]['p_between_fdr'] for r in regions]
        
        # Within-group vs baseline (if you want to show markers later)
        p_g1_fdr = [self.cfos_results[r]['p_group1_vs1_fdr'] for r in regions]
        p_g2_fdr = [self.cfos_results[r]['p_group2_vs1_fdr'] for r in regions]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(regions))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, group1_means, width,
                      yerr=group1_sems, label=self.groups[0].upper(),
                      alpha=0.8, capsize=5, color='lightgray', edgecolor='black')
        
        bars2 = ax.bar(x + width/2, group2_means, width,
                      yerr=group2_sems, label=self.groups[1].upper(),
                      alpha=0.8, capsize=5, color='mediumpurple', edgecolor='black')
        
        # Horizontal baseline at 1.0 (homecage)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Add between-group significance markers
        for i, (p_raw, p_fdr) in enumerate(zip(p_between_raw, p_between_fdr)):
            p_for_stars = p_fdr
            
            if p_for_stars < 0.05:
                max_height = max(group1_means[i] + group1_sems[i],
                               group2_means[i] + group2_sems[i])
                
                if p_for_stars < 0.001:
                    stars = '***'
                elif p_for_stars < 0.01:
                    stars = '**'
                else:
                    stars = '*'
                
                y = max_height * 1.10
                ax.plot([i - width/2, i + width/2], [y, y], 'k-', linewidth=1.5)
                ax.text(i, y * 1.02, stars,
                       ha='center', va='bottom', fontsize=12, color='red',
                       fontweight='bold')
        
        # Baseline y position for within-group significance markers
        max_height_all = max(max(np.array(group1_means) + np.array(group1_sems)),
                            max(np.array(group2_means) + np.array(group2_sems)))
        y_baseline = 0 - max_height_all * 0.08
        
        # Add within-group vs baseline significance markers
        for i, r in enumerate(regions):
            # group 1 vs baseline
            if p_g1_fdr[i] < 0.01:
                ax.text(x[i] - width/2, y_baseline, '***',
                       ha='center', va='top', fontsize=10,
                       color='blue', fontweight='bold')
            
            # group 2 vs baseline
            if p_g2_fdr[i] < 0.01:
                ax.text(x[i] + width/2, y_baseline, '***',
                       ha='center', va='top', fontsize=10,
                       color='purple', fontweight='bold')
        
        ax.set_xlabel('Brain Region')
        ax.set_ylabel('c-fos Expression (normalized)')
        ax.set_title("c-fos Expression Comparison\nWelch's t-test (bars), one-sample vs 1.0 (markers), FDR-corrected")
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cfos_expression_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'cfos_expression_comparison.eps'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  c-fos expression comparison plot saved png/eps")
    
    def _plot_network_metrics_permutation(self, output_dir):
        """Plot network metrics permutation test results - UPDATED: 6 metrics (2x3 grid)"""
        if not hasattr(self, 'statistical_results'):
            return
        
        global_results = self.statistical_results['global_metrics']
        
        # Create 2x3 subplot grid for 6 metrics
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        metric_names_order = ['mean_degree', 'mean_strength', 'average_clustering',
                             'global_efficiency', 'small_worldness', 'modularity']
        
        for idx, metric_name in enumerate(metric_names_order):
            if metric_name not in global_results:
                axes[idx].set_visible(False)
                continue
            
            results = global_results[metric_name]
            observed_diff = results['observed_difference']
            permuted_diffs = results['permuted_distribution']
            p_value = results['p_value']
            
            ax = axes[idx]
            
            ax.hist(permuted_diffs, bins=50, alpha=0.7, density=True,
                   color='lightblue', edgecolor='black')
            ax.axvline(observed_diff, color='red', linestyle='--', linewidth=2,
                      label=f'Observed\n(p = {p_value:.4f})')
            
            ax.set_xlabel(f'Permuted Differences ({metric_name})')
            ax.set_ylabel('Density')
            ax.set_title(f"{metric_name.replace('_', ' ').title()}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Global Network Metrics: Animal-Level Permutation Tests', fontsize=16)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'global_metrics_permutation_tests.png'),
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'global_metrics_permutation_tests.eps'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Global metrics permutation tests plot saved png/eps")
    
    def _plot_hub_analysis(self, output_dir):
        """Plot hub analysis results with z-score on y-axis"""
        if not self.hub_classifications:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, group in enumerate(self.groups):
            hubs = self.hub_classifications[group]
            regions = list(hubs.keys())
            
            # Extract z-scores and participation coefficients
            z_scores = np.array([hubs[r]['within_module_z_score'] for r in regions])
            participation_coeffs = np.array([hubs[r]['participation_coefficient'] for r in regions])
            hub_types = [hubs[r]['hub_type'] for r in regions]
            
            # ---- jitter to separate overlapping nodes ----
            coords = np.column_stack((participation_coeffs, z_scores))
            unique, inverse, counts = np.unique(coords, axis=0,
                                              return_inverse=True, return_counts=True)
            
            jittered_x = participation_coeffs.copy()
            jittered_y = z_scores.copy()
            
            radius = 0.01  # adjust for how far apart you want points
            for k, (x0, y0) in enumerate(unique):
                idxs = np.where(inverse == k)[0]
                n = len(idxs)
                if n > 1:
                    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                    jittered_x[idxs] = x0 + radius * np.cos(angles)
                    jittered_y[idxs] = y0 + radius * np.sin(angles)
            # ---------------------------------------------
            
            type_colors = {
                'ultra-peripheral': 'lightgray',
                'non-hub_connector': 'lightyellow',
                'provincial_hub': 'orange',
                'connector_hub': 'red',
                'kinless_hub': 'purple'
            }
            
            colors = [type_colors.get(ht, 'black') for ht in hub_types]
            
            scatter = axes[idx].scatter(jittered_x, jittered_y,
                                      c=colors, s=100, alpha=0.7, edgecolors='black')
            
            # Label hubs (z >= 0.16)
            for i, region in enumerate(regions):
                if z_scores[i] >= 0.16:
                    axes[idx].annotate(region, (jittered_x[i], jittered_y[i]),
                                     xytext=(5, 5), textcoords='offset points',
                                     fontsize=8, ha='left')
            
            # Reference lines matching classification thresholds
            axes[idx].axhline(y=0.16, color='red', linestyle='--', linewidth=1.5, alpha=0.6,
                            label='Hub threshold (z=0.16)')
            axes[idx].axvline(x=0.3, color='black', linestyle='--', alpha=0.5,
                            label='Provincial/Connector (P=0.3)')
            axes[idx].axvline(x=0.75, color='black', linestyle=':', alpha=0.5,
                            label='Connector/Kinless (P=0.75)')
            
            axes[idx].set_xlabel('Participation Coefficient', fontsize=12)
            axes[idx].set_ylabel('Within-Module Z-Score', fontsize=12)
            axes[idx].set_title(f'{group.upper()} Hub Classification', fontsize=14, fontweight='bold')
            axes[idx].set_xlim(-0.05, 1.0)
            axes[idx].grid(True, alpha=0.3)
            
            # Add small legend for threshold lines
            axes[idx].legend(loc='upper right', fontsize=7, framealpha=0.9)
        
        legend_elements = [plt.scatter([], [], c=color, s=100, label=hub_type.replace('_', ' ').title())
                          for hub_type, color in type_colors.items()]
        
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hub_classification.png'),
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'hub_classification.eps'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Hub classification plot saved png/eps")
    
    def _plot_community_structure(self, output_dir):
        """Plot community structure analysis"""
        if not self.community_assignments:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, group in enumerate(self.groups):
            communities = self.community_assignments[group]
            
            community_sizes = defaultdict(int)
            for region, community in communities.items():
                community_sizes[community] += 1
            
            comm_ids = list(community_sizes.keys())
            sizes = list(community_sizes.values())
            
            axes[idx].bar(comm_ids, sizes, alpha=0.7)
            axes[idx].set_xlabel('Community ID')
            axes[idx].set_ylabel('Number of Regions')
            axes[idx].set_title(f'{group.upper()} Community Structure\n'
                              f'{len(comm_ids)} communities')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'community_structure.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Community structure plot saved")
    
    def save_results(self, output_dir="brain_network_analysis_results"):
        """Save all analysis results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving results to {output_dir}/...")
        
        # 1. Correlation matrices and p-values
        for group in self.groups:
            corr_df = pd.DataFrame(self.group_correlations[group],
                                  index=self.brain_regions,
                                  columns=self.brain_regions)
            corr_df.to_csv(os.path.join(output_dir, f'{group}_correlation_matrix.csv'))
            
            pval_df = pd.DataFrame(self.group_pvalues[group],
                                  index=self.brain_regions,
                                  columns=self.brain_regions)
            pval_df.to_csv(os.path.join(output_dir, f'{group}_pvalue_matrix.csv'))
        
        # 2. c-fos expression results
        if hasattr(self, 'cfos_results'):
            cfos_df = pd.DataFrame(self.cfos_results).T
            cfos_df.to_csv(os.path.join(output_dir, 'cfos_expression_results.csv'))
        
        # 3. Network metrics (UPDATED: includes modularity from detect_communities)
        network_metrics_data = []
        for group in self.groups:
            metrics = self.network_metrics[group]
            for metric, value in metrics['global'].items():
                network_metrics_data.append({
                    'group': group,
                    'level': 'global',
                    'region': 'all',
                    'metric': metric,
                    'value': value
                })
        
        network_df = pd.DataFrame(network_metrics_data)
        network_df.to_csv(os.path.join(output_dir, 'network_metrics.csv'), index=False)
        
        # 4. Hub classifications
        if self.hub_classifications:
            hub_data = []
            for group in self.groups:
                for region, hub_info in self.hub_classifications[group].items():
                    hub_data.append({
                        'group': group,
                        'region': region,
                        **hub_info
                    })
            
            hub_df = pd.DataFrame(hub_data)
            hub_df.to_csv(os.path.join(output_dir, 'hub_classifications.csv'), index=False)
        
        # 5. Statistical test results
        if hasattr(self, 'statistical_results'):
            stats_data = []
            for metric, results in self.statistical_results['global_metrics'].items():
                stats_data.append({
                    'level': 'global',
                    'region': 'all',
                    'metric': metric,
                    'observed_difference': results['observed_difference'],
                    'p_value': results['p_value'],
                    'test_type': 'animal_level_permutation'
                })
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_csv(os.path.join(output_dir, 'statistical_test_results.csv'), index=False)
        
        print("  All results saved successfully!")
    
    def run_complete_analysis(self, n_permutations=1000, output_dir="brain_network_analysis_results"):
        """Run the complete analysis pipeline"""
        print("="*80)
        print("COMPREHENSIVE BRAIN FUNCTIONAL CONNECTIVITY ANALYSIS - COMPLETE")
        print("WITH FISHER R-TO-Z TRANSFORMATION")
        print("UPDATED: 6 GLOBAL METRICS (REMOVED: transitivity, char_path_length, density; ADDED: modularity)")
        print("UPDATED: Z-SCORE BASED HUB CLASSIFICATION WITH Z-SCORE PLOTTING")
        print("="*80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Compute functional connectivity
        self.compute_functional_connectivity()
        
        # Step 3: Create networks
        self.create_networks()
        
        # Step 4: Detect communities (needed for modularity) - SAVES MODULARITY
        self.detect_communities()
        
        # Step 5: Calculate network metrics (UPDATED: 6 metrics + modularity from step 4)
        self.calculate_network_metrics()
        
        # Step 6: Classify hubs (UPDATED: z-score based)
        self.classify_hubs()
        
        # Step 7: Statistical comparisons
        self.compare_individual_cfos_expression()
        self.compare_network_metrics_statistically(n_permutations=n_permutations)
        
        # Step 8: Fisher r-to-z transformation
        self.compare_correlation_matrices_fisher_with_validation(output_dir)
        
        # Step 9: Create visualizations (UPDATED: z-score plots)
        self.create_comprehensive_plots(output_dir)
        
        # Step 10: Save results
        self.save_results(output_dir)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved in: {output_dir}/")
        print("\nThreshold Configuration:")
        print(f"  Method: {self.threshold_method}")
        print(f"  Correlation Threshold: {self.correlation_threshold}")
        print(f"  P-value Threshold: {self.p_value_threshold}")


# Main execution
if __name__ == "__main__":
    # Example: Using coefficient thresholding only (original behavior)
    analyzer = BrainNetworkAnalyzer(
        data_file='Imuno 2025 - Dados brutos MIF PrL Remota vMoises2 .xlsx',
        groups=['vei', 'mif'],
        correlation_threshold=0.7333,
        p_value_threshold=0.05,
        threshold_method="coefficient"  # Use absolute correlation threshold only
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis(
        n_permutations=10000,
        output_dir="graphs/brain_network_analysis_final_extended_REM90"
    )
