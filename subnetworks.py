import pandas as pd
import numpy as np
from itertools import combinations
import os

os.makedirs('output', exist_ok=True)

# --- Load data ---
cfos = pd.read_excel('allv1.xlsx', sheet_name=0)
cfos['Timepoint'] = cfos['Timepoint'].str.lower().map({'rec': 'REC', 'rem': 'REM'})
cfos['Treatment'] = cfos['Treatment'].str.lower().map({'vei': 'VEH', 'mif': 'MIF'})
cfos['Group'] = cfos['Treatment'] + '_' + cfos['Timepoint']

# --- Region list (must match your column names exactly) ---
regions = ['dDG','dCA3','dCA1','vDG','vCA3','vCA1','BLA','CeA','ACC','PrL','aIC','aRSC']

# --- Subnetwork definitions (edit as needed) ---
hipp_d   = ['dDG','dCA3','dCA1']
hipp_v   = ['vDG','vCA3','vCA1']
hipp_all = hipp_d + hipp_v
amyg     = ['BLA','CeA']
cort     = ['ACC','PrL','aIC','aRSC']

subnetworks = {
    'HPC_within':    [(a,b) for a,b in combinations(hipp_all, 2)],
    'dHPC_within':   [(a,b) for a,b in combinations(hipp_d, 2)],
    'vHPC_within':   [(a,b) for a,b in combinations(hipp_v, 2)],
    'Amy_within':    [(a,b) for a,b in combinations(amyg, 2)],
    'Cortex_within': [(a,b) for a,b in combinations(cort, 2)],
    'HPC_Amy':       [(h,a) for h in hipp_all for a in amyg],
    'HPC_Cortex':    [(h,c) for h in hipp_all for c in cort],
    'Amy_Cortex':    [(a,c) for a in amyg for c in cort],
}

def fisher_z(r):
    # Clamp avoids infinite values at exactly ±1
    return np.arctanh(np.clip(r, -0.9999, 0.9999))

# KEY FIX: correlations are computed ACROSS RATS within each group
# (rats = observations, regions = variables)
# This is the standard c-Fos co-activation approach.
# Each edge gives one rho/z value per group.
# The old approach (correlating within a single rat's row) was meaningless.

groups = ['VEH_REC', 'VEH_REM', 'MIF_REC', 'MIF_REM']

edge_rows    = []  # one row per group x edge — use for plotting (like "Data 6")
summary_rows = []  # one row per group x subnetwork — use for ANOVA / Kruskal-Wallis

for grp in groups:
    grp_data = cfos[cfos['Group'] == grp][regions]
    # Keep only regions with at least 3 non-NaN values
    grp_data = grp_data.loc[:, grp_data.notna().sum() >= 3]

    # Spearman correlation matrix across rats
    corr_mat = grp_data.corr(method='spearman')

    for net_name, edges in subnetworks.items():
        net_zs = []
        for a, b in edges:
            if a in corr_mat.index and b in corr_mat.columns:
                r = corr_mat.loc[a, b]
                if pd.notna(r):
                    z = fisher_z(r)
                    net_zs.append(z)
                    edge_rows.append({
                        'Group':      grp,
                        'Treatment':  grp.split('_')[0],
                        'Timepoint':  grp.split('_')[1],
                        'Subnetwork': net_name,
                        'Edge':       f'{a}-{b}',
                        'rho':        round(r, 4),
                        'Fisher_z':   round(z, 4),
                    })
        if net_zs:
            mean_z = np.mean(net_zs)
            summary_rows.append({
                'Group':                   grp,
                'Treatment':               grp.split('_')[0],
                'Timepoint':               grp.split('_')[1],
                'Subnetwork':              net_name,
                'n_edges':                 len(net_zs),
                'mean_Fisher_z':           round(mean_z, 4),
                # Back-transform to rho ONLY for interpretation/plotting
                'mean_rho_backtransformed': round(np.tanh(mean_z), 4),
            })

edge_df    = pd.DataFrame(edge_rows)
summary_df = pd.DataFrame(summary_rows)

# --- Save outputs ---
# Use edge_df for plotting (one dot per edge, like "Data 6" figure)
# Use summary_df for group-level comparisons / ANOVA input
edge_df.to_csv('output/subnetwork_edges_per_group.csv', index=False)
summary_df.to_csv('output/subnetwork_means_per_group.csv', index=False)

print("Done.")
print(f"  Edge-level CSV: {len(edge_df)} rows  -> output/subnetwork_edges_per_group.csv")
print(f"  Summary CSV:    {len(summary_df)} rows  -> output/subnetwork_means_per_group.csv")
