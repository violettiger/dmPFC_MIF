# dmPFC Glucocorticoid Receptor Blockade and Fear Memory Network Reorganization

This repository contains the analysis code supporting the manuscript **"Prefrontal cortex glucocorticoid receptors during fear memory consolidation shift the balance between salience and default-mode networks at retrieval in rats"** (Corrêa et al., submitted to *bioRxiv*).

The scripts reproduce the c-Fos functional connectivity analyses, graph-theoretic metrics, permutation-based statistical thresholding, and behavioral–endocrine correlation analyses reported in the manuscript. Rats received bilateral intra-dmPFC infusions of vehicle (VEH) or the glucocorticoid receptor (GR) antagonist mifepristone (MIF) immediately after contextual fear conditioning, and were tested for fear memory retrieval at recent (2-day) or remote (14-day) time points, followed by c-Fos immunohistochemistry across hippocampal, amygdalar, and neocortical regions.

## Repository Contents

| File | Description |
|---|---|
| `network_complete.py` | Core functional connectivity pipeline: computes Spearman correlation matrices of regional c-Fos expression per group, applies Fisher r-to-z transformation for between-group comparisons, performs animal-level permutation testing of global graph metrics (degree, strength, clustering, efficiency, modularity, small-worldness), classifies nodes as hubs (participation coefficient × within-module z-score) using Louvain community detection, and generates all network visualizations (correlation heatmaps, circular graphs, hub classification plots, community structure plots). |
| `permutation_analysis.py` | Standalone script for empirically calibrating the correlation threshold used to binarize functional connectivity graphs. Builds a null distribution by randomly reassigning animals to surrogate groups, computes Spearman correlation matrices for each permutation, and compares the real vs. null distributions using bootstrap resampling, Mann-Whitney U, and Kolmogorov-Smirnov tests. Outputs the recommended empirical threshold (70th percentile of the null distribution) used for graph binarization in `network_complete.py`. |
| `subnetworks.py` | Computes Fisher z-transformed Spearman correlations for anatomically defined subnetworks (within dorsal/ventral hippocampus, within amygdala, within cortex, hippocampus-amygdala, hippocampus-cortex, amygdala-cortex) across the four experimental groups (VEH/MIF × recent/remote). Produces edge-level and group-summary CSVs used for subnetwork-level statistical comparisons. |
| `Behavioral_correlations.ipynb` | Computes Spearman correlation matrices and p-values between behavioral measures (freezing, generalization index), corticosterone parameters (30/60 min levels, decay ratio, AUC), and regional c-Fos expression, separately for VEH and MIF groups. Generates annotated correlation heatmaps (significant correlations marked with asterisks) used in Figures 1, 2, and 4. |

## Requirements

- Python ≥ 3.9
- numpy, pandas, scipy, statsmodels
- networkx, igraph, python-louvain (`community`)
- matplotlib, seaborn
- openpyxl (for reading `.xlsx` input files)

Install dependencies:

```bash
pip install numpy pandas scipy statsmodels networkx python-igraph python-louvain matplotlib seaborn openpyxl
```

## Data

Raw and processed data (behavioral, endocrine, and c-Fos immunohistochemistry datasets) are deposited at Mendeley Data:

**https://data.mendeley.com/datasets/gvykybtvw7**

Scripts in this repository expect the c-Fos dataset (`allv1.xlsx` or equivalent CSV) with one row per animal and columns for `Treatment` (vei/mif), `Timepoint` (rec/rem), and normalized c-Fos counts for each sampled region (dDG, dCA3, dCA1, vDG, vCA3, vCA1, BLA, CeA, ACC, PrL, aIC, aRSC). Behavioral/endocrine correlation analyses additionally require columns for freezing, generalization index, and corticosterone measures (Cort1, Cort2, Decay, AUC).

## Usage

Typical workflow order:

1. **Threshold calibration** — run `permutation_analysis.py` on the full c-Fos dataset to determine the empirical correlation threshold for graph binarization.
2. **Subnetwork analysis** — run `subnetworks.py` to generate edge- and group-level Fisher z-transformed correlation summaries for anatomically defined subnetworks.
3. **Full network analysis** — run `network_complete.py` for group-wise correlation matrices, Fisher r-to-z group comparisons, hub classification, community detection, and permutation-based global metric comparisons.
4. **Behavioral-endocrine correlations** — run `Behavioral_correlations.ipynb` to generate correlation heatmaps linking behavior, corticosterone dynamics, and c-Fos expression.

Example:

```bash
python permutation_analysis.py --input allv1.xlsx --iterations 10000 --bootstrap 10000
python subnetworks.py
python network_complete.py
```

Outputs (CSV tables and PNG/EPS figures) are written to the `output/` directory (or the working directory, depending on script).

## Citation

If you use this code, please cite:

Corrêa et al. Prefrontal cortex glucocorticoid receptors gate the balance between salience and default-mode retrieval networks during fear memory consolidation. *bioRxiv* (in preparation/submission).
Code archive: Corrêa, M.S. & Fornari, R.V. (2026). dmPFC_MIF: Analysis code for c-Fos functional connectivity and graph-theoretic characterization of fear memory networks (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.22025170

## Contact

Moisés dos Santos Corrêa, Ph.D. — mscorrea.86@gmail.com
Raquel Vecchio Fornari, Ph.D. (Lead Contact) — raquel.fornari@ufabc.edu.br

## License

This project is licensed under the **MIT License**. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this software, provided the original copyright notice is retained. See [LICENSE](LICENSE) for full terms.
