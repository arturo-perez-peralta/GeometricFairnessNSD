# Algorithmic Fairness in Neural Sheaf Diffusion via Geometrical Laplacian modifications

Exploration of geometrical ideas to achieve fairness in ML models.

## Dependencies

### Python Version
The code is designed to work with **Python 3.12.2**.
It is important to ensure that the correct Python version is installed to avoid compatibility issues.

### Installing Dependencies
To reproduce the environment used in this project, follow these steps:

1. **Ensure Python 3.12.2 is Installed**:
2. **Install Dependencies**:
   - The project dependencies are listed in the `environment_gpy.yml` file.
   - Use the following command to install them:
    ```bash
    conda env create -f environment_gpu.yml
    ```

## Reproduce our results

* `sh_main.sh`: Performs the grid search over real-world datasets, results stored in `results/filter/experiment`.
* `sh_synthetic.sh`: Performs the synthetic experiment, results stored in `results/filter/synthetic`.
* `nb_german.ipynb`: Visualization of the spectrum and Dirichlet energy of the laplacian modifications on the German dataset, figures stored in `results/filter/figs`
* `nb_results.ipynb`: Notebook generating the rest of the figures and tables.
* `sh_tradeoff.sh`: Performs the sensitivity analysis on the Pokec-z dataset, results stored in `results/filter/tradeoff`.
* `sh_ablation.sh`: Performs the ablation study on the NSD architecture, implementing our methods for a GCN and GAT. Results stored in `results/filter/ablation`.

## Cite this work
TBA
