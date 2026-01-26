# FairSheafDiffusion

Exploration of topological ideas to debias models through graph configurations.

## Code structure

* `results_nb`: Jupyter notebook generating tables and figures from the results of the experiments.
* `run_experiment.py`: Python code implementing Fair Sheaf Diffusion experiments.
* `run_neural.py`: Python code implementing Neural Sheaf Diffusion models with fair graph laplacians.
* `run_shap.py`: Python code implementing shap values for the Fair Sheaf Diffusion models.
* `exec_grid.sh`: Bash code to run the grid of design choices on the german dataset. It is generated through `utils/experiment_grid.py`.
* `exec_datasets.sh`: Bash code to run the remaining experiments.
* `exec_simulations.sh`: Bash code to run the simulation study.
* `exec_single.sh`: Bash code to run experiments on a single dataset.
* `NSD`: Modified code of the original Neural Sheaf Diffusion implementation, original repo: [https://github.com/twitter-research/neural-sheaf-diffusion/tree/master](https://github.com/twitter-research/neural-sheaf-diffusion/tree/master).
* `utils`: Folder storing the implementation of the different parts of the training pipeline.
    * `data_processing.py`: Implementation of the data processing pipeline: graph creation, train-val split, pre-processing,...
    * `experiment_grid.py`: Script generating `exec_grid.sh` file.
    * `parser.py`: Parsers of `run_experiment.py`, `run_neural.py`, `run_shap.py`.
    * `training.py`: Implementation of the training loop through a custom class.
* `models`: Folder storing the implementation of the Fair Sheaf Diffusion models.
    * `fair_nsd.py`: Implementation of Fair Sheaf Diffusion models.
    * `fair_sheaf.py`: Implementing Neural Sheaf Diffusion models with fair graph laplacians.

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
### Cite this work
TBA