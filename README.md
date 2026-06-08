# DLS: Brain Connectivity Modelling Through Joint Estimation of Parcels and Gradients

## Overview
DLS (Dense Connectome to Low-rank and Sparse components) is a framework for modelling whole-brain connectivity topography from resting-state fMRI. It decomposes the dense connectome into low-rank components associated with functional segregation and sparse components that capture connectivity gradients. By combining low-rank + sparse decomposition with embedding methods such as ICA, ISOMAP and UMAP, DLS provides a unified representation of both abrupt and gradual variations in brain connectivity, achieving a more accurate reconstruction of the dense connectome and producing global gradients that closely align with task-based brain organisation.

## Installation

1. If you have Git installed, first clone the repository:
```
git clone https://github.com/arekavandi/DLS.git
```
2. Go to the project folder and make a new environment and install the required packages by
```
cd DLS
conda env create -f environment.yml
```
3. Change the environment
```
conda activate dls
```
### Simulated Data Tutorial

The notebook `simulated.ipynb` is designed primarily for **learning and demonstration purposes**. It provides a step-by-step walkthrough of the DLS framework using simulated data and includes comparisons with alternative methods.

To maximize transparency and educational value, this notebook directly calls the internal functions used by the `DLSModel` class, allowing users to inspect each stage of the pipeline individually, including:

- Dense connectivity estimation
- Low-rank and sparse decomposition
- Incremental PCA (MIGP)
- Spatial ICA estimation
- Gradient estimation (UMAP/ISOMAP)
- Comparisons with existing approaches

This notebook is recommended for users who want to understand the methodology and reproduce individual processing steps.

---

### Real Data Analysis

For real neuroimaging datasets (e.g., HCP CIFTI time-series data), users should use the:

```python
DLSModel

