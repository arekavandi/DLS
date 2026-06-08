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

The notebook `simulated.ipynb` is designed primarily for **learning and demonstration purposes**. It provides a step-by-step walkthrough of the DLS framework using simulated data and includes comparisons with alternative methods. This notebook directly calls the internal functions used by the `DLSModel` class, allowing users to inspect each stage of the pipeline individually, including:

- Low-rank and sparse decomposition
- Spatial ICA estimation
- Gradient estimation (UMAP/ISOMAP)
- Comparisons with existing approaches

This notebook is recommended for users who want to understand the methodology and reproduce individual processing steps.

---

### Real Data Analysis

For real neuroimaging datasets (e.g., HCP CIFTI time-series data), users should use the `DLSModel()` class. 

#### DLSModel Class: Summary of Methods

The `DLSModel` class provides an end-to-end workflow for joint estimation of brain gradients and spatial parcels from dense functional connectivity data. It decomposes connectivity matrices into low-rank and sparse components and uses these methods to extract meaningful brain features.

#### Initialisation

**`__init__(res=0.25, k=2)`**  
- **Purpose:** Initialize the model with spatial downsampling and neighborhood parameters.  
- **Parameters:**  
  - `res`: float, downsampling factor (default 0.25)  
  - `k`: int, number of nearest neighbors for downsampling (default 2)  

#### Data Loading

**`data_to_corr(data_dir=None, N_sub=None, data_type='timeseries')`**  
- **Purpose:** Load CIFTI time-series data and compute group-average dense connectivity matrices.  
- **Parameters:**  
  - `data_dir`: path to subject time-series CIFTI files  
  - `N_sub`: number of subjects to load  
- **Returns:**  
  - `DC_train`, `DC_val`: dense connectivity matrices for training and validation  
- **Notes:** Automatically downsamples vertices and handles left/right hemispheres.


#### Model Fitting and Transformation

**`fit_transform(X, method='UMAP', N=100, g=2, r=9, MR=9, max_iter=250)`**  
- **Purpose:** Decompose dense connectivity into low-rank and sparse components, estimate parcels and gradients.  
- **Parameters:**  
  - `X`: input connectivity matrix  
  - `method`: 'UMAP' or 'ISOMAP' for gradient estimation  
  - `N`: number of neighbors for manifold learning  
  - `g`: number of gradients  
  - `r`: number of parcels  
  - `MR`: max rank for Robust PCA  
- **Outputs:** Updates class attributes:
  - `Ls`: low-rank component  
  - `Ss`: sparse component  
  - `parcels`: spatial ICA components  
  - `grads`: cortical gradients  


#### Visualisation

**`vis_DCs()`**  
- **Purpose:** Visualise low-rank, sparse, and full dense connectivity matrices with eigenvalue curves.  

**`vis_grads(side='left', idx=0)`**  
- **Purpose:** Visualise a selected cortical gradient on the left or right hemisphere.  

**`vis_parcels(side='left', idx=0)`**  
- **Purpose:** Visualise a selected parcel (spatial ICA component) on the left or right hemisphere.  

---

### Notes for Users

- **Simulated data:** The notebook `simulated_data.ipynb` demonstrates internal functions of `DLSModel` for learning and comparison with other methods.  
- **Real data:** For HCP or other datasets, use `DLSModel` and its full set of methods.  
- **Step-by-step instructions:** See the notebook `real_data.ipynb` for detailed workflow examples for the real data experiments.

---
# Citations
If you found this GitHub page helpful, please cite the following paper:
```
@article{rekavandi2026certified,
  title={Brain Connectivity Modelling Through Joint Estimation of Parcels and Gradients},
  author={Miri Rekavandi, Aref and Jbabdi, Saad and Smith, Stephen M.},
  journal={bioRxiv},
  year={2026}
}
```


