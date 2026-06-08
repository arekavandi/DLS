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

The notebook:

