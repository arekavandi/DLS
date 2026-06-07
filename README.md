# Brain Connectivity Modelling Through Joint Estimation of Parcels and Gradients
## (DLS)
## Overview
DLS (Dense Connectome to Low-rank and Sparse components) is a framework for modelling whole-brain connectivity topography from resting-state fMRI. It decomposes the dense connectome into low-rank components associated with functional segregation and sparse components that capture connectivity gradients. By combining low-rank + sparse decomposition with embedding methods such as ICA, ISOMAP and UMAP, DLS provides a unified representation of both abrupt and gradual variations in brain connectivity, achieving a more accurate reconstruction of the dense connectome and producing global gradients that closely align with task-based brain organisation.

We included all the files/models/datasets required to run the project and there is no need to download external files. 

## Installation
Synthetic simulation requires the following Python libraries/packages:
```
plotly
matplotlib
numpy 
scipy 
