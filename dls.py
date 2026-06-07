# Authors: Aref Miri Rekavandi

import numpy as np

class Dls:
    """Dense Connectome to Low-rank + Sparse Components

    Brain Connectivity Modelling Through Joint Estimation of Parcels and Gradients

    Parameters
    ----------
    lamb : positive float
        Sparse component coefficient.
        if user doesn't set it:
            lamb = 1/sqrt(max(M.shape))
        A effective default value from the reference.

    mu : positive float
        Coefficient for augmented lagrange multiplier
        if user doesn't set it:
            n1, n2 = M.shape
            mu = n1*n2/4/norm1(M) # norm1(M) is M's l1-norm
        A effective default value from the reference.

    max_rank : positive int
        The maximum rank allowed in the low rank matrix
        default is None --> no limit to the rank of the low
        rank matrix.

    tol : positive float
        Convergence tolerance

    max_iter : positive int
        Maximum iterations for alternating updates

    use_fbpca : bool
        Determine if use fbpca for SVD. fbpca use Fast Randomized SVDself.
        default is False

    fbpca_rank_ratio : float, between (0, 1]
        If max_rank is not given, this sets the rank for fbpca.pca()
        fbpca_rank = int(fbpca_rank_ratio * min(M.shape))

    Attributes:
    -----------
    L : 2D array
        Lower rank dense 2D matrix

    S : 2D array
        Sparse but not low-rank 2D matrix

    Reference:
    ----------
    Brain Connectivity Modelling Through Joint Estimation of Parcels and Gradients
    
    https://github.com/arekavandi/DLS

    """

    def __init__(self, lamb=None, mu=None, max_rank=None, tol=1e-6, max_iter=100, use_fbpca=False, fbpca_rank_ratio=0.2):
        self.lamb = lamb

    def corr_load (self, X, tau):
        
        """load data from the path
        Returns
        -------
        shirnked 2D array
        """
        return X
    def fit (self, X):
        return X
        
        

