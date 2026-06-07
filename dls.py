# Authors: Aref Miri Rekavandi

import nibabel as nib
import numpy as np
from pathlib import Path
import random
from functions import utils_gradients as utils

def timedemean(matrix):
    # Calculate the mean of each column
    column_means = np.mean(matrix, axis=1)
    return (matrix.T - column_means).T

class Gradient:
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

    def __init__(self):


    def data_to_corr (self, data_dir = None, N_sub= None, res = 0.25, k = 2, type = 'timeseries'):

        self.factor = res
        self.k = k
        
        """load data from the path
        Returns
        -------
        shirnked 2D array
        """

        subject_ids=utils.get_random_files(data_dir, N = N_sub)

        print(subject_ids)

        assert subject_ids.endswith(".dtseries.nii"), f"File {data_dir} must include only CIFTI time series (.dtseries.nii)"
            
        print('Number of subjects: ',len(set(subject_ids)))
        
        print('CIFTI Files are: ', subject_ids)
        
        FIRST_TIME=True

        for i,subject_id in enumerate(subject_ids):
            
            data_file = data_dir / subject_id
            
            values = nib.load(data_file)

            # Extract data and append to the list
            data_array = np.array(values.get_fdata().T, dtype=np.float64)
            
            if FIRST_TIME:
                # Map indices for left and right hemispheres
                map_left = 1
                map_right = 2
                pointer = 1
                indices_for_left = values.header.matrix._mims[pointer]._maps[map_left]._vertex_indices
                indices_for_right = values.header.matrix._mims[pointer]._maps[map_right]._vertex_indices
                Nvleft = values.header.matrix._mims[pointer]._maps[map_left].surface_number_of_vertices  #1*32492
                left_offset = values.header.matrix._mims[pointer]._maps[map_left].index_offset
                Nvright = values.header.matrix._mims[pointer]._maps[map_right].surface_number_of_vertices  #1*32492
                right_offset = values.header.matrix._mims[pointer]._maps[map_right].index_offset
                leftindices=list(range(left_offset,left_offset+len(indices_for_left)))
                rightindices=list(range(right_offset,right_offset+len(indices_for_right)))
                fakerightindices=list(range(len(indices_for_right)))
                
            if i==0:
                sub_data=timedemean(data_array)
                dataleft=sub_data[leftindices,:]
                dataright=sub_data[rightindices,:]
                leftsurf = nib.load("human.L.inflated.surf.gii")
                rightsurf= nib.load("human.R.inflated.surf.gii")
            
                both_hem_data = np.concatenate((dataleft, dataright), axis=0)
                sampled_data,correspondence,indices_picked=utils.down_sample(both_hem_data,self.factor, self.k, np.concatenate((leftsurf.darrays[0].data[indices_for_left,:] , 100+rightsurf.darrays[0].data[indices_for_right,:] ), axis=0))
            
            else:
                sub_data=timedemean(data_array)
                dataleft=sub_data[leftindices,:]
                dataright=sub_data[rightindices,:]
                both_hem_data = np.concatenate((dataleft, dataright), axis=0)
                sampled_data,correspondence,indices_picked=utils.down_sample(both_hem_data,self.factor, self.k, np.concatenate((leftsurf.darrays[0].data[indices_for_left,:] , 100+rightsurf.darrays[0].data[indices_for_right,:] ), axis=0), correspondence, indices_picked)
            
            
            if FIRST_TIME:
                Dense_C=np.zeros((sampled_data.shape[0],sampled_data.shape[0]))
                Dense_C_train=np.zeros((sampled_data.shape[0],sampled_data.shape[0]))
                Dense_C_val=np.zeros((sampled_data.shape[0],sampled_data.shape[0]))
                FIRST_TIME=False
            Dense_C+=(1/len(subject_ids))*np.corrcoef(sampled_data)

            if i%2==0:
                Dense_C_train+=np.corrcoef(sampled_data)
            else:
                Dense_C_val+=np.corrcoef(sampled_data) 
                
            print(f"Loaded {sp} for subject {subject_id} with sampled connectivity: {Dense_C.shape}")
            
        Dense_C_train=(2/len(subject_ids))*Dense_C_train
        Dense_C_val=(2/len(subject_ids))*Dense_C_val
        full_corr_train_val=np.corrcoef(Dense_C_train.flatten(),Dense_C_val.flatten())[0,1]
        half_corr_train_val=np.corrcoef(Dense_C_train[np.triu_indices(Dense_C_train.shape[0], k=1)],Dense_C_val[np.triu_indices(Dense_C_val.shape[0], k=1)])[0,1]
        print('Train vs Val Full Corr:, ',full_corr_train_val)
        print('Train vs Val Half Corr:, ',half_corr_train_val)
        print(f"Data Concatenation is complete!")
        return Dense_C_train, Dense_C_val
        
    def fit (self, X):
        return X
        
        

