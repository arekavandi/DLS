# Author: Aref Miri Rekavandi

import nibabel as nib
import numpy as np
from pathlib import Path
import random
from functions import utils_gradients as utils
from RobustPCA.rpca import RobustPCA
from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap
import umap
import time

def timedemean(matrix):
    # Calculate the mean of each column
    column_means = np.mean(matrix, axis=1)
    return (matrix.T - column_means).T
    
def matrix_MIGP(C, n_dim=1000, d_pca=1000, keep_mean=True):
    """Apply incremental PCA to C
    Inputs:
    C (2D array) : should be wide i.e. nxN where N bigger than n
    We pretend that the matrix C is made of column blocks, each block is
    one 'subject', and 'time' is the column dimension.

    n_dim (int)  : C is split up into nXn_dim matrices
    n_pca (int)  : maximum number of pcs kept (set to n_dim if larger than n_dim)
    keep_mean (bool) : keep the mean of C

    Returns:
    reduced version of C (size nxmin(n_dim,n_pca)
    """
    # Random order for columns of C (create a view rather than copy the data)

    if keep_mean:
        C_mean = np.mean(C, axis=0, keepdims=True)
        print('mean shape: ',C_mean.shape)
        #raise(Exception('Not implemented keep_mean yet!'))

    if d_pca > n_dim:
        d_pca = n_dim

    print('...Starting MIGP')
    _, N = C.shape
    #random_idx = np.random.permutation(N)
    #Cview = C[:, random_idx]
    Cview = C.copy()
    Cview=demean(Cview)
    proj_mat=[]
    W = None
    for i in tqdm(range(0,N,n_dim)):
        data = Cview[:, i:min(i+n_dim, N+1)].T  # transpose to get time as 1st dimension
        if W is not None:
            W = np.concatenate((W, (data)), axis=0)
        else:
            W = (data)
        k = min(d_pca, n_dim)
        _, U  = eigsh(W@W.T, k)

        W = U.T@W
        proj_mat.append(U)
    data = W[:min(W.shape[0], d_pca), :].T

    print(f'...Old matrix size : {C.shape[0]}x{C.shape[1]}')
    print(f'...New matrix size : {data.shape[0]}x{data.shape[1]}')
    return data,proj_mat,C_mean

class Gradient:
    """Dense Connectome to Low-rank + Sparse Components

    Brain Connectivity Modelling Through Joint Estimation of Parcels and Gradients

    Parameters
    ----------
    
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

    def __init__(self, res=0.25, k=2):
        self.k=k
        self.factor=res
        self.L = None
        self.S = None
        self.grad = None
        self.parcel = None
        self.correspondence = None
        self.indices_picked = None

    def data_to_corr (self, data_dir = None, N_sub = None, data_type = 'timeseries'):        
        """load data from the path
        Returns
        -------
        2D array
        """
        if data_dir is None:
            raise ValueError("You must provide data_dir")

        print("Loading from:", data_dir)
        subject_ids=utils.get_random_files(data_dir, N = N_sub)

        print(subject_ids)

        assert all(f.endswith(".dtseries.nii") for f in subject_ids), \
        "All files must be CIFTI time series (.dtseries.nii)"
            
        print('Number of subjects: ',len(set(subject_ids)))
        
        print('CIFTI Files are: ', subject_ids)
        
        FIRST_TIME=True

        for i,subject_id in enumerate(subject_ids):
            
            data_file = Path(data_dir) / subject_id
            
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
                sampled_data,self.correspondence,self.indices_picked=utils.down_sample(both_hem_data,self.factor, self.k, np.concatenate((leftsurf.darrays[0].data[indices_for_left,:] , 100+rightsurf.darrays[0].data[indices_for_right,:] ), axis=0))
            
            else:
                sub_data=timedemean(data_array)
                dataleft=sub_data[leftindices,:]
                dataright=sub_data[rightindices,:]
                both_hem_data = np.concatenate((dataleft, dataright), axis=0)
                sampled_data,self.correspondence,self.indices_picked=utils.down_sample(both_hem_data,self.factor, self.k, np.concatenate((leftsurf.darrays[0].data[indices_for_left,:] , 100+rightsurf.darrays[0].data[indices_for_right,:] ), axis=0), self.correspondence, self.indices_picked)
            
            
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
                
            print(f"Loaded data for subject {subject_id} with sampled connectivity: {Dense_C.shape}")
            
        Dense_C_train=(2/len(subject_ids))*Dense_C_train
        Dense_C_val=(2/len(subject_ids))*Dense_C_val
        full_corr_train_val=np.corrcoef(Dense_C_train.flatten(),Dense_C_val.flatten())[0,1]
        half_corr_train_val=np.corrcoef(Dense_C_train[np.triu_indices(Dense_C_train.shape[0], k=1)],Dense_C_val[np.triu_indices(Dense_C_val.shape[0], k=1)])[0,1]
        print('Train vs Val Full Corr:, ',full_corr_train_val)
        print('Train vs Val Half Corr:, ',half_corr_train_val)
        print(f"Data Concatenation is complete!")
        return Dense_C_train, Dense_C_val
        
    def fit_transform (self, X, method = 'UMAP', N = 100, g = 2, r = 9, MR = 9, max_iter = 250 ):
        rpca = RobustPCA(max_rank=MR,max_iter=250,tol=0.00001*X.shape[0]*X.shape[1],use_fbpca=True)
        rpca.fit(X)
        self.L = rpca.get_low_rank()
    
        pca_approx_dense,proj,C_mean=matrix_MIGP(self.L,n_dim=500, d_pca=r)
        ica = FastICA(n_components=M_ICA,max_iter=2000,tol=0.0005)
        independent_S =ica.fit_transform(pca_approx_dense)
        independent_A =ica.mixing_
        self.parcel=utils.up_sample(independent_S, self.correspondence)
        print('Global connectivity components have been estimated by FastICA!')
        L_up= utils.up_sample(self.L, correspondence)
        self.L=utils.up_sample(Ll_up.T, correspondence)

        
        self.S = rpca.get_sparse()
        if method.lower() == 'isomap':
            print(f'Running ISOMAP with N_neighbor={N}...')
            embedded_I = Isomap(n_components=g,n_neighbors=N).fit_transform(self.S)
            self.grad=utils.up_sample(embedded_I, self.correspondence)
            print(f'Global connectivity gradients have been estimated by ISOMAP!')
        elif method.lower() == 'umap':
            print(f'Running UMAP with N_neighbor={N}...')
            embedded_U = umap.UMAP(n_components=g, metric='euclidean',n_neighbors=N).fit_transform(self.S)
            self.grad=utils.up_sample(embedded_U, self.correspondence)
            print(f'Global connectivity gradients have been estimated by UMAP!')
        else:
            raise ValueError("You must use either ISOMAP or UMAP for gradient estimation method!")
            
        S_up= utils.up_sample(self.S, self.correspondence)
        self.S=utils.up_sample(S_up.T, self.correspondence)
        
        
        

