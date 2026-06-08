# Author: Aref Miri Rekavandi

import nibabel as nib
from nilearn import plotting 
import numpy as np
from pathlib import Path
import random
from functions import utils_gradients as utils
from RobustPCA.rpca import RobustPCA
from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap
import umap
import time
import tqdm
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import colorcet as cc

def add_zoom(ax, img, zoom=6):
    """Add zoomed inset to an axis."""
    x1, x2, y1, y2 = 1000, 2000, 2000, 1000
    region=(x1, x2, y1, y2)
    # Create zoomed inset
    axins = zoomed_inset_axes(ax, zoom, loc=3)
    axins.imshow(img, vmin=ax.images[0].get_clim()[0], vmax=ax.images[0].get_clim()[1])
    
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)  # invert y-axis
    axins.set_xticks([])
    axins.set_yticks([])
    
    # Mark the area on the main plot
    mark_inset(ax, axins, loc1=4, loc2=2, fc="none", ec="red", lw=1)

def timedemean(matrix):
    # Calculate the mean of each column
    column_means = np.mean(matrix, axis=1)
    return (matrix.T - column_means).T


def demean(X, axis=0):
   # print(np.mean(X, axis=axis, keepdims=True).shape)
    return X - np.mean(X, axis=axis, keepdims=True)
    
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
    for i in tqdm.tqdm(range(0,N,n_dim)):
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
        self.Ls = None
        self.Ss = None
        self.DC_train = None
        self.DC_val = None
        self.grads = None
        self.parcels = None
        self.correspondence = None
        self.indices_picked = None
        self.Nvleft = None
        self.Nvright = None
        self.indices_for_left = None
        self.indices_for_right = None

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
                self.indices_for_left = indices_for_left
                indices_for_right = values.header.matrix._mims[pointer]._maps[map_right]._vertex_indices
                self.indices_for_right = indices_for_right
                Nvleft = values.header.matrix._mims[pointer]._maps[map_left].surface_number_of_vertices  #1*32492
                self.Nvleft = Nvleft
                left_offset = values.header.matrix._mims[pointer]._maps[map_left].index_offset
                Nvright = values.header.matrix._mims[pointer]._maps[map_right].surface_number_of_vertices  #1*32492
                self.Nvright = Nvright
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
        self.DC_train = Dense_C_train
        self.DC_val = Dense_C_val
        return Dense_C_train, Dense_C_val
        
    def fit_transform (self, X, method = 'UMAP', N = 100, g = 2, r = 9, MR = 9, max_iter = 250 ):
        rpca = RobustPCA(max_rank=MR,max_iter=250,tol=0.00001*X.shape[0]*X.shape[1],use_fbpca=True)
        rpca.fit(X)
        self.Ls = rpca.get_low_rank()
    
        pca_approx_dense,proj,C_mean=matrix_MIGP(self.Ls,n_dim=500, d_pca=r)
        ica = FastICA(n_components=r,max_iter=2000,tol=0.0005)
        independent_S =ica.fit_transform(pca_approx_dense)
        independent_A =ica.mixing_
        self.parcels=utils.up_sample(independent_S, self.correspondence)
        print('Global connectivity components have been estimated by FastICA!')
        #L_up= utils.up_sample(self.L, self.correspondence)
        #self.L=utils.up_sample(L_up.T, self.correspondence)

        
        self.Ss = rpca.get_sparse()
        if method.lower() == 'isomap':
            print(f'Running ISOMAP with N_neighbor={N}...')
            embedded_I = Isomap(n_components=g,n_neighbors=N).fit_transform(self.Ss)
            self.grads=utils.up_sample(embedded_I, self.correspondence)
            print(f'Global connectivity gradients have been estimated by ISOMAP!')
        elif method.lower() == 'umap':
            print(f'Running UMAP with N_neighbor={N}...')
            embedded_U = umap.UMAP(n_components=g, metric='euclidean',n_neighbors=N).fit_transform(self.Ss)
            self.grads=utils.up_sample(embedded_U, self.correspondence)
            print(f'Global connectivity gradients have been estimated by UMAP!')
        else:
            raise ValueError("You must use either ISOMAP or UMAP for gradient estimation method!")
            
        #S_up= utils.up_sample(self.S, self.correspondence)
        #self.S=utils.up_sample(S_up.T, self.correspondence)
    
    def vis_DCs(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 3))
        # Plot 1
        im0 = axs[0].imshow(self.Ls, vmin=-0.1, vmax=0.3)
        axs[0].set_title(r'$\hat{\text{L}}$')
        axs[0].set_xlabel('Nv')
        axs[0].set_ylabel('Nv')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        fig.colorbar(im0, ax=axs[0])
        add_zoom(axs[0], self.Ls)
        
        # Plot 2
        im1 = axs[1].imshow(self.Ss, vmin=-0.1, vmax=0.1)
        axs[1].set_title(r'$\hat{\text{S}}$')
        axs[1].set_xlabel('Nv')
        axs[1].set_ylabel('Nv')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.colorbar(im1, ax=axs[1])
        add_zoom(axs[1], self.DC_train)
        
        # Plot 3
        im2 = axs[2].imshow(self.DC_train, vmin=-0.1, vmax=0.3)
        axs[2].set_title('DC')
        axs[2].set_xlabel('Nv')
        axs[2].set_ylabel('Nv')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        fig.colorbar(im2, ax=axs[2])
        add_zoom(axs[2], self.DC_train)
        
        plt.tight_layout()
        plt.show()
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 3))
        eigenvalues, eigenvectors=eigsh(self.Ls,9)
        ascending_indices = eigenvalues.argsort() 
        descending_indices = ascending_indices[::-1] 
        axs[0].plot(eigenvalues[descending_indices], '.', label='Eigvalues')
        axs[0].set_title(r'Eigen Curve of $\hat{\text{L}}$')
        axs[0].set_xlabel('Index')
        axs[0].set_ylabel('Value')
        
        eigenvalues, eigenvectors=eigsh(self.Ss,5*9)
        ascending_indices = eigenvalues.argsort() 
        descending_indices = ascending_indices[::-1] 
        axs[1].plot(eigenvalues[descending_indices], '.', label='Eigvalues')
        axs[1].set_title(r'Eigen Curve of $\hat{\text{S}}$')
        axs[1].set_xlabel('Index')
        axs[1].set_ylabel('Value')
        
        eigenvalues, eigenvectors=eigsh(self.DC_train,5*9)
        ascending_indices = eigenvalues.argsort() 
        descending_indices = ascending_indices[::-1] 
        axs[2].plot(eigenvalues[descending_indices], '.', label='Eigvalues')
        axs[2].set_title('Eig curve of DC')
        axs[2].set_xlabel('Index')
        axs[2].set_ylabel('Value')
        plt.show()
        
    def vis_grads (self, side = 'left', idx=0):
        if idx > self.grads.shape[1]:
            raise ValueError("Please pick the right index within the range!")
        if side.lower() == 'left'
            print(f'{idx+1}-Gradient: Left Hemisphere')
            mymap = np.full(self.Nvleft, np.nan)
            mymap[self.indices_for_left]=self.grads[:len(self.indices_for_left),idx]-np.percentile(self.grads[:,idx],0)   
            view = plotting.view_surf(surf_mesh = 'human.L.inflated.surf.gii', symmetric_cmap=False,
                               surf_map  = mymap,vmin=0, vmax=(np.nanpercentile(self.grads[:,idx],100)-np.nanpercentile(self.grads[:,idx],0)),
                               cmap      =  cc.m_rainbow)
            return view
        elif side.lower() == 'right':
            print(f'{idx+1}-Gradient: Right Hemisphere')
            mymap = np.full(self.Nvright, np.nan)
            mymap[self.indices_for_right]=self.grads[len(self.indices_for_left):,idx]-np.nanpercentile(self.grads[:,idx],0)
            view = plotting.view_surf(surf_mesh = 'human.R.inflated.surf.gii', symmetric_cmap=False,
                   surf_map  = mymap, vmin=0, vmax=(np.nanpercentile(self.grads[:,idx],100)-np.nanpercentile(embedded_Il_n1[:,idx],0)),
                   cmap      = cc.m_rainbow)
            return view
        else:
            raise ValueError("Please select either right or left!")

                
                
                
        
