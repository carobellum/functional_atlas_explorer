import numpy as np
import nibabel as nb
import SUITPy as suit
import pickle
from pathlib import Path
import atlas_map as am
import pandas as pd
import matplotlib.pyplot as plt
# import generativeMRF.evaluation as ev

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    pass
    # raise(NameError('Could not find base_dir'))


def plot_data_flat(data,atlas,
                    cmap = None,
                    dtype = 'label',
                    cscale = None,
                    labels = None,
                    render='matplotlib',
                    colorbar = False):
    """ Maps data from an atlas space to a full volume and
    from there onto the surface - then plots it. 

    Args:
        data (_type_): _description_
        atlas (str): Atlas code ('SUIT3','MNISymC3',...)
        cmap (_type_, optional): Colormap. Defaults to None.
        dtype (str, optional): 'label' or 'func'
        cscale (_type_, optional): Color scale 
        render (str, optional): 'matplotlib','plotly'

    Returns:
        ax: Axis / figure of plot
    """
    # Plot Data from a specific atlas space on the flatmap
    suit_atlas = am.get_atlas(atlas,base_dir + '/Atlases')
    Nifti = suit_atlas.data_to_nifti(data)
    
    # Figure out correct mapping space 
    if atlas[0:4]=='SUIT':
        map_space='SUIT'
    elif atlas[0:7]=='MNISymC':
        map_space='MNISymC'
    else:
        raise(NameError('Unknown atlas space'))

    # Plotting label 
    if dtype =='label':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='mode',
            space=map_space,ignore_zeros=True)
        ax = suit.flatmap.plot(surf_data, 
                render=render,
                cmap=cmap, 
                new_figure=False,
                label_names = labels,
                overlay_type='label',
                colorbar= colorbar)
    # Plotting funtional data 
    elif dtype== 'func':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='nanmean',
            space=map_space)
        ax = suit.flatmap.plot(surf_data, 
                render=render,
                cmap=cmap,
                cscale = cscale,
                new_figure=False,
                overlay_type='func',
                colorbar= colorbar)
    else:
        raise(NameError('Unknown data type'))
    return ax

def plot_multi_flat(data,atlas,grid,
                    cmap = None,
                    dtype = 'label',
                    cscale = None,
                    titles=None,
                    colorbar = False):
    """Plots a grid of flatmaps with some data 

    Args:
        data (array): NxP array of data 
        atlas (str): Atlas code ('SUIT3','MNISymC3',...)
        grid (tuple): (rows,cols) grid for subplot 
        cmap (colormap): Color map Defaults to None.
        dtype (str, optional):'label' or 'func'
        cscale (_type_, optional): Scale of data (None)
        titles (_type_, optional): _description_. Defaults to None.
    """
    for i in range(data.shape[0]):
        plt.subplot(grid[0],grid[1],i+1)
        plot_data_flat(data[i,:],atlas,
                    cmap = cmap,
                    dtype = dtype,
                    cscale = cscale,
                    render='matplotlib',
                    colorbar = (i==0) & colorbar) 
        if titles is not None: 
            plt.title(titles[i])
