"""Build a hierarchie of parcels from a parcelation
"""

import pandas as pd
import numpy as np
import atlas_map as am
import nibabel as nb
import nibabel.processing as ns
import SUITPy as suit
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import pickle
from util import *
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from numpy.linalg import eigh, norm
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


base_dir = '.'

def show_parcel_profile(p, profiles, conditions, datasets, show_ds='all', ncond=5, print=True):
    """Returns the functional profile for a given parcel either for selected dataset or all datasets
    Args:
        profiles: parcel scores for each condition in each dataset
        conditions: condition names of each dataset
        datasets: dataset names
        show_ds: selected dataset
                'Mdtb'
                'Pontine'
                'Nishimoto'
                'Ibc'
                'Hcp'
                'all'
        ncond: number of highest scoring conditions to show

    Returns:
        profile: condition names in order of parcel score

    """
    if show_ds =='all':
        # Collect condition names in order of parcel score from all datasets
        profile = []
        for d,dataset in enumerate(datasets):
            cond_name = conditions[d]
            cond_score = profiles[d][:,p].tolist()
            # sort conditions by condition score
            dataset_profile = [name for _,name in sorted(zip(cond_score,cond_name),reverse=True)]
            profile.append(dataset_profile)
            if print:
                print('{} :\t{}'.format(dataset, dataset_profile[:ncond]))

    else:
        # Collect condition names in order of parcel score from selected dataset
        d = datasets.index(show_ds)
        cond_name = conditions[d]
        cond_score = profiles[d][:,p].tolist()

        # sort conditions by condition score
        dataset_profile = [name for _,name in sorted(zip(cond_score,cond_name))]
        profile = dataset_profile
        if print:
            print('{} :\t{}'.format(datasets[d], dataset_profile[:ncond]))

    return profile

def get_clusters(Z,K,num_cluster):
    cluster = np.zeros((K+Z.shape[0]),dtype=int)
    next_cluster = 1
    for i in np.arange(Z.shape[0]-num_cluster,-1,-1):
        indx = Z[i,0:2].astype(int)
        # New cluster number
        if (cluster[i+K]==0):
            cluster[i+K]  = next_cluster
            cluster[indx] = next_cluster
            next_cluster += 1
        # Cluster already assigned - just pass down
        else:
            cluster[indx]=cluster[i+K]
    return cluster[:K],cluster[K:]

def agglomative_clustering(similarity,
                        sym=False,
                        num_clusters=5,
                        method = 'ward',
                        plot=True,
                        groups = ['0','A','B','C','D','E','F','G'],
                        cmap=None):
    # setting distance_threshold=0 ensures we compute the full tree.
    # plot the top three levels of the dendrogram
    K = similarity.shape[0]
    sym_sim=(similarity+similarity.T)/2
    dist = squareform(1-sym_sim.round(5))
    Z = linkage(dist,method)
    cleaves,clinks = get_clusters(Z,K,num_clusters)

    ax=plt.gca()
    R = dendrogram(Z,color_threshold=-1,no_plot=not plot) # truncate_mode="level", p=3)
    leaves = R['leaves']
    # make the labels for the dendrogram
    labels = np.empty((K,),dtype=object)

    current = -1
    for i,l in enumerate(leaves):
        if cleaves[l]!=current:
            num=1
            current = cleaves[l]
        labels[i]=f"{groups[cleaves[l]]}{num}"
        num+=1

    # Make labels for mapping
    current = -1
    if sym:
        labels_map = np.empty((K*2+1,),dtype=object)
        clusters = np.zeros((K*2,),dtype=int)
        labels_map[0] = '0'
        for i,l in enumerate(leaves):
            if cleaves[l]!=current:
                num=1
                current = cleaves[l]
            labels_map[l+1]   = f"{groups[cleaves[l]]}{num}L"
            labels_map[l+K+1] = f"{groups[cleaves[l]]}{num}R"
            clusters[l] = cleaves[l]
            clusters[l+K] = cleaves[l]
            num+=1
    else:
        labels_map = np.empty((K+1,),dtype=object)
        clusters = np.zeros((K,),dtype=int)
        labels_map[0] = '0'
        for i,l in enumerate(leaves):
            labels_map[l+1]   = labels[i]
            clusters[l] = cleaves[l]
    if plot & (cmap is not None):
        ax.set_xticklabels(labels)
        ax.set_ylim((-0.2,1.5))
        draw_cmap(ax,cmap,leaves,sym)
    return labels_map,clusters,leaves

def draw_cmap(ax,cmap,leaves,sym):
    """ Draws the color map on the dendrogram"""
    K = len(leaves)
    for k in range(K):
        rect = Rectangle((k*10, -0.05), 10,0.05,
        facecolor=cmap(leaves[k]+1),
        fill=True,
        edgecolor=(0,0,0,1))
        ax.add_patch(rect)
    if sym:
        for k in range(K):
            # Left:
            rect = Rectangle((k*10, -0.1), 10,0.05,
            facecolor=cmap(leaves[k]+1+K),
            fill=True,
            edgecolor=(0,0,0,1))
            ax.add_patch(rect)

def calc_mds(G,center=False):
    N = G.shape[0]
    if center:
        H = np.eye(N)-np.ones((N,N))/N
        G = H @ G @ H
    G = (G + G.T)/2
    Glam, V = eigh(G)
    Glam = np.flip(Glam,axis=0)
    V = np.flip(V,axis=1)
    W = V[:,:3] * np.sqrt(Glam[:3])

    return W

"""elif type=='hsv':
        Sat=np.sqrt(W[:,0:1]**2)
        Sat = Sat/Sat.max()
        Hue=(np.arctan2(W[:,1],W[:,0])+np.pi)/(2*np.pi)
        Val = (W[:,2]-W[:,2].min())/(W[:,2].max()-W[:,2].min())*0.5+0.4
        rgb = mpl.colors.hsv_to_rgb(np.c_[Hue,Sat,Val])
    elif type=='hsv2':
        Sat=np.sqrt(V[:,0:1]**2)
        Sat = Sat/Sat.max()
        Hue=(np.arctan2(V[:,1],V[:,0])+np.pi)/(2*np.pi)
        Val = (V[:,2]-V[:,2].min())/(V[:,2].max()-V[:,2].min())*0.5+0.4
        rgb = mpl.colors.hsv_to_rgb(np.c_[Hue,Sat,Val])
    elif type=='rgb_cluster':

        rgb = (W-W.min())/(W.max()-W.min()) # Scale between zero and 1
    else:
        raise(NameError(f'Unknown Type: {type}'))
"""

def get_target(cmap):
    if isinstance(cmap,str):
        cmap = mpl.cm.get_cmap(cmap)
    rgb=cmap(np.arange(cmap.N))
    # plot_colormap(rgb)
    tm=np.mean(rgb[:,:3],axis=0)
    A=rgb[:,:3]-tm
    tl,tV=eigh(A.T@A)
    tl = np.flip(tl,axis=0)
    tV = np.flip(tV,axis=1)
    return tm,tl,tV

def make_orthonormal(U):
    """Gram-Schmidt process to make
    matrix orthonormal"""
    n = U.shape[1]
    V=U.copy()
    for i in range(n):
        prev_basis = V[:,0:i]     # orthonormal basis before V[i]
        rem = prev_basis @ prev_basis.T @ U[:,i]
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[:,i] = U[:,i] - rem
        V[:,i] /= norm(V[:,i])
    return V

def plot_colormap(rgb):
    N,a = rgb.shape
    if a==3:
        rgb = np.c_[rgb,np.ones((N,))]
    rgba = np.r_[rgb,[[0,0,0,1],[1,1,1,1]]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(rgba[:,0],rgba[:,1], rgba[:,2], marker='o',s=70,c=rgba)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    m=np.mean(rgb[:,:3],axis=0)
    A=rgb[:,:3]-m
    l,V=eigh(A.T@A)
    l = np.flip(l,axis=0)
    V = np.flip(V,axis=1)

    B = V * np.sqrt(l) * 0.5
    for i in range(2):
        ax.quiver(m[0],m[1],m[2],B[0,i],B[1,i],B[2,i])
    return m,l,V


def colormap_mds(W,target=None,scale=False,clusters=None,gamma=0.3):
    """Map the simularity structure of MDS to a colormap
    Args:
        W (_type_): _description_
        plot (str, optional): _description_. Defaults to '2d'.
        type (str, optional): _description_. Defaults to 'hsv'.
        target (stg or )
    Returns:
        colormap: _description_
    """
    N = W.shape[0]
    if target is not None:
        tm=target[0]
        tl=target[1]
        tV = target[2]
        m=np.mean(W[:,:3],axis=0)
        A=W-m
        l,V=eigh(A.T@A)
        l = np.flip(l,axis=0)
        V = np.flip(V,axis=1)
        Wm = A @ V @ tV.T
        Wm += tm
    # rgb = (W-W.min())/(W.max()-W.min()) # Scale between zero and 1
    Wm[Wm<0]=0
    Wm[Wm>1]=1
    if clusters is not None:
        M = np.zeros((clusters.max(),3))
        for i in np.unique(clusters):
            M[i-1,:]=np.mean(Wm[clusters==i,:],axis=0)
            Wm[clusters==i,:]=(1-gamma) * Wm[clusters==i,:] + gamma * M[i-1]

    colors = np.c_[Wm,np.ones((N,))]
    colorsp = np.r_[np.zeros((1,4)),colors] # Add empty for the zero's color
    newcmp = ListedColormap(colorsp)
    return newcmp


def save_lut(index,colors,labels,fname):
    L=pd.DataFrame({
            "key":index,
            "R":colors[:,0].round(4),
            "G":colors[:,1].round(4),
            "B":colors[:,2].round(4),
            "Name":labels})
    L.to_csv(fname,header=None,sep=' ',index=False)


def renormalize_probseg(probseg):
    X = probseg.get_fdata()
    xs = np.sum(X,axis=3)
    xs[xs<0.5]=np.nan
    X = X/np.expand_dims(xs,3)
    X[np.isnan(X)]=0
    probseg_img = nb.Nifti1Image(X,probseg.affine)
    parcel = np.argmax(X,axis=3)+1
    parcel[np.isnan(xs)]=0
    dseg_img = nb.Nifti1Image(parcel.astype(np.int8),probseg.affine)
    dseg_img.set_data_dtype('int8')
    # dseg_img.header.set_intent(1002,(),"")
    probseg_img.set_data_dtype('float32')
    # probseg_img.header.set_slope_inter(1/(2**16-1),0.0)
    return probseg_img,dseg_img




if __name__ == "__main__":
    mname = 'Models_04/sym_MdPoNiIb_space-MNISymC3_K-34'
    # make_asymmetry_map(mname)
    analyze_parcel(mname,sym=True)
    # cmap = mpl.cm.get_cmap('tab20')
    # rgb=cmap(np.arange(20))
    # plot_colormap(rgb)
    pass

