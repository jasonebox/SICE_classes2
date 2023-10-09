#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:41:27 2023

@authors: Jason, Rasmus, Adrien, Jasper

issues:
    For Jasper: see !! 
    for Rasmus:
    - make relative paths smarter
    - have code integrate better with Thredds, to not have to DL what files are needed locally
    - see !! was 4 now is n_bands (currently 3), lines ~505 and ~339
    - want to re-insert the band data that is masked out in the final classification
        adjust code to not clip data that's outside the training set
    - better results for more training data, e.g. different SZA, sza
        in the training, this code wants to load more than one date
    - how to feed in a different date for the prediction? in this case, monthly means, see !! below ~line 496
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from pathlib import Path
from sklearn import svm
import xarray as xr
import rasterio
from rasterio.mask import mask
from rasterio.transform import Affine
from pyproj import CRS as CRSproj
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import time


st_all = time.time()


def read_S3(fn):
    test_file = Path(fn)
    # print(fn)
    r = np.zeros((5424, 2959)) * np.nan
    if test_file.is_file():
        print("reading " + fn)
        rx = rasterio.open(fn)
        r = rx.read(1)
        # r[r > 1] = np.nan
    else:
        print("no file")
    return r





# ## change to your system's login name to change dir for local work
if os.getlogin() == "jason":
    base_path = "/Users/jason/Dropbox/S3/SICE_classes2/"
if os.getlogin() == "adrien":
    base_path = "/home/adrien/EO-IO/SICE_classes2/"
if os.getlogin() == "rasmus":
# !! Rasmus paths
    current_path = os.getcwd()
    base_path = os.path.abspath('..')
if os.getlogin() == "Jasper":
    base_path = "E:/Jasper/Denmark/GEUS/SICE_classes2/"

os.chdir(base_path)

# relative paths
path_raw = "./SICE_rasters/"
# path_raw = "/Users/jason/0_dat/S3/opendap/"
path_ROI = "./ROIs/"
path_Figs = "./Figs/"

raster_path = base_path + path_raw

# bands to consider for classification
bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_21"]
bands = ["r_TOA_02", "r_TOA_NDXI_0806", "r_TOA_NDXI_1110", "r_TOA_NDXI_0802", "r_TOA_21"] ; version_name='5bands_3NDXI'
bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_08", "r_TOA_10", "r_TOA_11", "r_TOA_21"] ; version_name='7bands_02_04_06_08_10_11_21'
bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_08", "r_TOA_21"] ; version_name='5bands_02_04_06_08_21'
# bands = ["r_TOA_02", "r_TOA_06", "r_TOA_08", "r_TOA_21"] ; version_name='4bands_02_06_08_21'
n_bands = len(bands)

region_name='Greenland'
datex='2019-08-02'
# run successfully then problems
dates=['2017-07-28','2019-08-02','2020-07-22']
# run successfully
dates=['2017-07-28','2021-07-30']



dates=['2019-08-02'] # incomplete re-labeling

# still to run
# dates=['2017-07-12','2021-07-30','2022-07-31']
# 
# issue with huber_w
# dates=['2017-07-12']

for datex in dates:
    
    year=datex[0:4]
    # datex='2021-07-30'; year='2021'
    # datex='2017-07-28' ; year='2017'
    
    #!! other dates
    # datex = "2017-07-12"; year = "2017"
    # datex = "2020-07-22"; year = "2020"
    # datex='2022-07-31'; year='2022'
    
    
    NDXI=read_S3(f"{path_raw}{region_name}/{year}/{datex}_r_TOA_NDXI_0608.tif")
    b21=read_S3(f"{path_raw}{region_name}/{year}/{datex}_r_TOA_21.tif")
    labels=read_S3(f"{path_raw}/{datex}_labels_5bands_02_04_06_08_21.tif")
    path_S3='/Users/jason/0_dat/S3/opendap/Greenland/'
    BBA=read_S3(f"{path_S3}/{year}/{datex}_albedo_bb_planar_sw.tif")

    # lat=read_S3(f"/Users/jason/Dropbox/S3/ancil/lat.tif")
    # np.shape(lat)
    
    #%% imshow
    nj = 5424 ; ni = 2959  # all greenland raster dimensions
    #SW
    i0=500 ; i1=1000
    j0=3000 ; j1=nj-1
    #Sukkertoppen
    i0=500 ; i1=800
    j0=3800 ; j1=4000
    labels_temp=labels[j0:j1,i0:i1]

    NDXI_temp=NDXI[j0:j1,i0:i1]
    BBA_temp=BBA[j0:j1,i0:i1]
    b21_temp=b21[j0:j1,i0:i1]

    # NDXI_temp[labels_temp==3]=1
    plt.imshow(BBA_temp)
    # plt.imshow(labels_temp)
    # plt.imshow(NDXI_temp)
    # plt.imshow(b21_temp)
    plt.axis("Off")
    plt.colorbar()
    
    #%%
    from numpy.polynomial.polynomial import polyfit
    from scipy import stats
    from scipy.stats import gaussian_kde


    regions=['dark','light']
    for rr,region in enumerate(regions):
        if rr==1:
            # v=np.where(labels_temp==3)
            x=-NDXI_temp[labels_temp==3]
            x[x<0]=np.nan
            y=BBA_temp[labels_temp==3]
            
            
            # y[y>0.45]-=0.17
            y[y<0.46]+=0.17
            # if rr==0:
            #     y[y>0.45]=np.nan
            # if rr==1:
            #     y[y<0.46]=np.nan
            
            
            # v=np.where(np.isfinite(y))
            v=np.where(((np.isfinite(y))&(np.isfinite(x))))
            # plt.plot(x,y,'.',c=(0.8, 0., 0.),label=lab) 
            b, m = polyfit(x[v[0]], y[v[0]], 1)
            coefs=stats.pearsonr(x[v[0]],y[v[0]])
            # y=b21_temp[labels_temp==3]
        
            #  plot 
            ly='x'
            
            fs=12 # fontsize
            
            plt.close()
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # plt.scatter(x,y,marker='.')
            # Calculate the point density
            
            v=np.where(((np.isfinite(x))& (np.isfinite(y))))
            x=x[v] ; y=y[v]
            xy = np.vstack([x,y])
            z = gaussian_kde(xy)(xy)
            ax.scatter(x, y, c=z, s=20)

            xx=[np.min(x),np.max(x)]
            # xx=[0,0.05]
            xx=np.array(xx)
            # plt.plot(xx, b + m * xx, '--',c='k',linewidth=2)
            print((b + m * xx[0])-(b + m * xx[1]))
            plt.ylabel('broadband albedo')#', {region}')
            plt.xlabel('NDIX=(R_665 nm-R_560 nm)/(R_665 nm+R_560 nm)')
            # plt.title('Sukkertoppen ice cap red snow, 2 August 2019')
            yy0=0.615
            # plt.hlines(yy0,xx[0],xx[1],color='k')
            yy1=0.48
            # plt.hlines(yy1,xx[0],xx[1],color='k')
            print(f'range of albedo {yy1-yy0}')
            
            if ly == 'x':plt.show()
            
            if ly == 'p':
                band='classes'
                # opath='/Users/jason/0_dat/S3/opendap/Figs/'+region_name+'/'
                os.system('mkdir -p '+path_Figs)
                figname=path_Figs+datex+'_classes_SVM'+version_name+'.png' 
                plt.savefig(figname, bbox_inches='tight', dpi=600, facecolor='k')
                os.system('open '+figname)
        