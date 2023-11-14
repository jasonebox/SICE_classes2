#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:14:17 2023

@author: jason
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


datex='2019_08_02'
datex='2021_07_03'
version_name='sicemachine'

fn='/Users/jason/Dropbox/S3/SICE_classes2/output/2019_08_02_SICE_surface_classes.tif'
fn=f'/Users/jason/Dropbox/S3/SICE_classes2/output/{datex}_SICE_surface_classes.tif'


features=['dark_ice','bright_ice','red_snow','lakes','flooded_snow','melted_snow','dry_snow']
            
nams=features

ly='p'

bandx = rasterio.open(fn)
profile=bandx.profile
classes=bandx.read(1)

fs=12 # fontsize

plt.close()
fig, ax = plt.subplots(figsize=(10, 10))

co=150

palette = np.array([
            [0,0,0], # 0 NaN
            [200,100,200],   # 1 dark bare ice
            [co,co,co],   # 2 bright bare ice
            [255, 0, 0], # 3 red snow
            [255,255,0], # 4 lakes
            [100,100,250], # 5 flooded snow
            [255,200,200],   # 6 melted snow
            [255, 255, 255], # 7 dry snow
            ])

classesx=classes.copy()
classesx+=1
classesx[np.isnan(classesx)]=0
RGB=palette[classesx.astype(int)]
cntr=ax.imshow(RGB)

plt.axis('off')

mult=0.6
xx0=0.6 ; yy0=0.04 ; dy=0.02 ; cc=0
for i,nam in enumerate(nams):
    plt.text(xx0, yy0+dy*cc,nam,
              color=palette[i+1]/255,
              transform=ax.transAxes, fontsize=fs*mult,ha="left")
    cc+=1

cc=0
xx0=0.015 ; yy0=0.955
xx0=0.62 ; yy0=0.2
mult=0.8

props = dict(boxstyle='round', facecolor='k', alpha=1,edgecolor='k')
plt.text(xx0, yy0, datex,
        fontsize=fs*mult,color='w',bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5

if ly == 'x':plt.show()

if ly == 'p':
    band='classes'
    # opath='/Users/jason/0_dat/S3/opendap/Figs/'+region_name+'/'
    path_Figs='./Figs'
    path_Figs='/Users/jason/Dropbox/S3/SICE_classes2/Figs/'
    os.system('mkdir -p '+path_Figs)
    datexx=datex.replace('_','-')
    figname=f'{path_Figs}{datexx}_{version_name}.png' 
    plt.savefig(figname, bbox_inches='tight', dpi=600, facecolor='k')
    os.system('open '+figname)

