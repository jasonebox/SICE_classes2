#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:47:15 2023

@author: jason
"""

#%%
make_gif=1
import os

os.chdir('/Users/jason/Dropbox/S3/SICE_classes2/qgis/')

sensors=['S2','S3']

if make_gif:
    print("making gif")
    animpath='/Users/jason/Dropbox/S3/SICE_classes2/qgis/'    
    import imageio.v2 as imageio

    if make_gif == 1:
        images=[]
        for sensor in sensors:
            images.append(imageio.imread(f'{animpath}Sukkertoppen_20190802 {sensor}.jpg'))
        imageio.mimsave(f'{animpath}Sukkertoppen_20190802.gif', images,   duration=1000)