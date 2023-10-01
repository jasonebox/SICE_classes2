#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:59:15 2023

@author: jason
"""

import geopandas as gpd
points = gpd.read_file('/Users/jason/Dropbox/S3/SICE_classes2/ROIs/Greenland/2019/2019-08-02/2019-08-02_red_snow.shp')
points.to_file('/Users/jason/Dropbox/S3/SICE_classes2/ROIs/Greenland/2019/2019-08-02/exported/2019-08-02_red_snow.shp')
