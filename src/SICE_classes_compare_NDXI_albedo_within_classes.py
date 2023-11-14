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
# import time


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
        print("no file",fn)
    return r


def normalisedx(write_out,fn_band_A,fn_band_B,ofile):
    # normalised difference index
    test_file = Path(fn_band_A)
    if test_file.is_file():
        # print(fn_band_A)
        # print(fn_band_B)
        band_Ax = rasterio.open(fn_band_A)
        profile=band_Ax.profile
        band_A=band_Ax.read(1)
    
        band_Bx = rasterio.open(fn_band_B)
        profile=band_Bx.profile
        band_B=band_Bx.read(1)

        #(band 8-band6)/(band 8 + band6)
        normalised=(band_A-band_B)/(band_A+band_B)
        # normalised[normalised<-0.2]=np.nan
        # normalised[normalised>0.2]=np.nan
        if write_out:
            # resx = (x[1] - x[0])
            # resy = (y[1] - y[0])
            # transform = Affine.translation((x[0]),(y[0])) * Affine.scale(resx, resy)
            with rasterio.Env():
                with rasterio.open(ofile, 'w', **profile) as dst:
                    dst.write(normalised, 1)
            # with rasterio.open(
            #     ofile,
            #     'w', #**profile,
            #     driver='GTiff',
            #     height=normalised.shape[0],
            #     width=normalised.shape[1],
            #     count=1,
            #     compress='lzw',
            #     dtype=normalised.dtype,
            #     # dtype=rasterio.uint8,
            #     crs=PolarProj,
            #     transform=transform,
            #     ) as dst:
            #         dst.write(normalised, 1)
    else:
        print('file missing')
    return normalised

def opentiff(filename):
    
    "Input: Filename of GeoTIFF File "
    "Output: xgrid,ygrid, data paramater of Tiff, the data projection"
   
    da = xr.open_rasterio(filename)
    proj = CRSproj.from_string(da.crs)


    transform = Affine(*da.transform)
    elevation = np.array(da.variable[0],dtype=np.float32)
    nx,ny = da.sizes['x'],da.sizes['y']
    x,y = np.meshgrid(np.arange(nx,dtype=np.float32), np.arange(ny,dtype=np.float32)) * transform

    da.close()
   
    return x,y,elevation,proj


# ## change to your system's login name to change dir for local work
if os.getlogin() == "jason":
    base_path = "/Users/jason/Dropbox/S3/SICE_classes2/"


os.chdir(base_path)

# relative paths
path_raw = "./SICE_rasters/"
path_raw = "/Users/jason/0_dat/S3/opendap/"
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



dates=['2019-08-02'] 
dates=['2021-08-23'] 
# dates=['2023-07-14'] 
# dates=['2023-07-08'] 

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
    
    
    do_generate_rasters=0
    
    if do_generate_rasters:
        # for red snow
        normalised=normalisedx(1,f"{path_raw}{region_name}/{year}/{datex}_r_TOA_06.tif",
              f"{path_raw}{region_name}/{year}/{datex}_r_TOA_08.tif",
              f"{path_raw}{region_name}/{year}/{datex}_r_TOA_NDXI_0608.tif")
    
        # normalised=normalisedx(1,f"{path_raw}{region_name}/{year}/{datex}_r_TOA_08.tif",
        #       f"{path_raw}{region_name}/{year}/{datex}_r_TOA_02.tif",
        #       f"{path_raw}{region_name}/{year}/{datex}_r_TOA_NDXI_0802.tif")
    
        # for flooded areas
        # ratio_BRx=ratio_image(1,f"{path_raw}{region_name}/{year}/{datex}_r_TOA_02.tif",
        #       f"{path_raw}{region_name}/{year}/{datex}_r_TOA_08.tif",
        #       f"{path_raw}{region_name}/{year}/{datex}_r_TOA_0802.tif")
    
        # ratio_BRx=ratio_image(1,f"{path_raw}{region_name}/{year}/{datex}_r_TOA_06.tif",
        #       f"{path_raw}{region_name}/{year}/{datex}_r_TOA_08.tif",
        #       f"{path_raw}{region_name}/{year}/{datex}_r_TOA_0806.tif")
    
        # temp=RGBx(f"{path_raw}{region_name}/{year}/{datex}_r_TOA_08.tif",
        #   f"{path_raw}{region_name}/{year}/{datex}_r_TOA_06.tif",
        #   f"{path_raw}{region_name}/{year}/{datex}_r_TOA_02.tif",
        #   f"{path_raw}{region_name}/{year}/{datex}_r_TOA_RGB.tif")
        # if show_plots:
        #     plt.imshow(temp)
        #     plt.axis("Off")

    NDXI=read_S3(f"{path_raw}{region_name}/{year}/{datex}_r_TOA_NDXI_0608.tif")
    b02=read_S3(f"{path_raw}{region_name}/{year}/{datex}_r_TOA_01.tif")
    b06=read_S3(f"{path_raw}{region_name}/{year}/{datex}_r_TOA_06.tif")
    b17=read_S3(f"{path_raw}{region_name}/{year}/{datex}_r_TOA_17.tif")
    b21=read_S3(f"{path_raw}{region_name}/{year}/{datex}_r_TOA_21.tif")
    # labels=read_S3(f"{path_raw}/{datex}_labels_5bands_02_04_06_08_21.tif")
    labels=read_S3(f"/Users/jason/0_dat/S3/class_and_proba/{datex.replace('-','_')}_SICE_surface_classes.tif")
    np.shape(labels)
    prob=read_S3(f"/Users/jason/0_dat/S3/class_and_proba/{datex.replace('-','_')}_SICE_probability.tif")
    np.shape(b21)
    path_S3='/Users/jason/0_dat/S3/opendap/Greenland/'
    BBA=read_S3(f"{path_S3}{year}/{datex}_albedo_bb_planar_sw.tif")

    a=1.003 ; b=0.058

    BBA_emp=a*((b02+b06+b17+b21)/4)+b
    # lat=read_S3(f"/Users/jason/Dropbox/S3/ancil/lat.tif")
    # np.shape(lat)
    
   #%%
    red_snow_class_id=2
    import matplotlib as mpl


    #  plot 
    ly='x'
    
    fs=12 # fontsize
    
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))

    regions=['SW','CW','Sukkertoppen','QAS']
    regions=['CW','Sukkertoppen','QAS']
    regions=['QAS']
    regions=['Sukkertoppen']
    regions=['CW']
    regions=['SE']
    regions=['NW']
    
    regions=['CNW']
    confidence_thresh=0.99
    # confidence_thresh=0.6
    
    for region in regions:
        nj = 5424 ; ni = 2959  # all greenland raster dimensions
        if region=='SW':
            i0=500 ; i1=1000
            j0=3000 ; j1=nj-1

        if region=='SE':
            i0=1700 ; i1=ni-1
            j0=3000 ; j1=3800

        if region=='NW':
            i0=100 ; i1=500
            j0=500 ; j1=1200

        if region=='CNW':
            i0=600 ; i1=1000
            j0=2400 ; j1=2700
            
        if region=='CW':
            i0=700 ; i1=1200
            j0=3000 ; j1=3500

        if region=='Sukkertoppen':
            i0=500 ; i1=800
            j0=3800 ; j1=4000
    
        if region=='QAS':
            i0=950 ; i1=1150
            j0=5000 ; j1=nj-300
    
        labels_temp=labels.copy()[j0:j1,i0:i1]
    
        NDXI_temp=NDXI[j0:j1,i0:i1]
        
        b21_temp=b21[j0:j1,i0:i1]
        prob_temp=prob.copy()[j0:j1,i0:i1]
    
        labels_temp[prob_temp<confidence_thresh]=np.nan
        # NDXI_temp[labels_temp==3]=1
    
        BBA_temp=BBA.copy()[j0:j1,i0:i1]; BBA_temp[BBA_temp>0.99]=np.nan
        BBA_emp_temp=BBA_emp.copy()[j0:j1,i0:i1]; BBA_emp_temp[BBA_temp>0.99]=np.nan
        
        # BBA_temp[labels_temp!=red_snow_class_id]=np.nan
    
        #
        # im=plt.imshow(BBA_emp_temp)
        # plt.imshow(prob_temp)
        
        do_classes=1
        if do_classes:
            class_names=['dark_ice','bright_ice','red_snow','lakes','flooded_snow','melted_snow','dry_snow']
            N=len(class_names)
          
            cmaplist=np.zeros((N,4))
            cmaplist[0] = (200/255,100/255,200/255, 1.0)  # dark bare ice
            co=150
            cmaplist[1] = (co/255,co/255,co/255, 1.0)  # bright bare ice
            cmaplist[2] = (1,0,0, 1)  # red snow
            cmaplist[3] = (1,1,0, 1.0)  # lakes
            cmaplist[4] = (.5,.5,1, 1.0)  # flooded_snow
            cmaplist[5] = (1,.5,.5, 1.0)  # melted_snow
            cmaplist[6] = (0,0,0, 1.0)  # dry_snow
            
            # print(np.shape(cmaplist))
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, N)
            
            bounds = np.linspace(0, 6, 7)
            norm = mpl.colors.BoundaryNorm(bounds, N)
        
            im=ax.imshow(labels_temp,cmap=cmap,vmin=0,vmax=6)#,norm=norm)
            # im=plt.imshow(labels_temp)
        else:
            # im=plt.imshow(BBA_temp)
            im=plt.imshow(BBA_emp_temp)
        max_BBA=np.nanmax(BBA_temp)
        max_BBA_emp_temp=np.nanmax(BBA_emp_temp)
        BBA_offset=max_BBA_emp_temp-max_BBA
        
        # ----------- annotation
        xx0=1.01 ; yy0=1
        mult=0.95 ; co=0.
        # props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
        ax.text(xx0, yy0, f"{datex}\n{region}\nconfidence>{confidence_thresh}",
                fontsize=fs*mult,color=[co,co,co],rotation=0,
                transform=ax.transAxes,zorder=20,va='top',ha='left') # ,bbox=props
    
        # plt.imshow(NDXI_temp)
        # plt.imshow(b21_temp)
        plt.axis("Off")
        cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])
        # cbar=fig.colorbar(pcm, ax=ax, cax=cax)
        cbar = plt.colorbar(im,ax=ax, cax=cax)

        # cbar = plt.colorbar(im,fraction=0.029, pad=0.04)
        if do_classes:
            cbar.ax.set_yticklabels(class_names)
    
    #%%
    from numpy.polynomial.polynomial import polyfit
    from scipy import stats
    from scipy.stats import gaussian_kde

    
    regions=['dark','light']
    for rr,region in enumerate(regions):
        if rr==1:
            x=-NDXI_temp[((labels_temp==red_snow_class_id)&(prob_temp>confidence_thresh))]
            x[x<0]=np.nan
            # y=BBA_temp[labels_temp==red_snow_class_id]+BBA_offset
            y=BBA_temp[labels_temp==red_snow_class_id]
            # y=BBA_emp_temp[((labels_temp==red_snow_class_id)&(prob_temp>confidence_thresh))]
            

            # 2019-08-02
            # y[y>0.45]-=0.17
            # y[y<0.46]+=0.17
            # if rr==0:
            #     y[y>0.45]=np.nan
            # if rr==1:
            #     y[y<0.46]=np.nan
            

            # 2021-08-23
            # y[y<0.55]+=0.17
            # y[y<0.46]+=0.17
            # if rr==0:
            #     y[y>0.45]=np.nan
            # if rr==1:
            #     y[y<0.46]=np.nan

            #2023-07-08
            # y[y<0.36]=np.nan
            #2023-07-08
            y[y<0.55]=np.nan
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

            do_fit=1
            
            if do_fit:
                xx=[np.min(x),np.max(x)]
                # xx=[0,0.05]
                xx=np.array(xx)
                plt.plot(xx, b + m * xx, '--',c='k',linewidth=2)
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
        
        #%% imshow rasters
        
        # import geopandas as gpd
        # import fiona
        # import rasterio.mask

        # with fiona.open("/Users/jason/Dropbox/S3/SICE_classes2/ROIs/QAS.shp", "r") as shapefile:
        #     shapes = [feature["geometry"] for feature in shapefile]
            # print(shapefile["geometry"] )
        # points = gpd.read_file('/Users/jason/Dropbox/S3/SICE_classes2/ROIs/QAS.shp')
        
        
        # from osgeo import ogr, osr
        # import os
        
        # in_epsg = 4326
        # out_epsg = 3413
        # in_shp = '/Users/jason/Dropbox/S3/SICE_classes2/ROIs/QAS.shp'
        # out_shp = '/Users/jason/Dropbox/S3/SICE_classes2/ROIs/QAS_EPSG3413.shp'
        
        # driver = ogr.GetDriverByName('ESRI Shapefile')
        
        # # input SpatialReference
        # inSpatialRef = osr.SpatialReference()
        # inSpatialRef.ImportFromEPSG(in_epsg)
        
        # # output SpatialReference
        # outSpatialRef = osr.SpatialReference()
        # outSpatialRef.ImportFromEPSG(out_epsg)
        
        # # create the CoordinateTransformation
        # coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
        
        # # get the input layer
        # inDataSet = driver.Open(in_shp)
        # inLayer = inDataSet.GetLayer()
        
        # # create the output layer
        # if os.path.exists(out_shp):
        #     driver.DeleteDataSource(out_shp)
        # outDataSet = driver.CreateDataSource(out_shp)
        # outLayer = outDataSet.CreateLayer("reproject", geom_type=ogr.wkbMultiPolygon)
        
        # # add fields
        # inLayerDefn = inLayer.GetLayerDefn()
        # for i in range(0, inLayerDefn.GetFieldCount()):
        #     fieldDefn = inLayerDefn.GetFieldDefn(i)
        #     outLayer.CreateField(fieldDefn)
        
        # # get the output layer's feature definition
        # outLayerDefn = outLayer.GetLayerDefn()
        
        # # loop through the input features
        # inFeature = inLayer.GetNextFeature()
        # while inFeature:
        #     # get the input geometry
        #     geom = inFeature.GetGeometryRef()
        #     # reproject the geometry
        #     geom.Transform(coordTrans)
        #     # create a new feature
        #     outFeature = ogr.Feature(outLayerDefn)
        #     # set the geometry and attribute
        #     outFeature.SetGeometry(geom)
        #     for i in range(0, outLayerDefn.GetFieldCount()):
        #         outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        #     # add the feature to the shapefile
        #     outLayer.CreateFeature(outFeature)
        #     # dereference the features and get the next input feature
        #     outFeature = None
        #     inFeature = inLayer.GetNextFeature()
        
        # # Save and close the shapefiles
        # inDataSet = None
        # outDataSet = None
    #%%
        # with rasterio.open(f"{path_raw}{region_name}/{year}/{datex}_r_TOA_NDXI_0608.tif") as src:
        #     out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        #     out_meta = src.meta
