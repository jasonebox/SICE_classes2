# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:47:18 2023

@author: rabni
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import os
import numpy as np
import geopandas as gpd
from sklearn import svm
import xarray as xr
import rasterio as rio
from rasterio.transform import Affine
from pyproj import CRS as CRSproj
from pyproj import Transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import glob
import json
import time
import datetime
import geopandas
from matplotlib import path
import matplotlib.pyplot as plt
import logging
import colorsys
import traceback
from matplotlib.colors import ListedColormap
import random
import pickle
import sys
from EE_get import EarthEngine_S2

def remove_tif(path):
    if path.endswith(".tif") and os.path.isfile(path):
        os.remove(path)
        
    
def compute_weighted_mean(w,d):
    return sum(w * d) / sum(w)

def tukey_w(w,d,sigma):
    # Tukey biweights, by variance for each band to allow higher prediction skill
    break_p = 4.685
    ml_est = sum(w * d) / sum(w)
    eps = (d - ml_est) / sigma 
    
    w[abs(eps) <= break_p] = (1-(eps[abs(eps) <= break_p]/break_p)**2)**2
    w[abs(eps) > break_p] = 0
    
    return w 

def opentiff(filename):

    "Input: Filename of GeoTIFF File "
    "Output: xgrid,ygrid, data paramater of Tiff, the data projection"
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    da = rio.open(filename)
    #proj = CRSproj(da.crs)

    z = np.array(da.read(1),dtype=np.float32)
    #nx,ny = da.width,da.height
    #x,y = np.meshgrid(np.arange(nx,dtype=np.float32), np.arange(ny,dtype=np.float32)) * da.transform

    da.close()

    return z

def huber_w(w,d,sigma):
    # Huber weights, by variance for each band to allow higher prediction skill

    break_p = 1.345  
    ml_est = sum(w * d) / sum(w)
    eps = (d - ml_est) / sigma 
    
    w[eps > break_p] = break_p/eps[eps > break_p]
    w[abs(eps) <= break_p] = 1
    w[eps < -break_p] = -break_p/eps[eps < -break_p]
    
    return w 

def color_hex(n_days): 
    c_list = ['#FF0000','#00FF00','#0000FF','#FFFF00','#00FFFF','#800080','#FFA500','#008000','#A52A2A']    
    return c_list[:n_days]

def predefined_colors(class_names):
    N_classes=len(class_names)
    color_list=np.zeros((N_classes,4))
    color_list[0] = (100/255,100/255,100/255, 1.0)  # dark bare ice
    co=150
    color_list[1] = (co/255,co/255,co/255, 1.0)  # bright bare ice
    color_list[2] = (130/255,70/255,179/255, 1.0)  # purple ice
    color_list[3] = (1,0,0, 1)  # red snow
    color_list[4] = (0.8,0.8,0.3, 1.0)  # lakes
    color_list[5] = (.5,.5,1, 1.0)  # flooded_snow
    color_list[6] = (239/255,188/255,255/255, 1)  # melted_snow
    color_list[7] = (0,0,0, 1.0)  # dry_snow'
    return color_list

def generate_diverging_colors_hex(num_colors, center_color='#808080'):
    colors = []
    for i in range(num_colors):
        if i < num_colors // 2:
            hue = random.uniform(0.7, 1.0)  # Warm colors
        else:
            hue = random.uniform(0.7, 0.85)  # Adjusted cool colors range
        saturation = random.uniform(0.5, 1.0)
        value = random.uniform(0.5, 1.0)
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append('#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    colors.insert(num_colors // 2, center_color)
    return colors

def date_format(date_string):
       try:
           datetime.date.fromisoformat(date_string)
       except ValueError:
           return 'err'
       return "OK"
   
def shp_features(shp_f):
    #print(shp_f)
    label_gdf = gpd.read_file(shp_f)
    
    return json.loads(label_gdf.exterior.to_json())['features']

def shp_epsg(shp_f):
    return int(gpd.read_file(shp_f).crs.to_epsg())
    
def freedman_bins(df): 
    quartiles = df.quantile([0.25, 0.75])
    iqr = quartiles.loc[0.75] - quartiles.loc[0.25]
    n = len(df)
    h = 2 * iqr * n**(-1/3)
    bins = (df.max() - df.min())/h 
    if np.isnan(np.array(bins)) or np.isinf(np.array(bins)):
        bins = 2 
    return int(np.ceil(bins))

def merge(list1, list2):
     
    merged_list = [(p1, p2) for idx1, p1 in enumerate(list1) 
    for idx2, p2 in enumerate(list2) if idx1 == idx2]
    return merged_list
    
class ClassifierSICE():
    
    """ Surface type classifier for SICE, 
    using Sentinel-3 Top of the Atmosphere reflectances (r_TOA) """
    
    def __init__(self,bands=False,classes=False):
        
            self.src_folder = os.getcwd()
            self.base_folder = os.path.abspath('..')
            self.pixel_search = False
            WGSProj = CRSproj.from_string("+init=EPSG:4326")
            PolarProj = CRSproj.from_string("+init=EPSG:3413")
            self.transformer = Transformer.from_proj(WGSProj, PolarProj)
            self.transformer_inv = Transformer.from_proj(PolarProj, WGSProj)
            if not bands:    
                self.training_bands = ["r_TOA_02" ,"r_TOA_04" ,"r_TOA_06", "r_TOA_08", "r_TOA_21"]
            else:
                self.training_bands = bands
                
            if not classes:
                # self.classes = ['dark_ice','bright_ice','red_snow','lakes','flooded_snow','melted_snow','dry_snow']
                # self.colours  = ['#005AFF', '#5974AF', '#02D26E74', '#800080', '#03EDFE', '#04A0E4F5', '#05E9FEFF']
                self.classes = ['dark_ice','bright_ice','purple_ice','red_snow','lakes','flooded_snow','melted_snow','dry_snow']
                self.colours  = ['#005AFF', '#5974AF', '8b05f2','#02D26E74', '#800080', '#03EDFE', '#04A0E4F5', '#05E9FEFF']
                # self.classes = ['dark_ice','bright_ice','purple_ice','red_snow','dust','lakes','flooded_snow','melted_snow','dry_snow']
                # self.colours  = ['#005AFF', '#5974AF', '8b05f2','#02D26E74','#7F2B0A' ,'#800080', '#03EDFE', '#04A0E4F5', '#05E9FEFF']
                
            else:
                self.classes
                
            #self.training_bands = ["r_TOA_02","r_TOA_03" ,"r_TOA_04","r_TOA_05" ,"r_TOA_06", "r_TOA_08", "r_TOA_21"]
            # self.training_bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_08", "r_TOA_21",'sza'] sza idea?
            
            logpath = self.base_folder + os.sep + 'logs'   
            
            if not os.path.exists(logpath):
                os.makedirs(logpath)
                    
            logging.basicConfig(
                    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler(logpath + os.sep + f'sicemachine_{time.strftime("%Y_%m_%d",time.localtime())}.log'),
                        logging.StreamHandler()
                    ])

    
    def dim_redux(self,data,method):
        
        
        print('Dim reduction by SVD')
        train_data,train_label,test_data,test_label = self._train_test_format(data)
        
        if method == 'SVD':
            k = len(train_data[0,:])
        
            U, Sigma, VT = randomized_svd(train_data, 
                                          n_components=k,
                                          n_iter=5,
                                          random_state=None)
            X_approx = U[:, :k] @ np.diag(Sigma[:k]) @ VT[:k, :]
            approx_error = np.linalg.norm(train_data - X_approx) / np.linalg.norm(train_data)
            Vk = VT[:k, :]
            
            #svd = TruncatedSVD(n_components=len(train_data[0,:]))
            #x_new = svd.fit_transform(train_data)
            
            feature_importance = np.abs(Vk).sum(axis=0)
        
            sorted_idx = np.argsort(feature_importance)[::-1]
            
            top_features = [self.training_bands[i] for i in sorted_idx[:]]
        elif method == 'cov': 
            
            top_features = {}
            
            for i,cl in enumerate(self.classes):
                
                class_data = train_data[train_label==i] 
                cov_m = np.cov(class_data.T)
                sum_cov = [np.sum(ii) for ii in cov_m]
                sorted_idx = np.argsort(sum_cov)[::-1]
                top = [self.training_bands[i] for i in sorted_idx]
                top_features[cl] = {'top_bands' : top}
                
                
        return top_features 
                
    def get_training_data(self,d_t=None,polar=None,local=False,bio_track=False):
        
        '''Imports training from thredds server using OPeNDAP.
        The training dates,area and features are defined by the shapefiles in the /labels folder
        
        Parameters
        ----------
        self :
            
        polar:
            
          
        Returns
        -------
        dict
            dictionarty of training data
        '''
        
        shp_files = glob.glob(self.base_folder + os.sep + 'ROIs' + os.sep + '**' + os.sep + '**.shp', recursive=True)
        training_dates = np.unique([d.split(os.sep)[-2].replace('-','_') for d in shp_files])
       
        if d_t: 
            d_t = [d.replace('-','_') for d in d_t]
            training_dates =  [d for d in training_dates if d in d_t]
        
        if len(training_dates) == 0: 
            print(f"{d_t} does not exist")
            return
        
        dataset_ids = ['sice_500_' + d + '.nc' for d in training_dates]
        regions = ([d.split(os.sep)[-4] for d in shp_files])
        features = np.unique([d.split(os.sep)[-1][:-4] for d in shp_files])
        features =  np.unique([f for f in features if len(f.split('_'))==2]) # checking if thera are more than one shp file per class
        
        
        #ds_ref = xr.open_dataset(f'https://thredds.geus.dk/thredds/dodsC/SICE_Greenland_500m/{ref_DATASET}')
        training_data = {}
                
        print(f"Training Dates {training_dates}")
        
        for d,ref,re in zip(training_dates,dataset_ids,regions):     
            print(f"Getting Training Data for {d}")
            training_data[d] = {}
            
            print(f'region: {re}')
            print(f'dataset: {ref}')
            
            if not local: 
                ds_id = f'https://thredds.geus.dk/thredds/dodsC/SICE_500m/{re}/{ref}'
            else: 
                ds_id = self.base_folder + os.sep + 'training_data' + os.sep + re + os.sep + ref
            
            ds = xr.open_dataset(ds_id)
            shp_files_date = [s for s in shp_files if d in s.replace('-','_')]
            
            for f in self.classes:
                
                shp = [s for s in shp_files_date if f.split('_')[0] in s]
                label_shps = [shp_features(s) for s in shp]
                crs_shps = [shp_epsg(s) for s in shp]
                label_shps = [item for sublist in label_shps for item in sublist]    
                
                
                #x = np.array(ds[self.training_bands[0]].x)
                #y = np.array(ds[self.training_bands[0]].y)
                xcoor,ycoor = tuple(ds[self.training_bands[0]].coords.keys())
                xgrid,ygrid =  np.meshgrid(ds[xcoor],ds[ycoor])
                mask = (np.ones_like(xgrid) * False).astype(bool)
                
                
                for ii,ls in enumerate(label_shps):
                    if ls['geometry'] is not None:
                        x_poly, y_poly = map(list, zip(*ls['geometry']['coordinates']))
                        
                        if 4326 in crs_shps:
                            x_poly,y_poly = self.transformer.transform(np.array(x_poly),np.array(y_poly))
                                                                       
                        p = path.Path(np.column_stack((x_poly,y_poly)))
                        idx_poly = p.contains_points(np.column_stack((xgrid.ravel(),ygrid.ravel())))
                        mask.ravel()[idx_poly] = True
                        
                        if bio_track and (f == 'red_snow') and (d == '2019_08_02'):
                            red_snow_x = xgrid[mask]
                            red_snow_y = ygrid[mask]
                            self._S2_bio_track(red_snow_x, red_snow_y, d.replace('_','-'),train=True,poly_n=ii)
                        
                training_data[d][f] = {k:np.array(ds[k])[mask] for k in self.training_bands}
                #training_data[d][f]['sza'] = {np.cos(np.radians(np.array(ds['sza'])[mask]))}
                #training_data[d][f] = {k:np.array(ds[k].where(mask))[mask] for k in self.training_bands}
                
                ds.close()
           
        return training_data
    
    def plot_training_data(self,training_data=None,output=False):
        
        if not training_data:
            training_data = self.get_training_data()
        
        t_days = list(training_data.keys())
        features = list(training_data[t_days[0]].keys())
        if output:
            hist_dict = {k:{f:{c:{} for c in self.training_bands} for f in self.classes} for k in t_days}
            
        alpha_value = 0.4
        #center_color = (0.5, 0.5, 0.5)  # Adjust the center color as needed
        color_multi = generate_diverging_colors_hex(len(t_days))
        color_multi = color_hex(len(t_days))
        pdf_all_no_w = {k:[] for k in self.training_bands}
        pdf_all_t_w = {k:[] for k in self.training_bands}
       
        for f_int,f in enumerate(features):
            data_all = []
            for i,d in enumerate(t_days):
                data = np.array([training_data[d][f][b] for b in self.training_bands]).T
                data[data>1] = np.nan
                data[data<0.001] = np.nan
                dates = np.ones_like(data[:,0]) * i
                data_w_dates = np.column_stack((data, dates))
                data_all.append(data_w_dates)
            
            data_all = np.vstack([arr for arr in data_all])
            df_col = self.training_bands + ['date']
            df_data = pd.DataFrame(data_all,columns=[df_col])
            
            column_names = [d[0] for d in df_data.columns]
            
            num_rows = -(-len(column_names) // 2)
            fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(40, 40), gridspec_kw={'hspace': 0.2})
            axes = axes.flatten()
                
            
            for i,col in enumerate(column_names):
                if 'date' not in col:
                    ax = axes[i]
                   
                    x = np.array(df_data[col]).ravel()
                    mu = np.nanmean(x)
                    sigma = np.nanstd(x)
                    std_mask = (abs(mu-x)<100*sigma)
                    x = np.sort(x[std_mask])
                    x = x[~np.isnan(x)]
                    
                    mu = np.nanmean(x)
                    sigma = np.nanstd(x)
                    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                         np.exp(-0.5 * ((x - mu) / sigma)**2))
                    
                    
                    col_class = np.ones_like(y) * f_int
                    pdf_stack = np.array([x,y,col_class]).T
                    pdf_all_no_w[col].append(pdf_stack)
                    
                    w = np.ones_like(x)
                    no_i = np.arange(50)
                    for ite in no_i:
                        w = tukey_w(w, x, sigma)
                        
                    mu = compute_weighted_mean(w, x)
                    
                    x_weigted = x[w>0]
                    sigma = np.nanstd(x_weigted)
                    
                    y_weighted = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                         np.exp(-0.5 * (1 / sigma * (x_weigted - mu))**2)) 
                    
                    #y_weighted = y_weighted / np.nanmax(y_weighted)
                    
                    #x_range = np.linspace(np.nanmin(x),np.nanmax(x),num=20)
                    
                    
                    col_class = np.ones_like(y_weighted) * f_int
                    pdf_stack = np.array([x_weigted,y_weighted,col_class]).T
                    pdf_all_t_w[col].append(pdf_stack)
                    
                    ax.plot(x,y, color ='red',linewidth=6,\
                            label='Combined Gaussian pdf ',zorder=1)
                    ax.plot(x,y, color ='black',linewidth=7,\
                            zorder=0)
                        
                        
                    no = 0
                    bins_no = 0
                    for date_id in np.unique(df_data['date']):
                      
                        mask = df_data['date']==date_id
                        date_df = df_data[col][mask.squeeze()]
                        date_data_std = np.nanstd(date_df)
                        no_points = len(date_df)
                        
                        date_name = t_days[int(date_id)]
                        

                        #print(f'{no_points} of {f} on {date_name}')
                        
                        #print(f'Band {col} sigma of {f} at {date_name}: {date_data_std}')
                        
                        bins = freedman_bins(date_df)
                        
                        date_np = np.array(date_df)
                        date_np = date_np[~np.isnan(date_np)]
                        n, bins_out = np.histogram(date_np, bins=bins)
                        ax.hist(date_df, bins=bins, alpha=1, density=True,zorder=-1,\
                                          edgecolor='black', linewidth=1.2,histtype='step')
                        ax.hist(date_df, bins=bins, alpha=alpha_value, density=True,\
                                label=f'{date_name}', color=color_multi[int(date_id)],zorder=-2)
                        if output:
                            
                            hist_dict[date_name][f][col] = {'bins' : bins_out, 'count' : n}
                            
                    ax.set_title(f'Band: {col}',fontsize=40)
                    ax.set_ylabel('Density Count',fontsize=40)
                    ax.set_xlabel('Reflectance',fontsize=40)
                    
                    

                    ax.tick_params(labelsize=24)
                    ax.legend(fontsize=24)
                    
                    
                            
                    
                    #### Add Combined Dist. ####
                    
                  
            #if len(column_names) % 2 == 1:
            #    fig.delaxes(axes[-1])  
            fig.delaxes(axes[-1])
            #plt.suptitle(f'Training Data Band Distributions of Class {f}', fontsize=40)  # Add a single title
            plt.subplots_adjust(wspace=0.1, hspace=0.1,top=0.85)
            #plt.tight_layout()  # Adjust layout to make space for the title
            if output:
                #plt.savefig(self.base_folder + os.sep + 'figs' + os.sep + f'histogram_{f}.png')
                plt.savefig(self.base_folder + os.sep + 'figs' + os.sep + f'histogram_{f}.png',dpi=400,bbox_inches='tight')
                #plt.savefig(self.base_folder + os.sep + 'figs' + os.sep + f'histogram_{f}.png',bbox='tight')
                #plt.close()
               
                
                # fig_out,ax_out = plt.subplots(figsize=(10,7))
                # ax_out.hist(date_df, bins=bins, alpha=1, density=True,zorder=-1,\
                #                   edgecolor='black', linewidth=1.2,histtype='step')
                # ax_out.hist(date_df, bins=bins, alpha=alpha_value, density=True,\
                #         label=f'{date_name}', color=color_multi[int(date_id)],zorder=-2)
                # ax_out.set_title(f'Class: {f} Band: {col}',fontsize=20)
                # ax_out.set_ylabel('Density Count',fontsize=20)
                # ax_out.set_xlabel('Reflectance',fontsize=20)
                # ax_out.tick_params(labelsize=16)
                # ax_out.legend()
            plt.show()
            
        
        spectrum = pd.read_csv('S3_spectrum.csv')
        mean_spectrum = {f:{'mean':[],'std':[],'wl':[]} for f in self.classes}
    
        
        num_rows = -(-len(self.training_bands) // 2)
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(22, 12),dpi=600, gridspec_kw={'hspace': 0.5})
        axes = axes.flatten()
        color_multi = predefined_colors(self.classes)
       
        
        for i,toa in enumerate(self.training_bands):
            data = pdf_all_t_w[toa]
            
            ax = axes[i]
            for j,cl in enumerate(data):
                
                x = cl[:,0]
                y = cl[:,1]
                class_int = int(np.unique(cl[:,2]))
                class_name = features[class_int]
                b_spec = toa[-2:]
                mean_spectrum[class_name]['mean'].append(np.nanmean(x))
                mean_spectrum[class_name]['std'].append(np.nanstd(x))
                mean_spectrum[class_name]['wl'].append(spectrum[b_spec])
                
                ax.plot(x,y, color = color_multi[class_int],linewidth=6,\
                        label=f'{class_name}',zorder=1)
                ax.plot(x,y, color ='black',linewidth=7,\
                        zorder=0)
            
            ax.set_title(f'Band: {toa}',fontsize=20)
            ax.set_ylabel('Density',fontsize=20)
            ax.set_xlabel('Reflectance',fontsize=20)
            ax.tick_params(labelsize=16)
            ax.legend()
            
        
        if len(column_names) % 2 == 1:
            fig.delaxes(axes[-1])  
        #fig.delaxes(axes[-1])      
        plt.suptitle('Gaussian PDF of all classes - With Tukey BiWeights', fontsize=30)  # Add a single title
        #plt.tight_layout()  # Adjust layout to make space for the title
        plt.show()
        
        
        
        # Initialize a figure
        start = 1
        # Initialize a figure
        fig, ax = plt.subplots(figsize=(24, 12),dpi=600)
 
        # Plot the line
        # the text bounding box
        bbox = {'fc': '0.8', 'pad': 0}
        for i,c in enumerate(list(mean_spectrum.keys())):
 
 
            x = np.array(mean_spectrum[c]['wl']).ravel()
            y = np.array(mean_spectrum[c]['mean'])
            z = np.array(mean_spectrum[c]['std'])
            
            
            ax.scatter(x, y,label=f'{c}',color = color_multi[i],s=29)
            ax.scatter(x, y,color = 'black',s=45,zorder=0)
            # Create shaded area around the line
            #ax.fill_between(x, y - z, y + z, alpha=0.3, color=color_multi[i])
            ax.plot(x, y,color = color_multi[i],zorder=0,linewidth = 1)
            ax.plot(x, y,color = 'black',zorder=-1,linewidth = 2)
            if start == 1 and c == 'red_snow':
                for ii, txt in enumerate(self.training_bands):
                    ax.annotate('B'+txt[-2:], (x[ii], -0.0),
                     rotation=90,zorder=5,size = 15)
                    ax.plot((x[ii],x[ii]), (0,1),color = 'black',alpha=0.2,
                            zorder=-5)
                start = 0
     
 
        # Customize the plot
        # graphics definitions
        band_type = self.training_bands[0][:-3]
        th=2 # line thickness
        formatx='{x:,.3f}' ; fs=18
        plt.rcParams["font.size"] = fs
        plt.rcParams['axes.facecolor'] = 'w'
        plt.rcParams['axes.edgecolor'] = 'k'
        plt.xticks(fontsize=30, rotation=90)
        plt.yticks(fontsize=30)
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.5
        plt.rcParams['grid.color'] = "#C6C6C6"
        plt.rcParams["legend.facecolor"] ='w'
        plt.rcParams["mathtext.default"]='regular'
        plt.rcParams['grid.linewidth'] = th/2
        plt.rcParams['axes.linewidth'] = 1
        ax.set_xlabel('wavelength, nm',fontsize=35)
        ax.set_ylabel(f'{band_type} reflectance',fontsize=35)
        #ax.set_title('Mean Reflectance on Training Data',fontsize=35)
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=35,markerscale=4)
        # Show the plot
        
        plt.savefig(self.base_folder + os.sep + 'figs' + os.sep + f'{band_type}_mean_reflectance_of_training_data.png',dpi=400,bbox_inches='tight')
        plt.show()
        
        # for i,c in enumerate(list(mean_spectrum.keys())):
            
        #     x = np.array(mean_spectrum[c]['wl']).ravel()
        #     y = np.array(mean_spectrum[c]['mean'])
        #     z = np.array(mean_spectrum[c]['std'])
            
        #     fig, ax = plt.subplots(figsize=(24, 18),dpi=600)
            
        #     start = 1
            
        #     min_y = 0
        #     max_y = 0
        #     for ii,cc in enumerate(list(mean_spectrum.keys())):
        #         if i != ii: 
                    
        #             yy = np.array(mean_spectrum[cc]['mean'])
        #             ratio = (y/yy)
        #             ax.scatter(x, ratio,label=f'{cc}',color = color_multi[ii],s=29)
        #             ax.scatter(x, ratio,color = 'black',s=45,zorder=0)
        #             # Create shaded area around the line
        #             #ax.fill_between(x, y - z, y + z, alpha=0.3, color=color_multi[i])
        #             ax.plot(x, ratio,color = color_multi[ii],zorder=0,linewidth = 1)
        #             ax.plot(x, ratio,color = 'black',zorder=-1,linewidth = 2)
                    
        #             if min_y > np.nanmin(ratio):
        #                 min_y = np.nanmin(ratio)
        #             if max_y < np.nanmax(ratio):
        #                 max_y = np.nanmax(ratio)    
                    
            
        #     for jj, txt in enumerate(self.training_bands):
        #         ax.annotate('B'+txt[-2:], (x[jj], -0.0),
        #          rotation=90,zorder=5,size = 15)
        #         ax.plot((x[jj],x[jj]), (0,max_y),color = 'black',alpha=0.2,
        #                 zorder=-5)
                
        #     th=2 # line thickness
        #     formatx='{x:,.3f}' ; fs=18
        #     plt.rcParams["font.size"] = fs
        #     plt.rcParams['axes.facecolor'] = 'w'
        #     plt.rcParams['axes.edgecolor'] = 'k'
        #     plt.xticks(fontsize=30, rotation=90)
        #     plt.yticks(fontsize=30)
        #     plt.rcParams['axes.grid'] = False
        #     plt.rcParams['axes.grid'] = True
        #     plt.rcParams['grid.alpha'] = 0.5
        #     plt.rcParams['grid.color'] = "#C6C6C6"
        #     plt.rcParams["legend.facecolor"] ='w'
        #     plt.rcParams["mathtext.default"]='regular'
        #     plt.rcParams['grid.linewidth'] = th/2
        #     plt.rcParams['axes.linewidth'] = 1
        #     ax.set_xlabel('wavelength, nm',fontsize=35)
        #     ax.set_ylabel('rBRR/TOA ratio',fontsize=35)
        #     ax.set_title(f'',fontsize=35)
        #     ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=35,markerscale=4)
        #     band_type = self.training_bands[0][:-3]
        #     plt.savefig(self.base_folder + os.sep + 'figs' + os.sep + f'{band_type}_{c}_ratio_to_all_classes.png',dpi=400,bbox_inches='tight')
        #     plt.show()
           
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(22, 18), gridspec_kw={'hspace': 0.5})
        axes = axes.flatten()
        color_multi = generate_diverging_colors_hex(len(features))
        color_multi = color_hex(len(features))
        for i,toa in enumerate(self.training_bands):
            data = pdf_all_no_w[toa]
            ax = axes[i]
            for j,cl in enumerate(data):
                
                
                
                x = cl[:,0]
                y = cl[:,1]
                class_int = int(np.unique(cl[:,2]))
                class_name = features[class_int]
                
            
                #
                ax.plot(x,y, color = color_multi[class_int],linewidth=6,\
                        label=f'{class_name}',zorder=1)
                ax.plot(x,y, color ='black',linewidth=7,\
                        zorder=0)
            
            ax.set_title(f'Band: {toa}',fontsize=20)
            ax.set_ylabel('Density',fontsize=20)
            ax.set_xlabel('Reflectance',fontsize=20)
            ax.tick_params(labelsize=16)
            ax.legend()
            
        if len(column_names) % 2 == 1:
            fig.delaxes(axes[-1])  
        fig.delaxes(axes[-1])      
        plt.suptitle('Gaussian PDF of all classes - No ML Estimation', fontsize=30)  # Add a single title
        #plt.tight_layout()  # Adjust layout to make space for the title
        plt.show()
        
        
        if output: 
            dicts = [hist_dict[d] for d in hist_dict]
            dfs = [pd.DataFrame.from_dict(d) for d in dicts]
            out = [df.to_csv(self.base_folder + os.sep + 'figs' + os.sep + f'hist_bins_{d}.csv') for df,d in zip(dfs,t_days)]        
    
        
        
        return 
    
    def _train_test_format(self,training_data,test_type=None,test_date=None,fold=None):
        
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        
        t_days = list(training_data.keys())
        features = list(training_data[t_days[0]].keys())
        
        for f_int,f in enumerate(features):
            data = [np.array([training_data[d][f][b] for b in self.training_bands]).T for d in t_days if d != test_date]
            
            data_stack = np.vstack([arr for arr in data])
            data_stack = np.array([dd for dd in data_stack[:] if not np.isnan(dd).any()])
            data_stack = np.array([dd for dd in data_stack[:] if len(dd[dd==0])==0])
            
            if test_type:
                if test_type == 'date':
                    data_test_stack = [np.array([training_data[test_date][f][b] for b in self.training_bands]).T]
                   
                    data_test_stack = np.vstack([arr for arr in data_test_stack])
                    data_test_stack = np.array([dd for dd in data_test_stack[:] if not np.isnan(dd).any()])
                    data_test_stack = np.array([dd for dd in data_test_stack[:] if len(dd[dd==0])==0])
                    
                elif test_type == 'ratio':
                    
                    seed_value = 42
                    np.random.seed(seed_value)
                    np.random.shuffle(data_stack)
                    
                    no_pixels = len(data_stack[:,0])
                    no_bands = len(self.training_bands)
                    fold_r_min = int(no_pixels/5) * (fold - 1)
                    fold_r_max = int(no_pixels/5) * (fold)
            
                    mask = np.zeros_like(data_stack, dtype=bool)
                    mask[fold_r_min:fold_r_max,:] = True
                    
                    
                    data_test_stack = np.ones_like(data_stack) * np.nan
                    
                    data_test_stack = data_stack[mask].reshape((fold_r_max-fold_r_min),no_bands)
                    data_stack = data_stack[~mask].reshape((no_pixels-(fold_r_max-fold_r_min)),no_bands)
                  
                test_data.append(data_test_stack)
                
                label = (np.ones_like(data_test_stack[:,0]) * f_int).reshape(-1,1)
                test_label.append(label)
            else:
                test_data.append(None)
                test_label.append(None)
                
                
            train_data.append(data_stack)
            label = (np.ones_like(data_stack[:,0]) * f_int).reshape(-1,1)
            train_label.append(label)
                    
        train_data = np.vstack([arr for arr in train_data])
        train_label = np.vstack([arr for arr in train_label]).ravel()
        
        test_data = np.vstack([arr for arr in test_data])
        test_label = np.vstack([arr for arr in test_label]).ravel()
        
        return train_data,train_label,test_data,test_label
                
    def train_svm(self,training_data=None,c=1,gamma='scale',weights=True,kernel='rbf',
                  prob=False,test_type=None,fold=None,test_date=None,export=None):
        
           
        if not training_data:
            training_data = self.get_training_data()
            
        t_days = list(training_data.keys())
        
        if test_type:
            if test_type == 'date':
                testing_date = test_date
                print(f'Test is Set to Date, Using {test_date} as Testing Date ')
            elif fold>5:
                print(f'Train/Test ratio is 80/20, you cannot use more than 5 folds since 5*20 = 100,,,, Test is set to None')
                test_type = None
            else:
                print(f'Test is Set to Ratio, Using 80/20 as Train/Test Ratio on Whole Dataset, Fold no {fold}')
                testing_date = None
        else:
            testing_date = None
        
        features = list(training_data[t_days[0]].keys())
        n_features = len(features)
        n_bands = len(self.training_bands)
            
        train_data = []
        train_label = []
        
        
        print('Formatting Training Data')
        train_data,train_label,test_data,test_label = self._train_test_format(training_data,\
                                                      test_type=test_type,test_date=test_date,fold=fold)

        if weights:
            print('Computing Weights')
            no_i = np.arange(50) #Number of iterations
            w_all = np.ones_like(train_data)
            
            for n in np.arange(n_features):
                for b in np.arange(n_bands):
                    w = w_all[:,b][train_label==n]
                    d = train_data[:,b][train_label==n]
                    sigma = np.std(d)
                    for i in no_i:
                        w = tukey_w(w,d,sigma)
                    w_all[:,b][train_label == n] = w 
                    
            w_samples = np.array([np.nanmean(w) for w in w_all])        
        else:
            w_samples = np.ones_like(train_label)
        
        print('Training Model....')
        model = svm.SVC(C = c,gamma=gamma, decision_function_shape="ovo",kernel=kernel,probability=bool(prob))
        model.fit(train_data, train_label,sample_weight=w_samples)
        print('Done')
        
        if test_type is None:
            data_split_svm = None
        else:
            data_split_svm = {}
            print('Splitting dataset')
            for i,f in enumerate(features): 
                data_split_svm[f] = {'train_data' : train_data[train_label==i],'train_label' : train_label[train_label==i],\
                                     'test_data' : test_data[test_label==i],'test_label' : test_label[test_label==i]}        
                    
            data_split_svm['meta'] = {'testing_date' : testing_date}        
        
        if export: 
            filename = self.base_folder + os.sep + 'model' + os.sep + 'model.sav'
            pickle.dump(model, open(filename, 'wb'))
        
        return model,data_split_svm

    def test_svm(self,model=None, data_split=None, export_error=None,mute=False):
        
        if data_split is None:
                model,data_split = self.train_svm()
        #enablePrint()
        #if mute:
            #blockPrint()
        
        print('Test SVM for each Class \n')
        
        meta = data_split['meta']['testing_date']
        
        if meta is not None: 
            print(f"The model is being tested on an independent date: {meta}")
        
        #classes = list(data_split.keys())
        
        acc_dict = {}
        acc_dict['meta'] = {'testing_date' : meta}
        
        
        for cl in self.classes:
            
            #print(f'Test Results for Class {cl}:')
            data_test = data_split[cl]['test_data']
            label_test = data_split[cl]['test_label']
            data_train = data_split[cl]['train_data']
            
            #Predicting on Test Data:
            labels_pred = model.predict(data_test)
            cm = confusion_matrix(labels_pred, label_test)
            ac = np.round(accuracy_score(labels_pred,label_test),3)
            
            acc_dict[cl] = {'acc' : ac}
            
            #print(f"Plotting Band Distribution in class {cl}")
            alpha_value = 0.35
            num_bins = 10
            den = False
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
            color_multi = ['#FFA500', '#FF8C00', '#FFD700', '#FF6347', '#FFA07A', '#FF4500', '#FF1493']
        
            #if len([l for l in labels_pred if l not in label_test]) > 0:    
            #print(f"Plotting Band Distribution of Predicted Label(s) in Class {cl}: \n")
            l_mask = np.array([True if l not in label_test else False for l in labels_pred])
            bad_labels = data_test[l_mask,:]
            bad_labels_cl = labels_pred[l_mask]
            good_labels = data_test[~l_mask,:]
            
            bad_labels_val = np.column_stack((bad_labels, bad_labels_cl))
            bad_labes_col = self.training_bands + ['label']
            bad_labels = pd.DataFrame(bad_labels_val,columns=[bad_labes_col])
            good_labels = pd.DataFrame(good_labels,columns=[self.training_bands])
            train_labels = pd.DataFrame(data_train,columns=[self.training_bands])
            
            # Get column names from the DataFrames
            column_names = good_labels.columns
            num_rows = -(-len(column_names) // 2) 
            
            # Creating overlapping histogram plots for all columns from both DataFrames
            fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(column_names):
                
                ax = axes[i]
                bins = freedman_bins(good_labels[col])
                
                good_labels[col].hist(ax=ax, bins=bins, alpha=alpha_value,\
                                      label='Correct Labelled Test Data', color=colors[0],\
                                      histtype='barstacked', density=den)
                    
                good_labels[col].hist(ax=ax, bins=bins, alpha=1, density=den,\
                                      edgecolor='black', linewidth=1.2,histtype='step')
                """
                bad_labels[col].hist(ax=ax, bins=num_bins, alpha=alpha_value,\
                                     label='Bad Labelled Test Data',  color=colors[1],\
                                     histtype='barstacked', density=True)
                """
                
                for class_id in np.unique(bad_labels['label']):
                    
                    if len(bad_labels)/len(good_labels) < 0.01: 
                        pass
                        #class_df = bad_labels[col]
                    else:
                        mask = bad_labels['label']==class_id
                        class_df = bad_labels[col][mask.squeeze()]
                        class_name = self.classes[int(class_id)]
                        bins = freedman_bins(class_df)
                     
                        ax.hist(class_df, bins=bins, alpha=alpha_value, density=den,\
                                label=f'Test Data Labelled Wrongly as {class_name}', color=color_multi[int(class_id)])
                        ax.hist(class_df, bins=bins, alpha=1, density=den,\
                                             edgecolor='black', linewidth=1.2,histtype='step')
                
                bins = freedman_bins(train_labels[col])
          
                train_labels[col].hist(ax=ax, bins=bins, alpha=alpha_value,\
                                     label='Traning Data', color=colors[2],\
                                     histtype='barstacked', density=den)
                train_labels[col].hist(ax=ax, bins=bins, alpha=1, density=den,\
                                     edgecolor='black', linewidth=1.2,histtype='step')
            
                #bad_labels[col].hist(ax=ax, bins=num_bins, alpha=1, density=True,\
                #                     edgecolor='black', linewidth=1.2,histtype='step')
               
                
                ax.set_title(f'Band: {col}',fontsize=20)
                ax.legend()
                
            if len(column_names) % 2 == 1:
                fig.delaxes(axes[-1])    
                
            plt.suptitle(f'Band Distributions of Predicted Class {cl}', fontsize=20)  # Add a single title
            plt.tight_layout()  # Adjust layout to make space for the title
            plt.show()
            
            if not mute:     
                print(f"Accuracy of Predicting {cl}: {ac}")
            #print(f"Confusion Matrix of {cl}: \n {cm} \n")
            
            for l in list(np.unique(labels_pred)):
                
                no_l_p = len(labels_pred[labels_pred==l])
                label_name_prd = self.classes[int(l)] 
                label_name_cor = cl
                if not mute: 
                    print(f'Model Classified {label_name_prd} {no_l_p} times, the Correct Class was {label_name_cor} \n')
                
        #enablePrint()
        
        #### TOTAL Confusion Matrix ####
        
        test_label_all = np.concatenate([data_split[d]['test_label'] for d in self.classes])
        test_data_all = np.vstack([data_split[d]['test_data'] for d in self.classes])
        
        labels_pred = model.predict(test_data_all)
        cm = confusion_matrix(labels_pred, test_label_all)
        con_dict = {}
        for i,cl in enumerate(self.classes):
            con_dict[cl] = {'ratio' : (len(data_split[cl]['test_data'])/(len(data_split[cl]['train_data'])\
                                     + len(data_split[cl]['test_data']))),\
                            'com' : ((sum(cm[i,:])-cm[i,i])/sum(cm[i,:])),\
                            'omm' : ((sum(cm[:,i])-cm[i,i])/sum(cm[:,i])),\
                            'confusion' : cm}
        
        return acc_dict,con_dict,cm
    
    def get_prediction_data(self,dates_to_predict):
        
        if dates_to_predict is None:
            logging.info("Please Specify a Date to Predict")
            return None
        
        if type(dates_to_predict) == str: 
            dates_to_predict = [dates_to_predict]
            
        
        for d in dates_to_predict:
            msg = date_format(d)
            if msg == 'err':
                logging.info(f"Incorrect date format for {d}, should be [YYYY-MM-DD] in a list!")
                return None
        
        dataset_ids = ['sice_500_' + d.replace('-','_') + '.nc' for d in dates_to_predict]
        prediction_data = {}
        for d,ref in zip(dates_to_predict,dataset_ids): 
            logging.info(f'Loading {d} ......')
            try:
                ds = xr.open_dataset(f'https://thredds.geus.dk/thredds/dodsC/SICE_500m/Greenland/{ref}')
                prediction_data[d] = {k:np.array(ds[k]) for k in self.training_bands}
                x = np.array(ds[self.training_bands[0]].x)
                y = np.array(ds[self.training_bands[0]].y)
                crs = CRSproj.from_string("+init=EPSG:3413")
                xgrid,ygrid = np.meshgrid(x,y)
                prediction_data[d]['meta'] = {'x' : xgrid, 'y' : ygrid,'crs' : crs}
                ds.close()
            except Exception as e: 
                logging.info(f'{d} does not exist on the thredds server')
                
                
        return prediction_data
        
    def predict_svm(self,dates_to_predict,model=None,training_predict=False,
                    prob=False,export='tif',bio_track=False):
        
        self.model = model
        self.prob = prob
        self.bio_track = bio_track
        
        if export not in ['tiff','tif','all','png']:
            logging.info('Please specify a correct export format, options = [tiff, tif, all, png]')
            return
        elif export == 'all':
            self.export = ['tiff','png']
        else: 
            self.export = [export]    
        
        
        if self.model is None:
                logging.info('Getting Training Data and Training Model......')
                self.model,data_split = self.train_svm()
        elif self.model == 'import':
             path = self.base_folder + os.sep + 'model' + os.sep + 'model.sav'
             self.model = pickle.load(open(path, 'rb'))
       
        logging.info('Loading Bands for Prediction Dates:')
        
        if training_predict:
            shp_files = glob.glob(self.base_folder + os.sep + 'ROIs' + os.sep + '**' + os.sep + '**.shp', recursive=True)
            dates_to_predict = np.unique([d.split(os.sep)[-2].replace('-','_') for d in shp_files])
        
        
        self.prediction_data = self.get_prediction_data(dates_to_predict)
        
        if self.prediction_data is None:
            return
       
        p_days = list(self.prediction_data.keys())
       
        if not os.path.exists(self.base_folder + os.sep + "output"):
            os.mkdir(self.base_folder + os.sep + "output")
        
        for d in p_days:
            self._predict_for_date(d)
           
    
    def _predict_for_date(self,date):

        logging.info(f'Predicting Classes for {date}.....')
        data = np.array([self.prediction_data[date][b] for b in self.training_bands])
        self.xgrid = self.prediction_data[date]['meta']['x']
        self.ygrid = self.prediction_data[date]['meta']['y']
        self.crs = self.prediction_data[date]['meta']['crs']
        
        #mask = ~np.isnan(data[0,:,:])
        mask = ~np.isnan(data).any(axis=0)
        data_masked = data[:,mask].T
        # nan_rows = np.isnan(data_masked).any(axis=1)
        # data_masked = data_masked[~nan_rows]
        
        labels_predict = self.model.predict(data_masked)
        labels_grid = np.ones_like(self.xgrid) * np.nan
        labels_grid[mask] = labels_predict
        
        if self.prob:
            labels_prob = self.model.predict_proba(data_masked)
            labels_prob = np.amax(labels_prob, axis=1, keepdims=True).ravel()
            prob_grid = np.ones_like(self.xgrid) * np.nan
            prob_grid[mask] = labels_prob
            
       
             
        logging.info(f'Done for {date}')
        date_out = date.replace('-','_')
        
        
        for exp in self.export:
            logging.info(f'Saving as {exp}....')
            out_folder = f'{self.base_folder}{os.sep}output{os.sep}{exp}'
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            self.f_name = f'{out_folder}{os.sep}{date_out}_SICE_surface_classes.{exp}'
            self._export(labels_grid)
            
            if self.prob:
                self.f_name = f'{out_folder}{os.sep}{date_out}_SICE_probability.{exp}'
                self._export(prob_grid)
        
        if self.bio_track:
             logging.info(f'Checking red snow predictions for algae, date: {date}')
             
             prob_mask = prob_grid > 0.96
             
             red_snow_x = self.xgrid[prob_mask][labels_grid[prob_mask]==self.classes.index('red_snow')]
             red_snow_y = self.ygrid[prob_mask][labels_grid[prob_mask]==self.classes.index('red_snow')]
             
             self._S2_bio_track(self,red_snow_x,red_snow_y,date)
             
        logging.info('Done')
        
    def _S2_bio_track(self,red_snow_x,red_snow_y,date,train=False,poly_n = False):
        
        spectrum = pd.read_csv('S2_spectrum.csv')
        if not self.pixel_search:
            logging.info(f'Tracking Bio Signal Using S2 on date: {date}')
        
       
        for i,(x,y) in enumerate(zip(red_snow_x,red_snow_y)):
           
            x_lon,y_lat = self.transformer_inv.transform(np.array([x,x+500]),np.array([y,y-500]))
            
            if ((x_lon[0] > -52) and (x_lon[0] < -50) and  (y_lat[0] > 65) and  (y_lat[0] < 67)): 
               
                self.pixel_search = False
                
                bounds = [[x_lon[0], y_lat[0]],
                         [x_lon[1], y_lat[0]],
                         [x_lon[0],y_lat[1]],
                         [x_lon[1], y_lat[1]]] 
                
                if train:
                    id_tile = str(poly_n).zfill(2) + '_' + str(i).zfill(2)
                else:
                    id_tile = str(i).zfill(2)
                
                logging.info(f'id tile: {id_tile}')
                
                s2_ids = EarthEngine_S2(bounds,date,id_tile)
                
                if not s2_ids: 
                    logging.info(f"no S2 data at tile: {id_tile}, skipping...")
                else:
                    
                    bands = [s.split('.')[1] for s in s2_ids]
                    
                    da = rio.open(s2_ids[0])
                    
                    nx,ny = da.width,da.height
                    x_mask,y_mask = np.meshgrid(np.arange(nx,dtype=np.float32), np.arange(ny,dtype=np.float32)) * da.transform
                    da.close()
                    
                    mask = ((x_mask > x) * (x_mask < (x+500)) * (y_mask < y) * (y_mask > (y-500)))
                   
                    nan_check = opentiff(s2_ids[0])[mask]
                    
                    if ~np.isnan(np.nanmean(nan_check)):
                        
                        
                        s2_dict = {b:(opentiff(s2)[mask]).ravel() for b,s2 in zip(bands,s2_ids)}
                        
                        list(map(remove_tif,s2_ids))
                        
                      
                        
                        x_spec = np.array([spectrum[b] for b in bands]).ravel()
                        y_spec = np.array([(s2_dict[b]) for b in bands])
                        #std_spec = np.array([(s2_dict[b]) for b in bands])
                        
                        B5_ind = bands.index('B5')
                        B4_ind = bands.index('B4') 
                        B3_ind = bands.index('B3') 
                        B2_ind = bands.index('B2') 
                        
                        
                        y_R_G = y_spec[B3_ind,:] - y_spec[B4_ind,:]
                        y_R_RR = y_spec[B4_ind,:] - y_spec[B5_ind,:]
                        
                        
                        R_G_msk = y_R_G < 0.04
                        B5_mask = y_R_RR < 0
                        
                        algae_msk = R_G_msk * B5_mask
                        green_algae = ~R_G_msk * B5_mask
                        
                        y_spec_filt = y_spec[:,algae_msk]
                        y_spec_snow = y_spec[:,~algae_msk]
                        y_spec_green = y_spec[:,green_algae]
                        
                        fig, ax = plt.subplots(figsize=(12, 12),dpi=200)
                        
                        bbox = {'fc': '0.8', 'pad': 0}
                        
                        x_data_red = y_spec_filt[B3_ind,:] - y_spec_filt[B4_ind,:]
                        y_data_red = y_spec_filt[B4_ind,:] - y_spec_filt[B5_ind,:]
                        
                        ax.scatter(x_data_red, y_data_red,label='Red Snow',color = 'red',s=45,marker='s')
                        ax.scatter(x_data_red, y_data_red,color = 'black',s=60,zorder=0)
                        
                        x_data_snow = y_spec_snow[B3_ind,:] - y_spec_snow[B4_ind,:]
                        y_data_snow = y_spec_snow[B4_ind,:] - y_spec_snow[B5_ind,:]
                        
                         
                        ax.scatter(x_data_snow, y_data_snow,label='Snow',color = 'blue',s=45)
                        ax.scatter(x_data_snow, y_data_snow,color = 'black',s=60,zorder=0)

                        x_data_green = y_spec_green[B3_ind,:] - y_spec_green[B4_ind,:]
                        y_data_green = y_spec_green[B4_ind,:] - y_spec_green[B5_ind,:]
                         
                        ax.scatter(x_data_green, y_data_green,label='Green Snow',color = 'green',s=45,marker='d')
                        ax.scatter(x_data_green, y_data_green,color = 'black',s=60,zorder=0)
                                                
                        # Data for the first line
                        x1 = [-0.05, 0.15]
                        y1 = [0, 0]
                        
                        # Data for the second line
                        x2 = [0.04, 0.04]
                        y2 = [0, -0.25]
                                                
                        # Plotting the lines
                        ax.plot(x1, y1)
                        ax.plot(x2, y2)
                        
                        th=2 # line thickness
                        formatx='{x:,.3f}' ; fs=18
                        plt.rcParams["font.size"] = fs
                        plt.rcParams['axes.facecolor'] = 'w'
                        plt.rcParams['axes.edgecolor'] = 'k'
                        plt.xticks(fontsize=30, rotation=90)
                        plt.yticks(fontsize=30)
                        plt.rcParams['axes.grid'] = False
                        plt.rcParams['axes.grid'] = True
                        plt.rcParams['grid.alpha'] = 0.5
                        plt.rcParams['grid.color'] = "#C6C6C6"
                        plt.rcParams["legend.facecolor"] ='w'
                        plt.rcParams["mathtext.default"]='regular'
                        plt.rcParams['grid.linewidth'] = th/2
                        plt.rcParams['axes.linewidth'] = 1
                        ax.set_xlabel(f'B3 - B4',fontsize=35)
                        ax.set_ylabel(f'B4 - B5',fontsize=35)
                        ax.set_title(f'Tile no: {id_tile}',fontsize=35)
                        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=35,markerscale=4)
                        
                        # Show the plot
                        out_f = self.base_folder + os.sep + 'figs' + os.sep + 'bio_tracker'
                        plt.savefig(out_f + os.sep + f'S2_band_class_{id_tile}.png',dpi=200,bbox_inches='tight')
                        #plt.show()
                        plt.close()
                        if y_spec.size > 0:
                            
                            # Initialize a figure
                            fig, ax = plt.subplots(figsize=(24, 12),dpi=200)
                     
                            # Plot the line
                            # the text bounding box
                            bbox = {'fc': '0.8', 'pad': 0}
                            
                            y_data = np.nanmean(y_spec,axis=1)
                            y_data_std = np.nanstd(y_spec,axis=1)
                            x_pos = np.linspace(0,100,len(x_spec))
                            
                            ax.scatter(x_pos, y_data,label='Red Snow',color = 'red',s=29)
                            ax.scatter(x_pos, y_data,color = 'black',s=45,zorder=0)
                            # Create shaded area around the line
                            ax.fill_between(x_pos, y_data - y_data_std, y_data + y_data_std, alpha=0.3, color='red')
                            ax.plot(x_pos, y_data,color = 'red',zorder=0,linewidth = 1)
                            ax.plot(x_pos, y_data,color = 'black',zorder=-1,linewidth = 2)
                            tick_l = [str(xx) for xx in x_spec]
                           
                            ax.set_xticks(x_pos)
                            ax.set_xticklabels(tick_l)
                            # Customize the plot
                            # graphics definitions
                           
                            th=2 # line thickness
                            formatx='{x:,.3f}' ; fs=18
                            plt.rcParams["font.size"] = fs
                            plt.rcParams['axes.facecolor'] = 'w'
                            plt.rcParams['axes.edgecolor'] = 'k'
                            plt.xticks(fontsize=30, rotation=90)
                            plt.yticks(fontsize=30)
                            plt.rcParams['axes.grid'] = False
                            plt.rcParams['axes.grid'] = True
                            plt.rcParams['grid.alpha'] = 0.5
                            plt.rcParams['grid.color'] = "#C6C6C6"
                            plt.rcParams["legend.facecolor"] ='w'
                            plt.rcParams["mathtext.default"]='regular'
                            plt.rcParams['grid.linewidth'] = th/2
                            plt.rcParams['axes.linewidth'] = 1
                            ax.set_xlabel('wavelength, nm',fontsize=35)
                            ax.set_ylabel(f'Sentinel-2 reflectance',fontsize=35)
                            ax.set_title(f'Tile no: {id_tile}',fontsize=35)
                            ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=35,markerscale=4)
                            # Show the plot
                            out_f = self.base_folder + os.sep + 'figs' + os.sep + 'bio_tracker'
                            plt.savefig(out_f + os.sep + f'S2_mean_reflectance_{id_tile}.png',dpi=200,bbox_inches='tight')
                            #plt.show()
                            plt.close()
                            
                        else:
                             logging.info('No algae in this pixel')           
            else:
                if not self.pixel_search:
                    logging.info(f'Pixels outside boundary box, searching....')
                    self.pixel_search = True
                    
                    
    def _export(self,data):
        exp_format = self.f_name.split('.')[-1]
        if exp_format == 'png':
            self._export_as_png(data)
        else:
            self._export_as_tif(data)
        #getattr(self,f'_export_as_{exp_format}')
        return None
        
    def _export_as_png(self,data):
        cmap = ListedColormap(self.colors)
        plt.imsave(self.f_name,data,cmap=cmap,dpi=600)
        return None
    
        
    def _export_as_tif(self,data):
        
        "Input: xgrid,ygrid, data paramater, the data projection, export path, name of tif file"
        
        resx = (self.xgrid[0,1] - self.xgrid[0,0])
        resy = (self.ygrid[1,0] - self.ygrid[0,0])
        transform = Affine.translation((self.xgrid.ravel()[0]),(self.ygrid.ravel()[0])) * Affine.scale(resx, resy)
        
        if resx == 0:
            resx = (self.xgrid[0,0] - self.xgrid[1,0])
            resy = (self.ygrid[0,0] - self.ygrid[0,1])
            transform = Affine.translation((self.ygrid.ravel()[0]),(self.xgrid.ravel()[0])) * Affine.scale(resx, resy)
            
        with rio.open(
        self.f_name,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        compress='lzw',
        dtype=data.dtype,
        crs=self.crs,
        transform=transform,
        ) as dst:
            dst.write(data, 1)
        
        dst.close()
        return None 
    
    