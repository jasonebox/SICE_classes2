#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:23:41 2023

@author: jason
"""

from glob import glob
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from pathlib import Path
from pyproj import Transformer
import rasterio

# -------------------------------------- central_wavelengths
fn='~/Dropbox/S3/ancil/band_centers.txt'
df = pd.read_csv(fn,header=None)
central_wavelengths=df.iloc[:,0]
# -------------------------------------- band names
fn='~/Dropbox/S3/ancil/band_names.txt'
df = pd.read_csv(fn,header=None)
band_names=df.iloc[0,:]
bandsx=np.arange(21)+1

# -------------------------------------- band names
fn='~/Dropbox/S3/ancil/band_info.csv'
df = pd.read_csv(fn,header=None,names=['central_wavelength','width','color'])

def reproject_points(x,y,
    in_projection: str = "4326",
    out_projection: str = "3413"):

    inProj = f"epsg:{in_projection}"
    outProj = f"epsg:{out_projection}"

    trf = Transformer.from_crs(inProj, outProj, always_xy=True)
    x_coords, y_coords = trf.transform(x, y)

    return x_coords, y_coords


def getval(lon, lat):
    idx = dat.index(lon, lat, precision=1E-6)
    # print(idx)
    # return dat.xy(*idx), z[idx]
    return z[idx]


bands=['02','03','04','05','06','08','10','11','21']
bands=['02','03','04','05','06','07','08','09','10','11']
# bands=['05','06','07','08','09','10','11']

n_bands=len(bands)
refl=np.zeros(n_bands)
surface_name='dark bare ice'
lat,lon=72.45885626, -52.86210994

datex='2021-08-23'
location='ingia'
lat,lon=72.33749155, -52.74479464

location='7151'
lat,lon=72.33749155, -52.74479464
lat,lon=70.94670867, -51.35129315

# datex='2018-07-31'
# datex='2018-07-30'
# surface_name='red snow Chamberlin gl'
# lat,lon=69.43089431, -53.03725287

# surface_name='red snow disko'
# lat,lon=69.43922745, -53.04017437

# location='Qagssimiut'
# lat,lon=61.23735143, -46.70796312

# # surface_name='purple bare ice'
# # lat,lon=72.41344168, -52.84702217

# datex='2019-08-02'
# location='Sukkertoppen'
# lat,lon=65.93554686, -50.72546121
# lat,lon=65.93357275, -50.72518441
# # lat,lon=65.92589488, -50.79918391

surface_name=f'red snow {location}'

wavelength=[]
width=[]
band_name=[]
color=[]

choice='r_TOA' ; BOA_or_TOA='TOA'
# choice='rBRR'; BOA_or_TOA='BOA'

for i,band in enumerate(bands):
    dat = rasterio.open(f"/Users/jason/0_dat/S3/opendap/Greenland_all_pixels/{datex.split('-')[0]}/{datex}_{choice}_{band}.tif")
    # read all the data from the first band
    z = dat.read()[0]
    x,y= reproject_points(lon, lat)

    refl[i]=getval(x, y)
    wavelength.append(central_wavelengths[int(band)-1])
    width.append(df.width[int(band)-1])
    color.append(df.color[int(band)-1])
    band_name.append(band)
    
    print(band,central_wavelengths[int(band)-1],i,lon,lat,refl[i])

#%%


fn='/Users/jason/Dropbox/S2/ancil/MSI_Instrument_info.xlsx'
s2_info=pd.read_excel(fn,sheet_name='S2A')
s2_info = s2_info[s2_info.band != 10]
s2_info.reset_index(drop=True, inplace=True)

print(s2_info)
s2_info.columns

#%%

# graphics definitions
th=2 # line thickness
formatx='{x:,.3f}' ; fs=18
plt.rcParams["font.size"] = fs
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['axes.grid'] = False
# plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = "#C6C6C6"
plt.rcParams["legend.facecolor"] ='w'
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['grid.linewidth'] = th/2
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['legend.fontsize'] = fs*0.8

wavelength=np.array(wavelength)
width=np.array(width)

band_name=np.array(band_name)
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(wavelength,refl,'-s',color='grey',label=f'Sentinel-3 OLCI {BOA_or_TOA} {datex} ')
# ax.set_title(f'{datex} {surface_name} {"%.2f" %lat}°N, {"%.2f"%(-lon)}°W')
ax.set_title(f'{surface_name} {"%.2f" %lat}°N, {"%.2f"%(-lon)}°W')
# plt.ylabel('reflectance, bottom of atmosphere ')
plt.ylabel('reflectance')
plt.xlabel('wavelength, nm')
osx=10
plt.xlim(np.min(wavelength)-osx,np.max(wavelength)+osx)
# plt.xlim(420,750)
osy=0.003*np.mean(refl)
# plt.ylim(np.min(refl)-osy,np.max(refl)+osy)
# plt.ylim(0.65,0.85)
# plt.ylim(0.82,1.03)

for i,band in enumerate(band_name):
    # plt.text(wavelength[i],np.min(refl),
    #           'O'+band_name[i]+' '+str(wavelength[i])+' nm',ha='center',va='bottom',
    #           fontsize=fs*0.6,rotation=90,color='#'+color[i])
    print(wavelength[i],refl[i])
    ax.plot(wavelength[i],refl[i],'s',color='#'+color[i])
    xx=[wavelength[i]-width[i]/2,wavelength[i]+width[i]/2]
    ax.plot(xx,[refl[i],refl[i]],color='#'+color[i])


# S2
fn='/Users/jason/Dropbox/S2/refl_at_points/2018-07-31-Chamberlin.txt'
fn=f'/Users/jason/Dropbox/S2/refl_at_points/{datex}-S_diskoB.txt'
# fn=f'/Users/jason/Dropbox/S2/refl_at_points/{datex}-diskoA.txt'
datexx='2018-07-31'
fn=f'/Users/jason/Dropbox/S2/refl_at_points/{datexx}-diskoA.txt'
fn='/Users/jason/Dropbox/S2/refl_at_points/2021-08-23-QASA.txt'
fn=f'/Users/jason/Dropbox/S2/refl_at_points/{datex}-{location}.txt'
# fn=f'/Users/jason/Dropbox/S2/refl_at_points/{datex}-ingiaA.txt'
s2=pd.read_csv(fn,header=None,delimiter='\t',names=['band','refl'])
# print(s2)
do_s2=1
if do_s2:
    ax.plot(s2_info.wavelength,s2.refl,'-o',label=f'Sentinel-2 MSI TOA {datex}')

# ax.set_ylim(0.62,0.8)
# ax.set_ylim(0.48,0.65)
if location=='Sukkertoppen':
    ax.set_ylim(0.5,0.8)
if location=='ingia':
    ax.set_ylim(0.68,0.82)    
if location=='Qagssimiut':
    ax.set_ylim(0.5,0.9)    
if location=='7151':
    ax.set_ylim(0.65,0.85)    
ax2 = ax.twinx()

# field
# fn='/Users/jason/Dropbox/S3/field_spectra/2018-08-Chamberlin.csv'
# field=pd.read_csv(fn,header=None,names=['wavelength','refl'])
fn='/Users/jason/Dropbox/MCIT/Spectra_Chamberlin_2018-08-24/C.csv'
field=pd.read_csv(fn,header=None,names=['name','id','wavelength','refl'], encoding = "ISO-8859-1")
ax2.plot(field.wavelength,field.refl,'--',c='r',label='red snow, Disko, M. Citterio')

fn='/Users/jason/Dropbox/PDF/ice and snow algae/Snow_algae_Dolomites.xlsx'
dolomites=pd.read_excel(fn)
print(dolomites)
ax2.plot(dolomites.wavelength,dolomites.refl,'-',c='r',label='red snow, Dolomites, B. DiMauro')
ax2.set_ylim(0.62,0.92)
ax2.set_ylabel("reflectance")

ax2.spines['right'].set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.yaxis.label.set_color('r')

# plt.legend(loc=4)
mult=0.7
ax.legend(prop={'size': fs*mult},loc=2)
ax2.legend(prop={'size': fs*mult},loc=4)

plt.show()