#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 13:28:17 2022

@author: jason

collections
https://sentinelhub-py.readthedocs.io/en/latest/reference/sentinelhub.data_collections.html

"""
import sys
import os

sys.path.append('/Users/jason/Dropbox/S2/earthspy.old/')

import earthspy.earthspy as es

# auth.txt should contain username and password (first and second row)
cred_file="/Users/jason/Dropbox/S2/earthspy.old/secret/cred.txt"
cred_file="/Users/jason/Dropbox/S2/earthspy/secret/cred.txt"


job = es.EarthSpy(cred_file)

# os.system('cat '+cred_file)

region="QAS"
lon_w= -47.62 ; lon_e= -46.
lat_s= 60.94 ; lat_n= 61.3

region="red_snow"
lat_n,lon_w=  72.02395476, -54.29797315
lat_s,lon_e=  71.92711899, -53.90419565

region="red_snow_disko"
lat_n,lon_w=  69.89715506, -53.86865057
lat_s,lon_e=  69.37465286, -52.14642320

region="red_snow_sukkertoppen"
lat_n,lon_w=  66.32144891, -53.41496170
lat_s,lon_e=  65.65225591, -49.61648123

# 2017-07-28 red_snow_sukkertoppen
lat_n,lon_w=  65.85572276, -52.46412338
lat_s,lon_e=  65.60115679, -51.70678723


# red_snow_sukkertoppen
lat_n,lon_w=  66.29996022, -53.24558246
lat_s,lon_e=  65.75210554, -49.95195323
datex='2021-07-30'
datex='2019-08-02'


region="red_snow_rink"
lat_n,lon_w=  72.36604639, -53.42398332
lat_s,lon_e=  72.16282891, -51.83309634
datex='2019-08-02'

region="red_snow_N"
lat_n,lon_w=  81.06938651, -22.36860453
lat_s,lon_e=  80.40332693, -20.72150342
datex='2019-08-02'

region="red_snow_E"
lat_n,lon_w=  71.50994597, -22.27315874
lat_s,lon_e=  70.39353831, -21.50879502
datex='2019-08-02'

# region="red_snow_QAS"
# lat_n,lon_w=  61.37392375, -47.47809045
# lat_s,lon_e=  61.18073837, -46.26391748
# datex='2019-08-02'

region="red_snow_SW"
lat_n,lon_w=  63.87385184, -50.95036483
lat_s,lon_e=  63.59879769, -49.78344783
datex='2019-08-02'

region="red_snow_Lyngmarksbræen"
lat_n,lon_w=  69.36493894, -53.78286718
lat_s,lon_e=  69.27859525, -53.44499144
datex='2017-07-26'

# region="red_surface_far_N"
# lat_n,lon_w=  81.58913094, -30.88874646
# lat_s,lon_e=  81.30497029, -28.82698918

# region="flooded"
# lat_n,lon_w=  80.02297743, -27.96364998
# lat_s,lon_e=  79.50128003, -23.95036367

# region="Qaleralik"
# lat_n,lon_w=  61.10737658, -46.92220244
# lat_s,lon_e=  60.96888144, -46.51173234

# region="Sukkertoppen"
# lat_n,lon_w=  66.28376871, -52.22176177
# lat_s,lon_e=  65.59140845, -49.61996103


# eval_script = """
# //VERSION=3

# //VERSION=3
# function setup() {
#   return {
#     input: ["B02", "B03", "B04", "dataMask"],
#     output: {bands: 3}
#   };
# }

# // Set gain
# const gain = 4

# function evaluatePixel(sample) {
#   var band1 = sample.B02
#   var band2 = sample.B03
#   var band3 = sample.B04
#   return [band1 * gain, band2 * gain, band3 * gain]
# }
# # """


# eval_script = """
#     //VERSION=3
#     function setup(){
#       return{
#         input: ["B02", "B03", "B04", "dataMask"],
#         output: {bands: 4}
#       }
#     }

#     function evaluatePixel(sample){
#       // Set gain for visualisation
#       let gain = 2.5;
#       // Return RGB
#       return [sample.B04 * gain, sample.B03 * gain, sample.B02 * gain];
#     }

#     """

# # S2 fancy
# eval_script = """
# //VERSION=3

# function setup() {
#   return {
#     input: ["B04", "B03", "B02", "dataMask"],
#     output: { bands: 4 }
#   };
# }

# // Contrast enhance / highlight compress


# const maxR = 3.0; // max reflectance

# const midR = 0.13;
# const sat = 1.3;
# const gamma = 2.3;

# // remove the minimum Rayleigh scattering (check the Himalayas)

# const ray = { r: 0.013, g: 0.024, b: 0.041 };

# function evaluatePixel(smp) {
#   const rgbLin = satEnh(sAdj(smp.B04 - ray.r), sAdj(smp.B03 - ray.g), sAdj(smp.B02 - ray.b));
#   return [sRGB(rgbLin[0]), sRGB(rgbLin[1]), sRGB(rgbLin[2]), smp.dataMask];
# }

# const sAdj = (a) => adjGamma(adj(a, midR, 1, maxR));

# const gOff = 0.01;
# const gOffPow = Math.pow(gOff, gamma);
# const gOffRange = Math.pow(1 + gOff, gamma) - gOffPow;

# const adjGamma = (b) => (Math.pow((b + gOff), gamma) - gOffPow) / gOffRange;

# // Saturation enhancement

# function satEnh(r, g, b) {
#   const avgS = (r + g + b) / 3.0 * (1 - sat);
#   return [clip(avgS + r * sat), clip(avgS + g * sat), clip(avgS + b * sat)];
# }

# const clip = (s) => s < 0 ? 0 : s > 1 ? 1 : s;

# //contrast enhancement with highlight compression

# function adj(a, tx, ty, maxC) {
#   var ar = clip(a / maxC, 0, 1);
#   return ar * (ar * (tx / maxC + ty - 1) - ty) / (ar * (2 * tx / maxC - 1) - tx / maxC);
# }

# const sRGB = (c) => c <= 0.0031308 ? (12.92 * c) : (1.055 * Math.pow(c, 0.41666666666) - 0.055);
#     """


sensor='S2'
# sensor='L8'
# sensor='L7'

if sensor=='L7':

    # QAS
    dates=["2011-08-19"] # Qaleralik OK
    dates=["2010-08-16"] # Qaleralik OK
    dates=["2009-09-14"] # Qaleralik OK but snowy
    dates=["2009-07-28"] # Qaleralik OK
    dates=["2008-08-26"] # Qaleralik OK
    dates=["2007-08-24"] # Qaleralik OK
    dates=["2006-08-05"] # Qaleralik OK
    dates=["2005-09-19"] # Sermilik OK
    
    collection="LANDSAT_ETM_L2" # L8

    # Landsat 8
    eval_script = """
    //VERSION=3
    
    let minVal = 0.0;
    let maxVal = 0.99;
    
    let viz = new HighlightCompressVisualizer(minVal, maxVal);
    
    function evaluatePixel(samples) {
        let val = [samples.B03, samples.B02, samples.B01];
        val = viz.processList(val);
        val.push(samples.dataMask);
        return val;
    }
    
    function setup() {
      return {
        input: [{
          bands: [
            "B01",
            "B02",
            "B03",
            "dataMask"
          ]
        }],
        output: {
          bands: 4
        }
      }
    }
    """
    
if sensor=='L8':

    # QAS
    dates=["2017-09-12"]
    dates=["2016-06-30"]
    dates=["2015-08-15"]
    dates=["2014-08-19"]
    dates=["2013-09-26"] # snowy but ok
    dates=["2013-09-01"]
    
    collection="LANDSAT_OT_L1" # L8

    # Landsat 8
    eval_script = """
    //VERSION=3
    
    let minVal = 0.0;
    let maxVal = 0.99;
    
    let viz = new DefaultVisualizer(minVal, maxVal);
    
    function evaluatePixel(samples) {
        let val = [samples.B04, samples.B03, samples.B02];
        val = viz.processList(val);
        val.push(samples.dataMask);
        return val;
    }
    
    function setup() {
      return {
        input: [{
          bands: [
            "B02",
            "B03",
            "B04",
            "dataMask"
          ]
        }],
        output: {
          bands: 4
        }
      }
    }
    """


if sensor=='S2':

    dates=["2017-07-31","2018-08-10","2019-08-10","2020-08-19","2021-08-27","2022-09-08","2023-07-10"]
    dates=["2023-09-13"]
    dates=["2023-09-18"]
    dates=["2022-07-31"]
    dates=["2017-07-28"]
    dates=[datex]

    collection="SENTINEL2_L1C"

    eval_script = """
    
    //VERSION=3
    let minVal = 0.0;
    let maxVal = 0.9;
    
    let viz = new HighlightCompressVisualizer(minVal, maxVal);
    
    function setup() {
        return {
        input: ["B04", "B03", "B02","dataMask"],
        output: { bands: 4 }
      };
    }
    
    function evaluatePixel(samples) {
        let val = [samples.B04, samples.B03, samples.B02,samples.dataMask];
        return viz.processList(val);
    }
    """


# eval_script = """
#     //VERSION=3
#     function setup(){
#       return{
#         input: ["B11"],
#         output: {bands: 1,sampleType:"FLOAT32"}
#       }
#     }

#      function evaluatePixel(sample){
#       // Set gain for visualisation
#       // Return RGB
#       return [sample.B11];
#     }

#     """
    
# # L ETM
# eval_script = """
# //VERSION=3

# let minVal = 0.0;
# let maxVal = 0.4;

# let viz = new DefaultVisualizer(minVal, maxVal);

# function evaluatePixel(samples) {
#     let val = [samples.B04, samples.B03, samples.B02];
#     val = viz.processList(val);
#     val.push(samples.dataMask);
#     return val;
# }

# function setup() {
#   return {
#     input: [{
#       bands: [
#         "B02",
#         "B03",
#         "B04",
#         "dataMask"
#       ]
#     }],
#     output: {
#       bands: 4
#     }
#   }
# }
# """

# eval_script = """
# //VERSION=3

# function setup() {
#   return {
#     input: ["B04", "B03", "B02", "dataMask"],
#     output: { bands: 4 }
#   };
# }

# // Contrast enhance / highlight compress


# const maxR = 3.0; // max reflectance

# const midR = 0.13;
# const sat = 1.3;
# const gamma = 2.3;

# // remove the minimum Rayleigh scattering (check the Himalayas)

# const ray = { r: 0.013, g: 0.024, b: 0.041 };

# function evaluatePixel(smp) {
#   const rgbLin = satEnh(sAdj(smp.B04 - ray.r), sAdj(smp.B03 - ray.g), sAdj(smp.B02 - ray.b));
#   return [sRGB(rgbLin[0]), sRGB(rgbLin[1]), sRGB(rgbLin[2]), smp.dataMask];
# }

# const sAdj = (a) => adjGamma(adj(a, midR, 1, maxR));

# const gOff = 0.01;
# const gOffPow = Math.pow(gOff, gamma);
# const gOffRange = Math.pow(1 + gOff, gamma) - gOffPow;

# const adjGamma = (b) => (Math.pow((b + gOff), gamma) - gOffPow) / gOffRange;

# // Saturation enhancement

# function satEnh(r, g, b) {
#   const avgS = (r + g + b) / 3.0 * (1 - sat);
#   return [clip(avgS + r * sat), clip(avgS + g * sat), clip(avgS + b * sat)];
# }

# const clip = (s) => s < 0 ? 0 : s > 1 ? 1 : s;

# //contrast enhancement with highlight compression

# function adj(a, tx, ty, maxC) {
#   var ar = clip(a / maxC, 0, 1);
#   return ar * (ar * (tx / maxC + ty - 1) - ty) / (ar * (2 * tx / maxC - 1) - tx / maxC);
# }

# const sRGB = (c) => c <= 0.0031308 ? (12.92 * c) : (1.055 * Math.pow(c, 0.41666666666) - 0.055);
# """


# collection="SENTINEL2_L2A"
# collection="LANDSAT_ETM_L2"
# collection="LANDSAT_MSS_L1"
# as simple as it gets

for datex in dates:
    job.set_query_parameters(
        bounding_box=[lon_w,lat_s, lon_e,lat_n],
        evaluation_script=eval_script,
        # time_interval=["2021-08-01", "2021-08-22"], #nothing!
        # time_interval=["2022-06-01", "2021-06-30"], #
        # time_interval=["2021-08-23", "2021-08-30"],
        # time_interval=["2021-06-25", "2021-06-25"],
        # time_interval=["2021-06-24", "2021-06-24"],
        # time_interval=["2022-04-20", "2022-04-25"],
        # time_interval=["2016-08-24", "2016-08-24"],# best
        # time_interval=["2017-09-12", "2017-09-12"],# Landsat 8
        time_interval=[datex, datex],# QAS
        # time_interval=["2017-08-13", "2017-08-13"],#
        # time_interval=["2018-08-10", "2018-08-10"],# best
        # time_interval=["2019-08-28", "2019-08-28"],# best
        # time_interval=["2019-08-02", "2019-08-02"],# best
        # time_interval=["2019-08-10", "2019-08-10"],# best
        # time_interval=["2020-08-19", "2020-08-19"],# best
        # time_interval=["2021-08-27", "2021-08-27"],# best
        # time_interval=["2022-08-22", "2022-08-22"],# best
        # time_interval=["2022-09-08", "2022-09-08"],
        # time_interval=["2023-06-30", "2023-06-30"], # best
        # time_interval=["2023-07-05", "2023-07-05"], # white
        # time_interval=["2023-07-10", "2023-07-10"], # not yet gathered
        data_collection=collection,
        store_folder="/Users/jason/0_dat/S2/"+region,
        remove_splitboxes=True
    )
    
    # and off it goes!
    job.send_sentinelhub_requests()