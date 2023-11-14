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
from pathlib import Path

# auth.txt should contain username and password (first and second row)
cred_file="/Users/jason/Dropbox/S2/earthspy.old/secret/cred.txt"
cred_file="/Users/jason/Dropbox/S2/earthspy/secret/cred.txt"


job = es.EarthSpy(cred_file)

# os.system('cat '+cred_file)

regions=[
    "red_snow_disko","red_snow_NNW","red_snow_Ingia","red_snow_north_ice_cap",
         "red_snow_ingefield","dark_ice_CW","dark_ice_russel","dark_ice_S_Jako",
         "dark_ice_N_Jako","dark_ice_KPC","darkest_ice_SW","red_snow_nussuaq","red_snow_sukkertoppen",
         "red_snow_N","red_snow_E","red_snow_QAS","red_snow_SW","red_snow_Lyngmarksbræen","red_surface_far_N",
         ]

regions=[
    # "dark_ice_CW",
    # "dark_ice_russel",
    # "dark_ice_N_Jako",
    # "NE_ice_cap",
    # "dark_ice_KPC",
    # "red_snow_Ingia",
    # "red_snow_QAS",
    "red_snow_7151",
    # "red_snow_SW",
    # "kangerd",
    # "red_snow_SE",
    # "red_snow_disko",
    # "red_snow_sukkertoppen",
    # "CW_into_accum_area",
    # "dark_ice_S_Jako",
    # "Humboldt",
         ]
# if region=="QAS":
#     lon_w= -47.62 ; lon_e= -46.
#     lat_s= 60.94 ; lat_n= 61.3
dates=['2017_07_12','2017_07_28','2019_08_02','2020_07_22','2021_07_30','2022_07_31']
dates=['2017_07_28','2019_08_02','2020_07_22','2021_07_30','2022_07_31']
# dates=['2017_07_28']
dates=['2019_08_02']
dates=['2020_07_22']
# dates=['2022_07_31']
dates=['2023_07_05','2023_07_08']
dates=['2023_07_14']
dates=['2021_08_23']
# dates=['2023_08_23']
# dates=['2018_07_30','2018_07_31']

# cross check
# dates=['2023-07-26','2023-08-18','2023-08-22']
# dates=['2023_07_14']
# dates=['2023_07_20']
# dates=['2023_07_08','2023_07_14']
# dates=['2023_07_08']
# dates=['2023_07_05']
# dates=['2023_07_06']

for region in regions:
    
    opath="/Users/jason/0_dat/S2/"+region

    for datex in dates:
        datex=datex.replace('_','-')
        
        ofile=f'{opath}/{datex}_SENTINEL2_L1C_SM_mosaic.tif'
        test_file = Path(ofile)
        print(ofile)
        
        if not(test_file.is_file()):

            if region=="CW_into_accum_area":
                lat_n,lon_w=  66.83272098, -49.55271532
                lat_s,lon_e=  66.31696189, -46.87057457 
            
            if region=="NE_ice_cap":
                lat_n,lon_w=  76.24683191, -21.71139441
                lat_s,lon_e=  75.90719610, -20.60793019  
            
            if region=="Humboldt":
                lat_n,lon_w=  79.37871407, -67.09460429
                lat_s,lon_e=  78.92442390, -59.72390359    

            if region=="red_snow_disko":
                lat_n,lon_w=  69.95901396, -54.92572164
                lat_s,lon_e=  69.23436514, -52.23660637
                # datex='2020-07-22' # nothing for Lyngmarksbræen or Disko this date
                # datex="2022-07-31" # 

            if region=="red_snow_SE":
                lat_n,lon_w=  66.16100501, -39.33712652
                lat_s,lon_e=  65.47311193, -38.12260677
                # lat_n,lon_w=  66.42874315, -39.07792043
                # lat_s,lon_e=  65.94896570, -37.96369898 # 2022 07 31 didnt work
            
            if region=="red_snow_NNW":
                lat_n,lon_w=  82.13698217, -59.46714861
                lat_s,lon_e=  81.75274274, -56.50843801
                # datex='2020-07-22' # nothing for Lyngmarksbræen or Disko this date

            if region=="red_snow_7151":
                lat_n,lon_w=  71.05717126, -51.92221507
                lat_s,lon_e=  70.88402739, -51.09723371
                # datex='2020-07-22' # nothing for Lyngmarksbræen or Disko this date
            
            if region=="red_snow_Ingia":
                lat_n,lon_w=  72.44657169, -53.70165793
                lat_s,lon_e=  72.16005336, -51.79757873
                lat_n,lon_w=  72.39706856, -53.54892822
                lat_s,lon_e=  72.13349257, -52.19258129
                lat_n,lon_w=  72.49947725, -53.69076646
                lat_s,lon_e=  72.11570101, -52.08615704
                # datex='2020-07-22' # nothing for Lyngmarksbræen or Disko this date
                # datex="2022-07-31" # cloudy
            
            
            if region=="red_snow_north_ice_cap":
                lat_n,lon_w=  77.15264291, -69.83415629
                lat_s,lon_e=  76.77444794, -66.00702668
                # datex='2020-07-22' # 
            
            
            if region=="red_snow_ingefield":
                lat_n,lon_w= 78.30730042, -72.63508627
                lat_s,lon_e=  78.13976596, -69.17618255
                # datex='2020-07-22' # 
                # datex="2022-07-31"
                # datex='2017-07-28' # cloudy
            
            if region=="dark_ice_CW":
                lat_n,lon_w= 68.20410195, -49.44219500
                lat_s,lon_e= 67.28043617, -48.23075762
                # # # datex='2020-07-22' # 
                # # datex="2022-07-31" # no good
                # datex='2017-07-12' # this date seems not sufficiently melty
                # datex='2017-07-28'
                # datex="2022-07-31" # no data
                # datex='2019-08-02'
            
            if region=="dark_ice_russel":
                # lat_n,lon_w= 67.40817564, -49.66974025
                # lat_s,lon_e=  66.90277430, -48.24903528
                # # expanded south
                lat_n,lon_w= 67.37844453, -49.59951873
                lat_s,lon_e=  66.13842767, -48.15760897
                # datex="2022-07-31" # 
                # datex='2017-07-12' # this date seems not sufficiently melty
                # datex='2017-07-28'
            
            if region=="dark_ice_S_Jako":
                lat_n,lon_w= 69.05869006, -49.92422032
                lat_s,lon_e= 68.02258800, -47.58831331
                # datex="2022-07-31" # didnt work
            
            if region=="dark_ice_N_Jako":
                lat_n,lon_w= 69.76244092, -50.16415931
                lat_s,lon_e= 69.28754601, -48.49638642
                # datex='2019-08-02'
                # datex="2022-07-31" # didnt work
            
            if region=="dark_ice_KPC":
                lat_n,lon_w= 80.38635797, -27.36904445
                lat_s,lon_e=  79.49826544, -23.31783046
                # # datex="2022-07-31" # 
                # datex='2017-07-12' # this date seems not sufficiently melty
                # datex='2017-07-28'
            
            if region=="darkest_ice_SW":
                lat_n,lon_w= 65.84821040, -50.57043663
                lat_s,lon_e=  65.59429189, -49.68200145
                # datex='2020-07-22' # 

            if region=="kangerd": 
                lat_n,lon_w= 69.19141331, -33.35031272# crevassy
                lat_s,lon_e=  68.55106831, -32.72508392# crevassy
                lat_n,lon_w= 68.95674969, -33.15361657
                lat_s,lon_e=  68.54387532, -32.43512756

            
            if region=="red_snow_nussuaq":
                lat_n,lon_w= 70.76287368, -53.33050709
                lat_s,lon_e=  70.16456482, -50.94977724
                # datex='2020-07-22' # no good this date
            
            if region=="red_snow_sukkertoppen":
                # datex="2022-07-31" # no data
                # datex='2017-07-28'
                lat_n,lon_w=  66.42200123, -53.31628834
                lat_s,lon_e=  65.38968608, -49.99255904
                # # datex='2021-07-30'
                # # datex='2019-08-02'
                # datex='2020-07-22'
            
            # if region=="red_snow_rink"
            #     lat_n,lon_w=  72.36604639, -53.42398332
            #     lat_s,lon_e=  72.16282891, -51.83309634
            #     # datex='2019-08-02'
            
            if region=="red_snow_N":
                lat_n,lon_w=  81.06938651, -22.36860453
                lat_s,lon_e=  80.40332693, -20.72150342
                # datex='2019-08-02'
            
            if region=="red_snow_E":
                lat_n,lon_w=  71.50994597, -22.27315874
                lat_s,lon_e=  70.39353831, -21.50879502
                # datex='2019-08-02'
            
            if region=="red_snow_QAS":
                lat_n,lon_w=  61.37392375, -47.47809045
                lat_s,lon_e=  61.18073837, -46.26391748
                lat_n,lon_w=  61.23892844, -46.99238137 # 2023
                lat_s,lon_e=  61.11060722, -46.49503037
                lat_n,lon_w=  61.29387964, -46.97965734 # 2017 07 28
                lat_s,lon_e=  61.11060722, -46.30706108
                lat_n,lon_w=  61.27263485, -47.13028240 # 2023 expanded
                lat_s,lon_e=  61.08056265, -46.47359191
                
                # datex='2019-08-02'
            
            if region=="red_snow_SW":
                lat_n,lon_w=  63.87385184, -50.95036483
                lat_s,lon_e=  63.59879769, -49.78344783
                # lat_n,lon_w=  63.67979039, -50.25377501
                # lat_s,lon_e=  63.67979039, -50.25377501            
            if region=="red_snow_Lyngmarksbræen":
                lat_n,lon_w=  69.36493894, -53.78286718
                lat_s,lon_e=  69.27859525, -53.44499144
                # datex='2017-07-26'
                # datex='2020-07-22' # nothing for Lyngmarksbræen or Disko this date
            
            if region=="red_surface_far_N":
                lat_n,lon_w=  81.28814866, -22.66260736
                lat_s,lon_e=  80.20590251, -20.02470187
            
            sensor='S2'
            
            if sensor=='S2':
            
            
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
            

            job.set_query_parameters(
                bounding_box=[lon_w,lat_s, lon_e,lat_n],
                evaluation_script=eval_script,
                time_interval=[datex, datex],
                data_collection=collection,
                store_folder=opath,
                remove_splitboxes=True
            )
            
            # and off it goes!
            job.send_sentinelhub_requests()
        else:
            print("file exists")