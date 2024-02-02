# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:00:01 2023

@author: rabni
"""

import sicemachine as sm
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import set_start_method, get_context
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt


# bands = ["r_TOA_01" ,"r_TOA_02" ,"r_TOA_03", "r_TOA_04", "r_TOA_05","r_TOA_06" ,"r_TOA_07" ,"r_TOA_08", "r_TOA_09",\
#           "r_TOA_10" ,"r_TOA_11" ,"r_TOA_16" ,"r_TOA_17", "r_TOA_18", "r_TOA_19"\
#           ,"r_TOA_21" ]
    
    
#bands = ["r_TOA_18" ,"r_TOA_07" ,"r_TOA_06", "r_TOA_16", "r_TOA_17"]
# bands = ["rBRR_02" ,"rBRR_04" ,"rBRR_06", "rBRR_08", "rBRR_21"]

# bands = ["rBRR_01" ,"rBRR_02" ,"rBRR_03", "rBRR_04", "rBRR_05","rBRR_06" ,"rBRR_07" ,"rBRR_08", "rBRR_09",\
#           "rBRR_10" ,"rBRR_11" ,"rBRR_16" ,"rBRR_17", "rBRR_18", "rBRR_19"\
#           ,"rBRR_21" ]

bands = ["r_TOA_02" ,"r_TOA_04" ,"r_TOA_06", "r_TOA_08", "r_TOA_21"]
bands = ["rBRR_02" ,"rBRR_04" ,"rBRR_06",'rBRR_08', "rBRR_09","rBRR_11", "rBRR_21"]

classify = sm.ClassifierSICE(bands=bands)


# %%
###### Import Training Data ######
dates = None
#test_date = '2017_07_28'
dates=['2017_07_28','2019_08_02','2018_07_31','2020_07_22','2021_07_30','2022_07_31']
training_data = classify.get_training_data(d_t=dates,local=True,s2_bio_track='surface',s3_bio_track='surface')


# %%
###### Plot Training Data #######
classify.plot_training_data(training_data=training_data,output=True)


# %%%
###### SVD Dimensionality Reduction ########
band_importance = classify.dim_redux(training_data,'cov')


# %%

###### Train Model ######


g = 16
c = 15 


model, data_split = classify.train_svm(
    training_data=training_data, kernel='rbf',gamma=g,
    c=c,export=True, prob=True)

#####

# %% 

date = '2019-08-23'

classify.predict_svm(dates_to_predict=date, model='import',
                      training_predict=False, prob=True,bio_track=True)


# %% get test data as matrix: 
    
    
train_data,train_label,test_data,test_labe = classify._train_test_format(training_data=training_data)



# %%

###### Test Model ######

# Only for tesiting, not essential. Only works with more than one training date

#C_range = np.logspace(-2, 3, 12)
#C_range = [1]
#gamma_range = np.logspace(-9, 3, 13)
#param_grid = dict(gamma=gamma_range, C=C_range)

#for c in C_range:
    

# for i in list(training_data.keys()):

#     model_rbf, data_split_rbf = classify.train_svm(
#         training_data=training_data, c=1, kernel='rbf', test=i,weights=False)
#     acc_dict,con_dict, cm = classify.test_svm(
#         model=model_rbf, data_split=data_split_rbf, mute=True)

#     test_dict[i] = {'acc_dict': acc_dict,'con_dict': con_dict}
   

C_range = np.linspace(12, 17, 7)
C_range = np.array([14.5])
G_range = np.linspace(12, 17, 7)
G_range = np.array([16.2])


#no 2 ensemble: 

C_range = np.linspace(12, 17, 7)
G_range = np.linspace(12, 17, 7)

G_range = np.array([16.2])
C_range = np.array([14.5])
#no 1 ensemble: 
    
#C_range = np.linspace(0.5, 3, 5)
#G_range = np.linspace(0.5, 3, 5)


no_r = len(C_range) * len(G_range)

h_parameters = tuple((c,g) for g in G_range for c in C_range)

model_run = 1
model_ensemble = {}
for c in C_range:
    for g in G_range:
        if c != 0:    
            test_dict = {}
            
            test_type = 'date'
            #test_type = 'ratio'
            test_date = '2017_07_28'
            #test_date = None
            folds = np.arange(1,6)
            i = None
            #for i in list(folds):
                
                
                
            model_rbf, data_split_rbf = classify.train_svm(training_data=training_data,gamma=g,
                                                           c=c, kernel='rbf',test_type=test_type,
                                                           fold=i,weights=True,test_date=test_date)
            acc_dict,con_dict, cm = classify.test_svm(model=model_rbf, 
                                                      data_split=data_split_rbf, 
                                                      mute=True)
            
            test_dict = {'acc_dict': acc_dict,'con_dict': con_dict}
               
                    
            # t_dates = list(test_dict.keys())
            
            
            # print('model_ensemble no 5')
                
                
            # model_ensemble[str(model_run)] = test_dict
            # model_run += 1
            
            classes = ['dark_ice', 'bright_ice','purple_ice','red_snow',
                        'lakes', 'flooded_snow', 'melted_snow', 'dry_snow']
            
            # for cl in classes:
            
            #     acc_val = 0
            #     omi_val = 0
            #     com_val = 0
            #     #print(f'ratio for {cl} fold nr: ')
                
            #     acc_list = [test_dict[td]['acc_dict'][cl]['acc'] for td in test_dict]
            #     omi_list = [test_dict[td]['con_dict'][cl]['omm'] for td in test_dict]
            #     com_list = [test_dict[td]['con_dict'][cl]['com'] for td in test_dict]
            #     ratio = [test_dict[td]['con_dict'][cl]['ratio'] for td in test_dict]
            #     cm_list = [test_dict[td]['con_dict'][cl]['confusion'] for td in test_dict]
            #     #[print(f'{td}: {r}') for r,td in zip(ratio,list(test_dict.keys()))]
                
            #     acc_m = np.nanmean(np.array(acc_list))
            #     acc_std = np.nanstd(np.array(acc_list))
            #     omi_m = np.nanmean(np.array(omi_list))
            #     com_m = np.nanmean(np.array(com_list))
            #     cm_m = sum(cm_list)/5
                
            #     print(f'gamma is {g}')
            #     print(f'Regularization paramter C is {c}')
            #     print(f'Mean Accuracy for {cl} : {acc_m}')
            #     print(f'Accuracy Standard Deviation for {cl} : {acc_std}')
            #     print(f'Omission for {cl} : {omi_m}')
            #     print(f'Comission for {cl} : {com_m}')
                
                #[(l/sum(lars[:,6])*100) for l in lars[:,6]] acc across all classes 


# classes = ['lakes']
# acc_list_out = []
# models = list(model_ensemble.keys()) 
# cl = 'lakes'
# for m in models:
#     test_dict = model_ensemble[m]
#     print(f'Model no {m}')
#     acc_val = 0
#     omi_val = 0
#     com_val = 0
#     #print(f'ratio for {cl} fold nr: ')
    
#     acc_list = [test_dict[td]['acc_dict'][cl]['acc'] for td in list(test_dict.keys())]
#     omi_list = [test_dict[td]['con_dict'][cl]['omm'] for td in test_dict]
#     com_list = [test_dict[td]['con_dict'][cl]['com'] for td in test_dict]
#     ratio = [test_dict[td]['con_dict'][cl]['ratio'] for td in test_dict]
#     cm_list = [test_dict[td]['con_dict'][cl]['confusion'] for td in test_dict]
#     #[print(f'{td}: {r}') for r,td in zip(ratio,list(test_dict.keys()))]
    
#     acc_m = np.nanmean(np.array(acc_list))
#     acc_std = np.nanstd(np.array(acc_list))
#     omi_m = np.nanmean(np.array(omi_list))
#     com_m = np.nanmean(np.array(com_list))
#     cm_m = sum(cm_list)/5
    
#     print(f'Mean Accuracy for {cl} : {acc_m}')
#     print(f'Accuracy Standard Deviation for {cl} : {acc_std}')
#     acc_list_out.append(acc_m)

# %% plot results from 5 - CV 

# classes = ['dark_ice','bright_ice','purple_ice','red_snow','lakes','flooded_snow','melted_snow','dry_snow']
# ff = r'C:\Users\rabni\OneDrive - GEUS\Skrivebord\SICE_classes2\c_search.csv'
# data = pd.read_csv(ff)
# def predefined_colors(class_names):
#     N_classes=len(class_names)
#     color_list=np.zeros((N_classes,4))
#     color_list[0] = (100/255,100/255,100/255, 1.0)  # dark bare ice
#     co=150
#     color_list[1] = (co/255,co/255,co/255, 1.0)  # bright bare ice
#     color_list[2] = (130/255,70/255,179/255, 1.0)  # purple ice
#     color_list[3] = (1,0,0, 1)  # red snow
#     color_list[4] = (0.8,0.8,0.3, 1.0)  # lakes
#     color_list[5] = (.5,.5,1, 1.0)  # flooded_snow
#     color_list[6] = (239/255,188/255,255/255, 1)  # melted_snow
#     color_list[7] = (0,0,0, 1.0)  # dry_snow'
#     return color_list

# color_multi = predefined_colors(classes)

# fig, ax = plt.subplots(figsize=(24, 12),dpi=600)

# # Plot the line
# # the text bounding box
# bbox = {'fc': '0.8', 'pad': 0}


# C_range = np.linspace(12, 17, 7)
# G_range = np.linspace(12, 17, 7)
# # Create a meshgrid for hyperparameter values
# h1, h2 = np.meshgrid(C_range, G_range)

# # Plot the contour
# contour = plt.contourf(h1, h2, np.array(acc_list_out).reshape(h1.shape), cmap='viridis')

# th=2 # line thickness
# formatx='{x:,.3f}' ; fs=18
# plt.rcParams["font.size"] = fs

# plt.rcParams['axes.facecolor'] = 'w'
# plt.rcParams['axes.edgecolor'] = 'k'
# plt.xticks(fontsize=30, rotation=90)
# plt.yticks(fontsize=30)
# plt.rcParams['axes.grid'] = False
# plt.rcParams['axes.grid'] = True
# plt.rcParams['grid.alpha'] = 0.5
# plt.rcParams['grid.color'] = "#C6C6C6"
# plt.rcParams["legend.facecolor"] ='w'
# plt.rcParams["mathtext.default"]='regular'
# plt.rcParams['grid.linewidth'] = th/2
# plt.rcParams['axes.linewidth'] = 1
# cbar = plt.colorbar(contour)
# cbar.set_label('Accuracy', fontsize=35)
# ax.set_xlabel('regularization parameter C',fontsize=35)
# ax.set_ylabel(f'shape parameter gamma',fontsize=35)

# # Show the plot
# plt.show()

# # for i,c in enumerate(classes):

# #     if c =='lakes':
# #         x = np.array(data[data['class']==c]['C'])
# #         y = np.array(data[data['class']==c]['acc'])
# #         z = np.array(data[data['class']==c]['std'])
    
    
# #         #ax.scatter(x, y,label=f'{c}',color = color_multi[i],s=29)
# #         #ax.scatter(x, y,color = 'black',s=45,zorder=0)
        
# #         #ax.plot(x, y,label=f'{c}',color = color_multi[i],s=29)
# #         #ax.plot(x, y,color = 'black',s=45,zorder=0)
# #         # Create shaded area around the line
# #         ax.fill_between(x, y - z, y + z, alpha=0.3, color=color_multi[i])
# #         ax.plot(x, y,color = color_multi[i],zorder=0,linewidth = 1)
# #         ax.plot(x, y,color = 'black',zorder=-1,linewidth = 2)
        
        
# # # Customize the plot
# # # graphics definitions
# # th=2 # line thickness
# # formatx='{x:,.3f}' ; fs=18
# # plt.rcParams["font.size"] = fs
# # plt.rcParams['axes.facecolor'] = 'w'
# # plt.rcParams['axes.edgecolor'] = 'k'
# # plt.xticks(fontsize=30, rotation=90)
# # plt.yticks(fontsize=30)
# # plt.rcParams['axes.grid'] = False
# # plt.rcParams['axes.grid'] = True
# # plt.rcParams['grid.alpha'] = 0.5
# # plt.rcParams['grid.color'] = "#C6C6C6"
# # plt.rcParams["legend.facecolor"] ='w'
# # plt.rcParams["mathtext.default"]='regular'
# # plt.rcParams['grid.linewidth'] = th/2
# # plt.rcParams['axes.linewidth'] = 1
# # ax.set_xlabel('regularization parameter C',fontsize=35)
# # ax.set_ylabel(f'Mean Accuracy',fontsize=35)
# # #ax.set_title('Mean Reflectance on Training Data',fontsize=35)
# # ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=35,markerscale=4)
# # # Show the plot
# # plt.show()

# %%

#####

year = 2019
year_range = np.arange(year, year+1)

start_season = '07-28'
end_season = '07-30'

start_dates = [datetime.strptime(
    str(y) + '-' + start_season, '%Y-%m-%d').date() for y in year_range]
end_dates = [datetime.strptime(
    str(y) + '-' + end_season, '%Y-%m-%d').date() for y in year_range]
delta = end_dates[0] - start_dates[0]
days = [s_d + timedelta(days=i) for i in range(delta.days + 1)
        for s_d in start_dates]
days = sorted(list(map(lambda n: n.strftime("%Y-%m-%d"), days)))

# Set to false if you want to predict the period above
predict_training_dates = False

###### Predict Dates ######
for d in days:
        
    classify.predict_svm(dates_to_predict=d, model='import',
                         training_predict=predict_training_dates, prob=True)
