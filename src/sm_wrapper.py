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
import os



# bands = ["r_TOA_01" ,"r_TOA_02" ,"r_TOA_03", "r_TOA_04", "r_TOA_05","r_TOA_06" ,"r_TOA_07" ,"r_TOA_08", "r_TOA_09",\
#          "r_TOA_10" ,"r_TOA_11" , "r_TOA_14","r_TOA_15" ,"r_TOA_16" ,"r_TOA_17", "r_TOA_18", "r_TOA_19",\
#          "r_TOA_20","r_TOA_21" ]
    
    
bands = ["r_TOA_02" ,"r_TOA_04" ,"r_TOA_06", "r_TOA_08", "r_TOA_21"]

classify = sm.ClassifierSICE(bands=bands)


# %%

###### Import Training Data ######

dates = None

#dates=[ '2017_07_28','2019_08_02','2020_07_22','2021_07_30']
 
training_data = classify.get_training_data(d_t=dates)

# %%

###### Plot Training Data ######

classify.plot_training_data(training_data=training_data,output=True)

# %%

###### Train Model ######

model, data_split = classify.train_svm(
    training_data=training_data, kernel='rbf', export=True, prob=True)

#####

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
   
    
    
test_dict = {}

test_type = 'date'
test_type = 'ratio'
folds = np.arange(1,6)

for i in list(folds):

    model_rbf, data_split_rbf = classify.train_svm(
        training_data=training_data, c=1, kernel='rbf',test_type=test_type,fold=i,weights=True)
    acc_dict,con_dict, cm = classify.test_svm(
        model=model_rbf, data_split=data_split_rbf, mute=True)

    test_dict[i] = {'acc_dict': acc_dict,'con_dict': con_dict}
   
    
t_dates = list(test_dict.keys())
classes = ['dark_ice', 'bright_ice','purple_ice' ,'red_snow',
           'lakes', 'flooded_snow', 'melted_snow', 'dry_snow']

for cl in classes:

    acc_val = 0
    omi_val = 0
    com_val = 0
    print(f'ratio for {cl} fold nr: ')
    
    acc_list = [test_dict[td]['acc_dict'][cl]['acc'] for td in test_dict]
    omi_list = [test_dict[td]['con_dict'][cl]['omm'] for td in test_dict]
    com_list = [test_dict[td]['con_dict'][cl]['com'] for td in test_dict]
    ratio = [test_dict[td]['con_dict'][cl]['ratio'] for td in test_dict]
    cm_list = [test_dict[td]['con_dict'][cl]['confusion'] for td in test_dict]
    [print(f'{td}: {r}') for r,td in zip(ratio,list(test_dict.keys()))]
    
    acc_m = np.nanmean(np.array(acc_list))
    acc_std = np.nanstd(np.array(acc_list))
    omi_m = np.nanmean(np.array(omi_list))
    com_m = np.nanmean(np.array(com_list))
    cm_m = sum(cm_list)/5
    
    
    #print(f'Regularization paramter C is {c}')
    print(f'Mean Accuracy for {cl} : {acc_m}')
    print(f'Accuracy Standard Deviation for {cl} : {acc_std}')
    print(f'Omission for {cl} : {omi_m}')
    print(f'Comission for {cl} : {com_m}')

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
