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
import time

bands = ["r_TOA_02" ,"r_TOA_04" ,"r_TOA_06", "r_TOA_08", "r_TOA_21"]

classify = sm.ClassifierSICE()


# %%

###### Import Training Data ######

date = None

#date = ['2021-07-30']

# training_data = classify.get_training_data()

#dates = ['2021-07-30']
dates=['2017-07-28','2021-07-30']
dates=['2017_07_12','2017_07_28','2019_08_02','2020_07_22','2021_07_30','2022_07_31']
dates=[             '2017_07_28','2019_08_02','2020_07_22','2021_07_30','2022_07_31'] # 2017_07_12 issue: bright ice is dark, and red snow is too bright, dark ice also bright dark ice
dates=[             '2017_07_28','2019_08_02','2020_07_22'] # 2017_07_12 issue: bright ice is dark, and red snow is too bright, dark ice also bright dark ice
# dates=['2017_07_28','2019_08_02','2020_07_22','2021_07_30'] # 2017_07_12 and 2022 too bright dark ice

st = time.time()

# training_data = classify.get_training_data(d_t=dates)
training_data = classify.get_training_data(d_t=dates)

elapsed_time = time.time() - st
print('elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#%%

##%% Plot Training Data  
import matplotlib.pyplot as plt

# graphics definitions
th=2 # line thickness
formatx='{x:,.3f}' ; fs=12
plt.rcParams["font.size"] = fs
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = "#C6C6C6"
plt.rcParams["legend.facecolor"] ='w'
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['grid.linewidth'] = th/2
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['legend.fontsize'] = fs*0.8

# classify.plot_training_data(training_data=training_data)


classify.plot_training_data(training_data=training_data,output=False)

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
    
test_dict = {}

for i in list(training_data.keys()):

    model_rbf, data_split_rbf = classify.train_svm(
        training_data=training_data, c=1, kernel='rbf', test=i)
    acc_dict, cm = classify.test_svm(
        model=model_rbf, data_split=data_split_rbf, mute=True)

    test_dict[i] = {'acc_dict': acc_dict}

t_dates = list(test_dict.keys())
classes = ['dark_ice', 'bright_ice', 'red_snow',
           'lakes', 'flooded_snow', 'melted_snow', 'dry_snow']

for cl in classes:

    acc_val = 0

    for no, td in enumerate(t_dates):
        acc_val += test_dict[td]['acc_dict'][cl]['acc']

    acc_m = acc_val/len(t_dates)

    #print(f'Regularization paramter C is {c}')
    print(f'Accuracy for {cl} : {acc_m}')


# %%

#####

year = 2021
year_range = np.arange(year, year+1)

start_season = '08-23'
end_season = '08-23'

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

classify.predict_svm(dates_to_predict=days, model='import',
                     training_predict=predict_training_dates, prob=True)
