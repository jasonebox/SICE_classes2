# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:00:01 2023

@author: rabni
"""

import sicemachine as sm
import numpy as np
from datetime import datetime, timedelta
import time

classify = sm.ClassifierSICE()

#%% Import Training Data  

dates = None # for all cases, but has an 'old' data issue

#dates = ['2021-07-30']
dates=['2017-07-28','2021-07-30']
dates=['2017_07_12','2017_07_28','2019_08_02','2020_07_22','2021_07_30','2022_07_31']
dates=[             '2017_07_28','2019_08_02','2020_07_22','2021_07_30','2022_07_31'] # 2017_07_12 issue: bright ice is dark, and red snow is too bright, dark ice also bright dark ice
# dates=['2017_07_28','2019_08_02','2020_07_22','2021_07_30'] # 2017_07_12 and 2022 too bright dark ice

st = time.time()

training_data = classify.get_training_data(d_t=dates)

elapsed_time = time.time() - st
print('elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#%% Plot Training Data  
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

classify.plot_training_data(training_data=training_data)

#%%  Train Model  

st = time.time()

model,data_split = classify.train_svm(training_data=training_data)

elapsed_time = time.time() - st
print('elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


#%% Test Model  

# Only for tesiting, not essential. Only works with more than one training date

# classify.test_svm(model=model,data_split=data_split)


#%% load predict dates and predict

year = 2019
year = 2021
# year = 2023
year_range = np.arange(year,year+1)

start_season = '08-21'
end_season = '08-27'

start_season = '08-23'
end_season = '08-23'

# start_season = '08-01' ; end_season = '08-31'

# start_season = '07-03' ; end_season = '07-21'

# start_season = '07-08' ; end_season = start_season
# start_season = '07-05' ; end_season = start_season
# start_season = '07-06' ; end_season = start_season

# start_season = '07-14' ; end_season = start_season
# start_season = '07-07' ; end_season = start_season # wide red snow area S Greenland

start_dates = [datetime.strptime(str(y) + '-' + start_season, '%Y-%m-%d').date() for y in year_range]
end_dates = [datetime.strptime(str(y) + '-' + end_season, '%Y-%m-%d').date() for y in year_range]
delta = end_dates[0] - start_dates[0]
dates = [s_d + timedelta(days=i) for i in range(delta.days + 1) for s_d in start_dates]
dates = sorted(list(map(lambda n: n.strftime("%Y-%m-%d"), dates)))
print(dates)

st = time.time()

predict_training_dates = False # Set to false if you want to predict the period above

# Predict Dates
classify.predict_svm(dates_to_predict=dates,model=model,training_predict=predict_training_dates)

elapsed_time = time.time() - st
print('elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

