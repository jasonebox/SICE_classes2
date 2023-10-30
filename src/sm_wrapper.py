# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:00:01 2023

@author: rabni
"""

import sicemachine as sm
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import set_start_method,get_context
import pickle
import os

classify = sm.ClassifierSICE()


#%%

###### Import Training Data ######

date = None

#date = ['2021-07-30']

training_data = classify.get_training_data()

#%%

###### Plot Training Data ######

classify.plot_training_data(training_data=training_data)

#%%

###### Train Model ######

model,data_split = classify.train_svm(training_data=training_data,kernel='rbf',export=True,prob=True)

##### 

#%%

###### Test Model ######

# Only for tesiting, not essential. Only works with more than one training date
test_dict = {}
for i in list(training_data.keys()):
    
    model_rbf,data_split_rbf = classify.train_svm(training_data=training_data,kernel='rbf',test=i)
    acc_dict = classify.test_svm(model=model_rbf,data_split=data_split_rbf)
    test_dict[i] = {'acc_dict' : acc_dict}
    
#%%

t_dates = list(test_dict.keys())
classes = ['dark_ice','bright_ice','red_snow','lakes','flooded_snow','melted_snow','dry_snow']

for cl in classes:
    
    acc_val = 0
    
    for no,td in enumerate(t_dates):
        acc_val += test_dict[td]['acc_dict'][cl]['acc']
    
    acc_m = acc_val/len(t_dates)
    
    print(f'Accuracy for {cl} : {acc_m}')

    


#%%

##### 
    
year = 2023
year_range = np.arange(year,year+1)

start_season = '07-03'
end_season = '07-04'

start_dates = [datetime.strptime(str(y) + '-' + start_season, '%Y-%m-%d').date() for y in year_range]
end_dates = [datetime.strptime(str(y) + '-' + end_season, '%Y-%m-%d').date() for y in year_range]
delta = end_dates[0] - start_dates[0]
days = [s_d + timedelta(days=i) for i in range(delta.days + 1) for s_d in start_dates]
days = sorted(list(map(lambda n: n.strftime("%Y-%m-%d"), days)))

predict_training_dates = False # Set to false if you want to predict the period above

###### Predict Dates ######
       
classify.predict_svm(dates_to_predict=days,model='import',training_predict=predict_training_dates,prob=True)



