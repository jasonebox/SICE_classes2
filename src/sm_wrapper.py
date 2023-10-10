# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:00:01 2023

@author: rabni
"""

import sicemachine as sm
import numpy as np
from datetime import datetime, timedelta

classify = sm.ClassifierSICE()

#%%

###### Import Training Data ######

training_data = classify.get_prediction_data()

#%%

###### Plot Training Data ######

classify.plot_training_data(training_data=training_data)

#%%

###### Train Model ######

model,data_split = classify.train_svm(training_data=training_data)

#%%

###### Test Model ######

classify.test_svm(model=model,data_split=data_split)


#%%

year = 2019
year_range = np.arange(year,year+1)

start_season = '08-01'
end_season = '09-01'

start_dates = [datetime.strptime(str(y) + '-' + start_season, '%Y-%m-%d').date() for y in year_range]
end_dates = [datetime.strptime(str(y) + '-' + end_season, '%Y-%m-%d').date() for y in year_range]
delta = end_dates[0] - start_dates[0]
days = [s_d + timedelta(days=i) for i in range(delta.days + 1) for s_d in start_dates]
days = sorted(list(map(lambda n: n.strftime("%Y-%m-%d"), days)))

predict_training_dates = True # Set to false if you want to predict the period above

###### Predict Dates ######

classify.predict_svm(dates_to_predict=days,model=model,training_predict=predict_training_dates)


