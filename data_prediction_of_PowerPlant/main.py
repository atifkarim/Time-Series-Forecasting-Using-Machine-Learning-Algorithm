# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:46:22 2019

@author: atif
"""
import json
import collections
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import *
import matplotlib.pyplot as plt
from sklearn import linear_model

from dataset_analysis import create_dataframe
from dataset_analysis import conversion_timestamp_to_unixtime
from dataset_analysis import remove_rw_column
from dataset_analysis import alter_time
from dataset_analysis import rearrange_frame
from dataset_analysis import drop_zero_value_row_of_blast_furnace_signal
from dataset_analysis import drop_zero_value_row_of_target_signal
from dataset_analysis import drop_column_with_same_value
from dataset_analysis import drop_nan_value
from dataset_analysis import drop_row
from dataset_analysis import drop_string_column
from dataset_analysis import feature_selection_with_selectKbest
from dataset_analysis import pearson_correlation
from dataset_analysis import make_dataframe_with_high_correlated_value
from dataset_analysis import dataframe_date_time_type
from dataset_analysis import my_sum
from dataset_analysis import draw_graph

from model_file import make_dataset
from model_file import scikit_learn_model
from  model_file import plot_graph
from model_file import evaluation_metrices
e = my_sum(9,2)
print(e)


with open('variable_config.json', 'r') as f:
    config = json.load(f)

filepath = config['DEFAULT']['file_path']
filepath_ubuntu = config['DEFAULT']['file_path_ubuntu']
start_pos = config['DEFAULT']['start_point_dataframe']
end_pos = config['DEFAULT']['end_point_dataframe']
date_column = config['DEFAULT']['date_column']
target_column = config['DEFAULT']['target_column']
furnace_signal_column = config['DEFAULT']['blast_furnace_signal']
max_best_number = config['DEFAULT']['max_best_number']
correlation_threshold_min_value = config['DEFAULT']['correlation_threshold_min_value']
correlation_threshold_max_value = config['DEFAULT']['correlation_threshold_max_value']
print(type(correlation_threshold_max_value))
print(correlation_threshold_max_value)

#filepath = 'E:/University of Bremen MSc/masters_thesis/IAT_sebastian/dataset_26_april_3.csv'
# reading CSV file
initial_dataframe = create_dataframe(filepath_ubuntu)

# creating dateTime column
test_new = conversion_timestamp_to_unixtime(initial_dataframe)

test_new.head()

# dropping row_ID column. As it contains 'object' type data
test_new_1 = test_new.drop(['row ID'], axis = 1)

test_new_2 = remove_rw_column(test_new_1)

# Taking define number of row from the beginning
multivariate_data = alter_time(test_new_2, start_pos, end_pos)

# Changing target column and dateTime column's position
index_array=[0,-1]
req_column_name = [date_column, target_column]
rearranged_dataframe = rearrange_frame(multivariate_data,req_column_name,index_array)


# Checking signal for blast furnace B for turbine 9. If the value is 100 keep the ROW except drop
dataframe_no_zero_value_blast_furnace = drop_zero_value_row_of_blast_furnace_signal(rearranged_dataframe,furnace_signal_column)


# Checking target column's value. If ZERO drop the row.
#target_signal = 'AEWIHO_T9AV2'
dataframe_reset = dataframe_no_zero_value_blast_furnace.reset_index()
dataframe_no_zero_value_target_column = drop_zero_value_row_of_target_signal(dataframe_reset, target_column)


# Drop the column which has sam evalue in every ROW
dataframe_drop_column_with_same_value = drop_column_with_same_value(dataframe_no_zero_value_target_column)

# Drop the ROW which has NAN value
multivariate_data_drop_nan = drop_nan_value(dataframe_drop_column_with_same_value)


# Drop the row who has consecutive same value
dataframe_drop_row_consecutive_same_value = drop_row(multivariate_data_drop_nan)

# Drop the column who has 'objet' type value
dataframe_no_string = drop_string_column(dataframe_drop_row_consecutive_same_value)


# Make dataframe with dateTime index
dataframe_datetime = dataframe_no_string.set_index('dateTime')


# Feature selection with Sklearn feature best technique
sklearn_feature_best_dataframe = feature_selection_with_selectKbest(dataframe_datetime,max_best_number)

# feature selection with Pearson Correlation.

sklearn_correlation, main_correlation = pearson_correlation(sklearn_feature_best_dataframe, dataframe_datetime)

# make a dataframe with signal who is lies between a given range of correlation threshold value
main_frame = dataframe_datetime
correlated_frame = main_correlation

# main_frame = sklearn_feature_best_dataframe
# correlated_frame = sklearn_correlation

dataframe_high_correlation = make_dataframe_with_high_correlated_value(main_frame,correlated_frame,
                                                             correlation_threshold_min_value, correlation_threshold_max_value)

df = dataframe_date_time_type(dataframe_datetime)

dict_of_dates = {k: v for k, v in df.groupby('Date')}
dict_of_day_type = {k:v for k,v in df.groupby('TypeofDAY')}
dict_of_day_name = {k:v for k,v in df.groupby('day_name')}


date_key_value = collections.OrderedDict(dict_of_dates)
day_type_key_value = collections.OrderedDict(dict_of_day_type)
day_name_key_value = collections.OrderedDict(dict_of_day_name)

current_directory = os.getcwd()
print(current_directory)
address = 'image_folder'
final_directory = current_directory+'/'+str(address)
if not os.path.exists(final_directory):
    os.makedirs(final_directory)
    print('created : ', final_directory)
else:
    pass

draw_graph_date = draw_graph(date_key_value,dict_of_dates, target_column,final_directory, subfolder_name = 'date_fig')
draw_graph_week = draw_graph(day_type_key_value,dict_of_day_type, target_column,final_directory, subfolder_name = 'week_fig')
draw_graph_day = draw_graph(day_name_key_value,dict_of_day_name, target_column,final_directory, subfolder_name = 'day_fig')


train_input, train_output, test_input, test_output = make_dataset(dataframe_high_correlation)

model_list = [LinearRegression(), ExtraTreesRegressor()]
name = ['LinearRegression','ExtraTreesRegressor']

predicted_output = scikit_learn_model(model_list, name, train_input, train_output, test_input, test_output)

graph = plot_graph(test_output, predicted_output)

evaluate_model = evaluation_metrices(test_output, predicted_output)

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import mean_absolute_error
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

lr = 0.01

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

batch_size=32
epochs= 100

def NN_model():
    NN_model = Sequential()
    NN_model.add(Dense(32, kernel_initializer='normal',input_dim = train_input.shape[1], activation='relu'))
    NN_model.add(Dense(64, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(2048, kernel_initializer='normal', activation='relu'))
#     NN_model.add(Dense(1, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(1))
    return NN_model
NN_model=NN_model()
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error','accuracy'])
NN_model.summary()


NN_model.fit(train_input, train_output, epochs=epochs, batch_size=batch_size)

predicted_output = NN_model.predict(test_input)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
print('r_2 statistic: %.2f' % r2_score(test_output,predicted_output))
print("Mean_absolute_error: %.2f" % mean_absolute_error(test_output,predicted_output))
print("Mean squared error: %.2f" % mean_squared_error(test_output,predicted_output))
RMSE=math.sqrt(mean_squared_error(test_output,predicted_output))
print('RMSE: ',RMSE)