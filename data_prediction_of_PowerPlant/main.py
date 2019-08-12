# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:46:22 2019

@author: atif
"""

from dataset_analysis import create_dataframe
from dataset_analysis import conversion_timestamp_to_unixtime
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

import json

with open('variable_config.json', 'r') as f:
    config = json.load(f)

filepath = config['DEFAULT']['file_path']
start_pos = config['DEFAULT']['start_point_dataframe']
end_pos = config['DEFAULT']['end_point_dataframe']
date_column = config['DEFAULT']['date_column']
target_column = config['DEFAULT']['target_column']
furnace_signal_column = config['DEFAULT']['blast_furnace_signal']
print(type(date_column))
print(date_column)

#filepath = 'E:/University of Bremen MSc/masters_thesis/IAT_sebastian/dataset_26_april_3.csv'
# reading CSV file
initial_dataframe = create_dataframe(filepath)

# creating dateTime column
test_new = conversion_timestamp_to_unixtime(initial_dataframe)

test_new.head()

# dropping row_ID column. As it contains 'object' type data
test_new_1 = test_new.drop(['row ID'], axis = 1)

# Taking define number of row from the beginning
#start_pos = 0
#end_pos = 25000
multivariate_data = alter_time(test_new_1, start_pos, end_pos)

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
max_best_number = 20
sklearn_feature_best_dataframe = feature_selection_with_selectKbest(dataframe_datetime,max_best_number)

# feature selection with Pearson Correlation.

sklearn_correlation, main_correlation = pearson_correlation(sklearn_feature_best_dataframe, dataframe_datetime)

# make a dataframe with signal who is lies between a given range of correlation threshold value
correlation_threshold_value = 0.5
max_value = 0.9

main_frame = dataframe_datetime
correlated_frame = main_correlation

# main_frame = sklearn_feature_best_dataframe
# correlated_frame = sklearn_correlation

dataframe_high_correlation = make_dataframe_with_high_correlated_value(main_frame,correlated_frame,
                                                             correlation_threshold_value,max_value)
