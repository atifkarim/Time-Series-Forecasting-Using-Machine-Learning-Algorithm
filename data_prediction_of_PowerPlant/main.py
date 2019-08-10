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



filepath = 'E:/University of Bremen MSc/masters_thesis/IAT_sebastian/dataset_26_april_3.csv'
# reading CSV file
initial_dataframe = create_dataframe(filepath)

# creating dateTime column
test_new = conversion_timestamp_to_unixtime(initial_dataframe)

test_new.head()

# dropping row_ID column. As it contains 'object' type data
test_new_1 = test_new.drop(['row ID'], axis = 1)

# Taking define number of row from the beginning
start_pos = 0
end_pos = 25000
multivariate_data = alter_time(test_new_1, start_pos, end_pos)

# Changing target column and dateTime column's position
index_array=[0,-1]
req_column_name = ['dateTime','AEWIHO_T9AV2']
rearranged_dataframe = rearrange_frame(multivariate_data,req_column_name,index_array)


# Checking signal for blast furnace B for turbine 9. If the value is 100 keep the ROW except drop
blast_furnace_signal = 'DEWIHOBT9_I0'
dataframe_no_zero_value_blast_furnace = drop_zero_value_row_of_blast_furnace_signal(rearranged_dataframe,blast_furnace_signal)


# Checking target column's value. If ZERO drop the row.
