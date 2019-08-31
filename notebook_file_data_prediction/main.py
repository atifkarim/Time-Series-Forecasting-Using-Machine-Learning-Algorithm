import numpy as np
a = np.zeros((156816, 36, 53806), dtype='uint8')



import json
import collections
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn import linear_model
from sklearn import tree

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataset_analysis import create_dataframe
from dataset_analysis import conversion_timestamp_to_unixtime
from dataset_analysis import remove_rw_column
from dataset_analysis import alter_time
from dataset_analysis import rearrange_frame
from dataset_analysis import drop_zero_value_row_of_blast_furnace_signal
from dataset_analysis import drop_zero_value_row_of_target_signal

from dataset_analysis import check_A_B_blast_furnace
from dataset_analysis import no_zero_value_in_target

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
from dataset_analysis import draw_feature_vs_target


from dataset_analysis import remove_rw_column_1



from model_file import make_dataset
from model_file import scikit_learn_model
from model_file import plot_graph
from model_file import evaluation_metrices


with open('variable_config.json', 'r') as f:
    config = json.load(f)

filepath = config['DEFAULT']['file_path']
filepath_ubuntu = config['DEFAULT']['file_path_ubuntu']
filepath_ubuntu_1 = config['DEFAULT']['file_path_ubuntu_1']
filepath_server = config['DEFAULT']['file_path_server']

file_correlation = config['DEFAULT']['file_correlation']
matrix_correlation = config['DEFAULT']['matrix_correlation']

start_pos = config['DEFAULT']['start_point_dataframe']
end_pos = config['DEFAULT']['end_point_dataframe']
date_column = config['DEFAULT']['date_column']
target_column = config['DEFAULT']['target_column']
req_drop_value_target = config['DEFAULT']['req_drop_value_target']
furnace_signal_column_a = config['DEFAULT']['blast_furnace_signal_a']
furnace_signal_column_b = config['DEFAULT']['blast_furnace_signal_b']
value_A = config['DEFAULT']['req_value_of_blast_furnace_A']
value_B = config['DEFAULT']['req_value_of_blast_furnace_B']
max_best_number = config['DEFAULT']['max_best_number']
correlation_threshold_min_value = config['DEFAULT']['correlation_threshold_min_value']
correlation_threshold_max_value = config['DEFAULT']['correlation_threshold_max_value']
subfolder_feature_vs_target = config['DEFAULT']['subfolder_feature_vs_target']
evaluation_metrics_file_name = config['DEFAULT']['evaluation_metrics_file']
# print(type(correlation_threshold_min_value))
print(correlation_threshold_max_value)
print(correlation_threshold_min_value)
print(file_correlation)
print(matrix_correlation)

# In[6]:


initial_dataframe = create_dataframe(filepath_server)


# In[7]:


def create_dateTime(dataframe):
    dataframe = dataframe.sort_index()
    dataframe['dateTime'] = pd.to_datetime(dataframe['longTime'], unit='ms')
    dataframe = dataframe.drop(['longTime'], axis=1)
    try:
        dataframe.drop(['row ID'], axis = 1)
    except:
        None
    
    return dataframe


df_1 = create_dateTime(initial_dataframe)


# In[8]:


def distinct_month_1(dataframe, target_column,month_key):
    array_df = []
    for i in month_key:
        req_data_1=dataframe.loc[(dataframe[target_column]==i)]
        req_frame_1=pd.DataFrame(req_data_1,columns=dataframe.columns)
        
        array_df.append(req_frame_1)
    
    return array_df


# In[9]:


def draw_month(month_key_value, dict_of_month,target_column):
    for i in month_key_value:
        value = dict_of_month[i]
        value.iloc[:].plot(y=[target_column])
        plt.title('visualization of signal ' + str(target_column) + ' in time of ' + str(i))
        plt.xlabel('range')
        plt.ylabel('value')

        plt.rcParams['figure.figsize'] = (5,5)
        plt.savefig(str(i) + '.jpg')
        plt.show()


# In[10]:


# df_2 = df_1.copy()
# df_2 = df_2.set_index('dateTime')

def create_month(dataframe, target_column_month):
    dataframe = dataframe.set_index('dateTime')
    dataframe['year'] = pd.DatetimeIndex(dataframe.index).year
    dataframe['month'] = pd.DatetimeIndex(dataframe.index).month
    
    dict_of_month = {k:v for k,v in dataframe.groupby('month')}
    month_key_value = collections.OrderedDict(dict_of_month)
    
    month_array_df = distinct_month_1(dataframe, target_column_month, month_key_value)
    
#     draw_month_fig = draw_month(month_key_value, dict_of_month, target_column)
    
    return dataframe, month_array_df

date_df, month_array_df = create_month(df_1, 'month')


# In[13]:


def specific_month_df(dataframe,target_column_month):
    req_data_month=dataframe.loc[(dataframe[target_column_month]==2) | (dataframe[target_column_month]==3) | (dataframe[target_column_month]==4) | (dataframe[target_column_month]==5) ]
    req_frame_month=pd.DataFrame(req_data_month,columns=dataframe.columns)
    
    return req_frame_month

specific_month_df = specific_month_df(date_df,'month')


# In[11]:


def drop_month_year(dataframe):
#     dataframe = dataframe.drop(['year','month'], axis=1)
    dataframe = dataframe.reset_index()
    return dataframe

spec_month = drop_month_year(date_df)
# spec_month = drop_month_year(specific_month_df)
initial_dataframe = None
initial_dataframe.shape
# In[12]:


print(spec_month.shape)


# In[13]:


spec_month.tail(2)


# In[102]:


# df_2.columns.get_loc(target_column)
# print(df_2.columns[4142])
# target_df = df_2.loc[:, df_2.columns[4142]]


# In[103]:


# targ = pd.DataFrame(target_df)


# In[104]:


# df_month = pd.DataFrame(index=df_2.index)


# In[105]:


# df_month['year'] = pd.DatetimeIndex(df_month.index).year


# In[106]:


# df_month['month'] = pd.DatetimeIndex(df_month.index).month


# In[107]:


# df_month


# In[108]:


# df_month_final = pd.concat([df_month, target_df], axis=1)
# df_month_final.head()


# In[109]:


# u = initial_dataframe.columns.get_loc("DEWIHOAT9_I0")
# v = initial_dataframe.columns.get_loc("DEWIHOBT9_I0")
# w = initial_dataframe.columns.get_loc("DEWIHOAT8_I0")
# x = initial_dataframe.columns.get_loc("DEWIHOBT8_I0")
# print(u,v,w,x)


# # plotted blast furnace A and B for Turbine 8's output

# In[110]:


# print(initial_dataframe.shape)
# plt.plot(initial_dataframe["AEWIHO_T8AV2"], color = 'blue')
# plt.plot(initial_dataframe["DEWIHOBT8_I0"], color = 'red')
# plt.plot(initial_dataframe["DEWIHOAT8_I0"], color = 'black')
# plt.legend(["AEWIHO_T8AV2", "DEWIHOBT8_I0","DEWIHOAT8_I0"], loc='upper left')
# # plt.xlim(0,initial_dataframe.shape[0]+10)
# # plt.xticks(np.arange(0,initial_dataframe.shape[0],))
# plt.rcParams['figure.figsize'] = (20, 10)


# # plotted blast furnace A and B for Turbine 9's output

# In[30]:


print(initial_dataframe.shape)
plt.plot(initial_dataframe[target_column], color = 'blue')
plt.plot(initial_dataframe[furnace_signal_column_a], color = 'red')
plt.plot(initial_dataframe[furnace_signal_column_b], color = 'black')
# plt.legend([target_column, furnace_signal_column_a, furnace_signal_column_b], loc='upper left')
# plt.xlim(0,initial_dataframe.shape[0]+10)
# plt.xticks(np.arange(0,initial_dataframe.shape[0],))
plt.rcParams['figure.figsize'] = (10, 5)


# In[13]:


# test_new = conversion_timestamp_to_unixtime(initial_dataframe)


# In[14]:


test_new = spec_month


# In[15]:


# test_new_1 = test_new.drop(['row ID'], axis = 1) # need it if dataset made fro KNIME using ROW_ID
test_new_1 = test_new


# In[16]:


# test_new_2 = remove_rw_column_1(test_new_1,'RW')
test_new_2 = test_new_1


# In[17]:


test_new_2.shape
# test_new_2.shape[0]


# In[18]:



# Taking define number of row from the beginning
# multivariate_data = alter_time(test_new_2, start_pos, test_new.shape[0])
multivariate_data = alter_time(test_new_2, start_pos,test_new_2.shape[0])


# In[19]:


multivariate_data.tail(2)


# In[20]:


index_array=[0,-1]
req_column_name = [date_column, target_column]
# req_column_name = [date_column, furnace_signal_column]
rearranged_dataframe = rearrange_frame(multivariate_data,req_column_name,index_array)


# In[21]:


print(type(multivariate_data), multivariate_data.shape)
print(type(rearranged_dataframe), rearranged_dataframe.shape)


# In[22]:


def check_A_B_blast_furnace_1(dataframe,furnace_signal_column_a,value_A, furnace_signal_column_b,value_B):
    req_data=dataframe.loc[(dataframe[furnace_signal_column_a]>=value_A) | (dataframe[furnace_signal_column_b]>=value_B)]
    req_frame=pd.DataFrame(req_data,columns=dataframe.columns)
    
    return req_frame

dataframe_no_zero_value_blast_furnace = check_A_B_blast_furnace_1(rearranged_dataframe, furnace_signal_column_a, value_A,
                                                               furnace_signal_column_b, value_B)


# In[ ]:


# no_zero_furnace_1 = check_A_B_blast_furnace(rearranged_dataframe, furnace_signal_column_a, value_A,
#                                                                furnace_signal_column_b, value_B)


# In[ ]:


# no_zero_furnace_1.dtypes


# In[ ]:


# print(no_zero_furnace.shape)
# print(type(no_zero_furnace))

# print(no_zero_furnace_1.shape)
# print(type(no_zero_furnace_1))


# rearranged_dataframe.head()


# In[ ]:


# new_frame = pd.concat([first_col, sec_col], axis=1)


# In[ ]:


# for i in rearranged_dataframe:
#     if rearranged_dataframe[i].dtypes == 'object':
# #         rearranged_dataframe.loc[:,i].apply(pd.to_numeric)
        
#         print(i)


# In[ ]:


# for i in rearranged_dataframe:
#     if rearranged_dataframe[i].dtypes == '<M8[ns]' or rearranged_dataframe[i].dtypes == 'object':
# #         rearranged_dataframe.loc[:,i].apply(pd.to_numeric)
        
#         print(i)
    


# In[ ]:


# increase_1=0
# count_array_2 = []
# for i,v in enumerate(rearranged_dataframe[furnace_signal_column_a]):
    
#     if v < 98 and rearranged_dataframe.iloc[i][furnace_signal_column_b] <100:
        
# #         print("i: ",i," ,blast A: ",v," ,blast B: ",rearranged_dataframe.iloc[i][furnace_signal_column_b])
#         count_array_2 = np.append(count_array_2, i)
#         increase_1+=1


# In[23]:


def no_zero_value_in_target_1(dataframe, target_column, req_drop_value_target):
    # req_data_1=dataframe.loc[(dataframe[target_column]!=req_drop_value_target)]
    # req_data_1 = dataframe.loc[(dataframe[target_column] != req_drop_value_target) | (dataframe[target_column]<60)]
    req_data_1 = dataframe.loc[(dataframe[target_column]>=60)]
    req_frame_1=pd.DataFrame(req_data_1,columns=dataframe.columns)
    
    return req_frame_1

dataframe_no_zero_value_target_column = no_zero_value_in_target_1(dataframe_no_zero_value_blast_furnace,target_column, req_drop_value_target)
print(dataframe_no_zero_value_target_column.shape)

import tkinter
# import matplotlib
# matplotlib.get_backend()
import matplotlib
matplotlib.use('TkAgg')
plt.plot(dataframe_no_zero_value_target_column[target_column], color = 'blue')
# plt.show()
# plt.interactive(False)
plt.show(block=False)
plt.ioff()
# plt.interactive(False)
# In[28]:


# plt.plot(dataframe_no_zero_value_blast_furnace[target_column], color = 'blue')
# plt.plot(dataframe_no_zero_value_blast_furnace[furnace_signal_column_a], color = 'red')
# plt.plot(dataframe_no_zero_value_blast_furnace[furnace_signal_column_b], color = 'green')
# # plt.legend([target_column, furnace_signal_column_a, furnace_signal_column_b], loc='upper left')
# # plt.xlim(0,initial_dataframe.shape[0]+10)
# # plt.xticks(np.arange(0,initial_dataframe.shape[0],))
# plt.rcParams['figure.figsize'] = (10, 5)


# In[24]:


dataframe_no_zero_value_target_column.dtypes


# In[25]:


print(rearranged_dataframe.shape)
print(dataframe_no_zero_value_blast_furnace.shape)
print(dataframe_no_zero_value_target_column.shape)


# In[26]:


dataframe_no_zero_value_target_column.tail(2)


# In[27]:


dataframe_no_zero_value_target_column_1 = dataframe_no_zero_value_target_column.reset_index()
dataframe_no_zero_value_target_column_2 = dataframe_no_zero_value_target_column_1.drop(['index'], axis=1)
print(dataframe_no_zero_value_target_column_2.shape)
dataframe_no_zero_value_target_column_2.head(2)


# In[29]:


plt.plot(dataframe_no_zero_value_target_column_2[target_column], color = 'blue')
plt.plot(dataframe_no_zero_value_target_column_2[furnace_signal_column_a], color = 'red')
plt.plot(dataframe_no_zero_value_target_column_2[furnace_signal_column_b], color = 'green')
# plt.legend([target_column, furnace_signal_column_a, furnace_signal_column_b], loc='upper left')
# plt.xlim(0,initial_dataframe.shape[0]+10)
# plt.xticks(np.arange(0,initial_dataframe.shape[0],))
plt.rcParams['figure.figsize'] = (8, 5)
plt.show(block=False)


# In[ ]:


# dataframe_no_zero_value_target_column = no_zero_value_in_target(dataframe_no_zero_value_blast_furnace,target_column, req_drop_value_target)


# In[ ]:


# qq = dataframe_no_zero_value_target_column.apply(pd.to_numeric)


# In[ ]:


# qq['dateTime'].dtypes


# In[ ]:


# drop_object = dataframe_no_zero_value_target_column.select_dtypes(exclude=['object'])


# In[ ]:


# Drop the column which has sam evalue in every ROW
# dataframe_drop_column_with_same_value = drop_column_with_same_value(dataframe_no_zero_value_target_column)


# In[30]:



# Drop the ROW which has NAN value
multivariate_data_drop_nan = drop_nan_value(dataframe_no_zero_value_target_column_2)


# In[ ]:


# nunique = multivariate_data_drop_nan.apply(pd.Series.nunique)
# cols_to_drop = nunique[nunique == 1].index
# dataframe_drop_column_with_same_value = multivariate_data_drop_nan.drop(cols_to_drop, axis=1)

dataframe_drop_column_with_same_value = multivariate_data_drop_nan.drop(multivariate_data_drop_nan.std()[(multivariate_data_drop_nan.std() == 0)].index, axis=1)


# In[119]:


# dataframe_drop_column_with_same_value = drop_column_with_same_value(multivariate_data_drop_nan)


# In[37]:


print(type(dataframe_drop_column_with_same_value))
print(dataframe_drop_column_with_same_value.shape)
dataframe_drop_column_with_same_value.tail(1)


# In[38]:


dataframe_drop_column_with_same_value.dtypes


# In[122]:


# Drop the row who has consecutive same value
# dataframe_drop_row_consecutive_same_value = drop_row(multivariate_data_drop_nan)


# In[123]:


# dataframe_drop_row_consecutive_same_value= multivariate_data_drop_nan


# In[39]:


# Drop the column who has 'objet' type value
dataframe_no_string = drop_string_column(dataframe_drop_column_with_same_value)


# In[40]:


dataframe_no_string.dtypes


# In[46]:


# def remove_rw_column_1(dataframe,req_string):
#     print(req_string)
#     new_variable = []
#     for i in dataframe:
#         x = i[:2]
#         if x != req_string:
#             new_variable = np.append(new_variable, i)
    
#     dataframe = dataframe.iloc[:][new_variable]
#     return dataframe


# In[ ]:





# In[47]:


# no_DE = remove_rw_column_1(dataframe_no_string,'DE')


# In[48]:


# no_CW = remove_rw_column_1(no_DE,'CW')


# In[40]:


# dataframe_drop_row_consecutive_same_value = drop_row(dataframe_no_string)


# In[41]:


print(dataframe_no_string.shape)
# print(no_DE.shape)
# print(no_CW.shape)
# print(dataframe_drop_row_consecutive_same_value.shape)


# In[42]:


plt.plot(dataframe_no_string[target_column], color = 'blue')
# plt.plot(dataframe_no_zero_value_target_column[furnace_signal_column_a], color = 'red')
# plt.plot(dataframe_no_zero_value_target_column[furnace_signal_column_b], color = 'green')
# plt.legend([target_column, furnace_signal_column_a, furnace_signal_column_b], loc='upper left')
# plt.xlim(0,initial_dataframe.shape[0]+10)
# plt.xticks(np.arange(0,initial_dataframe.shape[0],))
plt.rcParams['figure.figsize'] = (8, 5)


# # drop column who has always same value

# In[ ]:


# new_frame = multivariate_data_drop_nan.drop(multivariate_data_drop_nan.std()[(multivariate_data_drop_nan.std() == 0)].index, axis=1)


# In[ ]:


# cols = multivariate_data_drop_nan.select_dtypes([np.number]).columns
# diff = multivariate_data_drop_nan[cols].diff().sum()
# no_same_value_column = multivariate_data_drop_nan.drop(diff[diff == 0].index, axis=1)


# In[ ]:


# print(dataframe_drop_column_with_same_value.shape)
# print(multivariate_data_drop_nan.shape)
# print(dataframe_drop_row_consecutive_same_value.shape)
# print(dataframe_no_string.shape)
# print(new_frame.shape)


# In[43]:


# Make dataframe with dateTime index
def dataframe_datetime(dataframe):
    dataframe_datetime = dataframe.set_index('dateTime')
    return dataframe_datetime

dataframe_datetime = dataframe_datetime(dataframe_no_string)


# In[130]:


# dataframe_datetime = dataframe_datetime.drop(['longTime'], axis = 1)


# In[44]:


dataframe_datetime.shape


# In[45]:


# Feature selection with Sklearn feature best technique
sklearn_feature_best_dataframe = feature_selection_with_selectKbest(dataframe_datetime,max_best_number)


# In[46]:


# feature selection with Pearson Correlation.

sklearn_correlation, main_correlation = pearson_correlation(sklearn_feature_best_dataframe, dataframe_datetime)


# In[49]:


correlation_csv = main_correlation.to_csv('high_correlation_feb_march_april_may.csv')


# In[50]:


dataframe_datetime.to_csv('dataframe_datetime_feb_march_april_may.csv')


# In[47]:


# make a dataframe with signal who is lies between a given range of correlation threshold value
main_frame = dataframe_datetime
correlated_frame = main_correlation

# main_frame = sklearn_feature_best_dataframe
# correlated_frame = sklearn_correlation

main_frame = pd.read_csv(file_correlation)
correlated_frame = pd.read_csv(matrix_correlation)


dataframe_high_correlation = make_dataframe_with_high_correlated_value(main_frame,correlated_frame,
                                                             correlation_threshold_min_value, correlation_threshold_max_value)


# In[48]:


print(dataframe_high_correlation.shape)
dataframe_high_correlation.describe()


# In[ ]:


# multivariate_column_label = ['AEDATZ_HONP0','AEDATZK_ASP0','AEDATZ_HO_V0','AEDAHO_T9_V2','AEWIHO_T9AV2'] #rmse error 1.94


# In[86]:


# for i in dataframe_high_correlation:
#     plt.plot(dataframe_high_correlation[i])
#     plt.title(str(i))
#     plt.show()
    
    
#     # plt.plot(dataframe_high_correlation['AEWIHOBKF_V0'])


# In[119]:


# dataframe_high_correlation_1 = dataframe_high_correlation.drop(['AEWIHOBKF_V0'], axis=1)
# print(dataframe_high_correlation_1.shape)
# plt.plot(dataframe_datetime['DEDAHGGHSZG1'], color = 'blue')


# In[51]:

current_directory = os.getcwd()
print('current_directory is: ',current_directory)
address = 'image_folder'
final_directory = current_directory+'/'+str(address)
if not os.path.exists(final_directory):
    os.makedirs(final_directory)
    print('created : ', final_directory)
else:
    print(final_directory,' has already created')
    pass


# In[52]:


from datetime import date
today = date.today()
print("Today's date:", today)


# In[53]:


dataframe_high_correlation.shape


# In[54]:


print(len(dataframe_high_correlation.columns))
subfolder_1 = 'feature_vs_target'+'_'+str(today)
draw_feature_vs_target = draw_feature_vs_target(dataframe_high_correlation,final_directory,subfolder_1)


# In[55]:


def make_dataset(dataframe, percentage):
    dataset = np.array(dataframe)
    NumberOfElements = int(len(dataset) * percentage)
    print('Number of Elements for training: ', NumberOfElements)
    print('dataset length: ', len(dataset))

    train_input = dataset[0:NumberOfElements, 0:-1]
    print('train_input shape: ', train_input.shape)
    train_output = dataset[0:NumberOfElements, -1]
    print('train_output shape: ', train_output.shape)

    test_input = dataset[NumberOfElements:len(dataset), 0:-1]
    print('test_input shape: ', test_input.shape)
    test_output = dataset[NumberOfElements:len(dataset), -1]
    print('test_output shape: ', test_output.shape)

    return train_input, train_output, test_input, test_output


# In[56]:


train_input, train_output, test_input, test_output = make_dataset(dataframe_high_correlation,0.99)

#s_array = dataframe_high_correlation.values
model_list = [LinearRegression(),linear_model.Lasso(alpha=0.1),linear_model.Ridge(alpha=.5),
              linear_model.BayesianRidge(), tree.DecisionTreeRegressor(max_depth=2),ExtraTreesRegressor(),
              BaggingRegressor(ExtraTreesRegressor()),GBR()]
name = ['LinearRegression','Lasso','Ridge','BayesianRidge','tree','ExtraTreesRegressor','BaggingRegressor','GBR']


# In[57]:


evaluation_metrics_file_path = final_directory+'/'+evaluation_metrics_file_name
if not os.path.isfile(evaluation_metrics_file_path):
    f = open(evaluation_metrics_file_path,'a')
    f.close()
    print('metrics file now created')
else:
    os.remove(evaluation_metrics_file_path)
    f = open(evaluation_metrics_file_path,'a')
    f.close()
    print('metrics file removed and created')


# In[58]:


model = scikit_learn_model(model_list, name, train_input, train_output, test_input, test_output, final_directory, evaluation_metrics_file_path)


# In[ ]:


# rowData = dataframe_datetime.loc[ '2019-08-16 11:20:00' :, : ]
# print(type(rowData))
# print(rowData.shape)
# rowData.head(3)

dataframe_high_correlation.tail()


# In[60]:


df = dataframe_date_time_type(dataframe_high_correlation)
# df = dataframe_date_time_type(temp_frame_1)

dict_of_dates = {k: v for k, v in df.groupby('Date')}
dict_of_day_type = {k:v for k,v in df.groupby('TypeofDAY')}
dict_of_day_name = {k:v for k,v in df.groupby('day_name')}


date_key_value = collections.OrderedDict(dict_of_dates)
day_type_key_value = collections.OrderedDict(dict_of_day_type)
day_name_key_value = collections.OrderedDict(dict_of_day_name)

draw_graph_date = draw_graph(date_key_value,dict_of_dates, target_column,final_directory, subfolder_name = '3_date_fig_target')
draw_graph_week = draw_graph(day_type_key_value,dict_of_day_type, target_column,final_directory, subfolder_name = '3_week_fig_target')
draw_graph_day = draw_graph(day_name_key_value,dict_of_day_name, target_column,final_directory, subfolder_name = '3_day_fig_target')


# In[ ]:





# In[147]:


import sklearn


# In[148]:


my_model = sklearn.tree.ExtraTreeRegressor()
my_model.fit(train_input, train_output)


# In[149]:


my_pred = my_model.predict(test_input)


# In[151]:


plt.plot((min(test_output), max(test_output)), (min(my_pred), max(my_pred)), color='red')
plt.scatter(test_output, my_pred, color='blue')


# In[152]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
print('r_2 statistic: %.2f' % r2_score(test_output,my_pred))
print("Mean_absolute_error: %.2f" % mean_absolute_error(test_output,my_pred))
print("Mean squared error: %.2f" % mean_squared_error(test_output,my_pred))
RMSE=math.sqrt(mean_squared_error(test_output,my_pred))
print('RMSE: ',RMSE)


# # Neural Network

# In[60]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import mean_absolute_error
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


# In[62]:


lr = 0.01

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

batch_size=32
epochs= 30

# define model
def NN_model():
    NN_model = Sequential()
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train_input.shape[1], activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#     NN_model.add(Dense(1, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(1))
    return NN_model
NN_model=NN_model()
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error','accuracy'])
NN_model.summary()


# In[63]:


NN_model.fit(train_input, train_output, epochs=epochs, batch_size=batch_size)


# In[64]:


predicted_output_NN = NN_model.predict(test_input)


# In[65]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
print('r_2 statistic: %.2f' % r2_score(test_output,predicted_output_NN))
print("Mean_absolute_error: %.2f" % mean_absolute_error(test_output,predicted_output_NN))
print("Mean squared error: %.2f" % mean_squared_error(test_output,predicted_output_NN))
RMSE=math.sqrt(mean_squared_error(test_output,predicted_output_NN))
print('RMSE: ',RMSE)


# In[66]:


test_output_NN = np.reshape(test_output,(-1,1))
test_output_NN.shape


# In[67]:


plot_graph(test_output_NN, predicted_output_NN, final_directory,'Neural_Network')


# In[68]:


evaluation_metrices(test_output_NN,predicted_output_NN,final_directory,'Neural Netowrk', evaluation_metrics_file_path)


# In[ ]:





# In[ ]:





# In[ ]:





# # LSTM

# In[69]:


from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import callbacks


# In[70]:


def make_dataset(PandaDataframe):
    dataset = np.array(PandaDataframe)

    NumberOfElements = int(len(dataset)*0.99)
    print('dataset length: ',len(dataset))
    print('Number of Elements for training: ',NumberOfElements)

    multiple_ip_train_data = dataset[0:NumberOfElements]
    multiple_ip_test_set = dataset[NumberOfElements:len(dataset)]
    
    return multiple_ip_train_data, multiple_ip_test_set

multiple_ip_train_data, multiple_ip_test_set = make_dataset(dataframe_high_correlation)

print('train_data_conv shape: ', multiple_ip_train_data.shape)
print('test_data_conv shape: ', multiple_ip_test_set.shape)


# In[71]:


# split a multivariate sequence into samples
def split_sequence(sequence, n_steps):
   X, y = list(), list()
   for i in range(len(sequence)):
       # find the end of this pattern
       end_ix = i + n_steps
       # check if we are beyond the sequence
       if end_ix > len(sequence)-1:
           break
       # gather input and output parts of the pattern
       seq_x, seq_y = sequence[i:end_ix,:-1], sequence[end_ix,-1]
       X.append(seq_x)
       y.append(seq_y)
   return array(X), array(y)

# define input sequence
# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps_1 = 2
# split into samples
X_Conv_Lstm, y_Conv_Lstm = split_sequence(multiple_ip_train_data, n_steps_1)

print('X_Conv_Lstm shape: ',X_Conv_Lstm.shape,'\tX_Conv_Lstm size: ',X_Conv_Lstm.size,'\tX_Conv_Lstm dimension: ',X_Conv_Lstm.ndim)
print('y_Conv_Lstm shape: ', y_Conv_Lstm.shape,' size: ',y_Conv_Lstm.size,' dim: ',y_Conv_Lstm.ndim)


# In[72]:


# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]

samples = X_Conv_Lstm.shape[0]
time = n_steps_1
rows = 1
n_features = X_Conv_Lstm.shape[-1]
cols = n_features
channels = 1

# X_Conv_Lstm_train = X_Conv_Lstm.reshape((X_Conv_Lstm.shape[0], n_seq, n_rows, n_steps_2, n_features))
X_Conv_Lstm_train = X_Conv_Lstm.reshape((samples, time, rows, cols, channels)) #last 2 is feature


# In[73]:


print('X_Conv_Lstm shape: ',X_Conv_Lstm.shape,'\tX_Conv_Lstm size: ',X_Conv_Lstm.size,'\tX_Conv_Lstm dimension: ',X_Conv_Lstm.ndim)
print('X_Conv_Lstm_train shape: ',X_Conv_Lstm_train.shape,'\tX_Conv_Lstm_train size: ',X_Conv_Lstm_train.size,'\tX_Conv_Lstm_train dimension: ',X_Conv_Lstm_train.ndim)


# In[74]:


# define model

# cbks = [callbacks.LearningRateScheduler(lambda x: 1. / (1. + x))]

lr = 0.01

def lr_schedule(epoch):
#     print('epoch value: ', epoch)
    lr_1 = lr * (0.1 ** int(epoch / 10))
#     print('now lr_1: ', lr_1)
    return lr_1

batch_size=32
epochs= 30
activation_function = 'relu'


# In[75]:


model = Sequential()
# n_seq, 1, n_steps_2, n_features
model.add(ConvLSTM2D(filters=64,data_format='channels_last', kernel_size=(1,2), activation=str(activation_function), input_shape=(time,rows,cols,channels),return_sequences=False))
# model.add(ConvLSTM2D(filters=64,data_format='channels_last', kernel_size=(1,2), activation=str(activation_function)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])


# In[76]:


train_model=model.fit(X_Conv_Lstm_train, y_Conv_Lstm, batch_size=batch_size, epochs=epochs, verbose=1,
                      shuffle=True, callbacks=[LearningRateScheduler(lr_schedule)])
#                      ModelCheckpoint('E:/CONV_LSTM_30.h5', save_best_only=True)])


# In[77]:


X_Conv_Lstm_test, y_Conv_Lstm_test = split_sequence(multiple_ip_test_set, n_steps_1)
print('X_Conv_Lstm_test shape: ', X_Conv_Lstm_test.shape,'\t X_Conv_Lstm_test dimension: ', X_Conv_Lstm_test.ndim)
print('y_Conv_Lstm_test shape: ', y_Conv_Lstm_test.shape,'\t y_Conv_Lstm_test dimension: ', y_Conv_Lstm_test.ndim)

test_sample = X_Conv_Lstm_test.shape[0]
# x_input = X_Conv_Lstm_test.reshape((X_Conv_Lstm_test.shape[0], n_seq, 1, n_steps_2, X_Conv_Lstm_test.shape[2]))
x_input = X_Conv_Lstm_test.reshape((test_sample, time, rows, cols, channels))


# In[78]:


from keras.models import load_model

# load_trained_model=load_model('/media/atif/BE0E05910E0543BD/University of Bremen MSc/masters_thesis/forecasting_sensor_data_Salzgitter_AG/trained_model_file/conv_LSTM_norm_100.h5')

yhat = model.predict(x_input, verbose=1)
# print(yhat)


# In[79]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
print('r_2 statistic: %.2f' % r2_score(y_Conv_Lstm_test,yhat))
print("Mean_absolute_error: %.2f" % mean_absolute_error(y_Conv_Lstm_test,yhat))
print("Mean squared error: %.2f" % mean_squared_error(y_Conv_Lstm_test,yhat))
RMSE=math.sqrt(mean_squared_error(y_Conv_Lstm_test,yhat))
print('RMSE: ',RMSE)


# In[80]:


# plt.plot((min(y_Conv_Lstm_test), max(y_Conv_Lstm_test)), (min(yhat), max(yhat)), color='red')
# plt.scatter(y_Conv_Lstm_test, yhat, color='blue')


# In[81]:


y_Conv_Lstm_test_reshape = np.reshape(y_Conv_Lstm_test,(-1,1))
y_Conv_Lstm_test_reshape.shape


# In[82]:


plot_graph(y_Conv_Lstm_test_reshape, yhat, final_directory,'CONV_LSTM')


# In[83]:


evaluation_metrices(y_Conv_Lstm_test_reshape, yhat, final_directory, 'CONV_LSTM',evaluation_metrics_file_path)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


dataframe_datetime.tail()


# In[ ]:


dataframe_datetime.iloc[0:5000].plot(y = dataframe_datetime.columns[-1], use_index=True)
plt.rcParams['figure.figsize'] =(15,5)


# In[ ]:




