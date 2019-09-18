# import numpy as np
# a = np.zeros((156816, 36, 53806), dtype='uint8')
import json
#import collections
import os
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn import linear_model
from sklearn import tree

from dataset_analysis import create_dataframe, create_dateTime, create_month
from dataset_analysis import specific_month_df, drop_month_year, alter_time, rearrange_frame
from dataset_analysis import check_A_B_blast_furnace_1, no_zero_value_in_target_1, dataframe_reset_index
from dataset_analysis import drop_nan_value, drop_unique_valued_columns, drop_string_column, dataframe_datetime

from dataset_analysis import feature_selection_with_selectKbest
from dataset_analysis import pearson_correlation
from dataset_analysis import make_dataframe_with_high_correlated_value
# from dataset_analysis import dataframe_date_time_type
from dataset_analysis import draw_graph
# from dataset_analysis import draw_feature_vs_target

from model_file import make_dataset, scikit_learn_model
from model_file import plot_graph, evaluation_metrices
from model_file import NN_model
from model_file import make_dataset_LSTM, split_sequence, LSTM_model

with open('variable_config.json', 'r') as f:
    config = json.load(f)

filepath = config['DEFAULT']['file_path']
filepath_ubuntu = config['DEFAULT']['file_path_ubuntu']
filepath_ubuntu_1 = config['DEFAULT']['file_path_ubuntu_1']
filepath_server = config['DEFAULT']['file_path_server']
file_windows_feb_march = config['DEFAULT']['file_windows_feb_march']
file_windows_april_may = config['DEFAULT']['file_windows_april_may']
file_windows_june_july = config['DEFAULT']['file_windows_june_july']

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
required_number_of_test_data = config['DEFAULT']['required_number_of_test_data']
subfolder_feature_vs_target = config['DEFAULT']['subfolder_feature_vs_target']
evaluation_metrics_file_name = config['DEFAULT']['evaluation_metrics_file']
number_of_step_lstm = config['DEFAULT']['n_steps_lstm']
epochs = config['DEFAULT']['epochs']
batch_size = config['DEFAULT']['batch_size']

print(correlation_threshold_min_value)

#reading csv file
initial_dataframe = create_dataframe(file_windows_april_may)

df_1 = create_dateTime(initial_dataframe,'row ID','Unnamed: 0')

date_df, month_array_df = create_month(df_1, 'month')

specific_month_df = specific_month_df(date_df,'month')

spec_month = drop_month_year(specific_month_df)

# print(initial_dataframe.shape)
# plt.plot(initial_dataframe[target_column], color = 'blue')
# plt.plot(initial_dataframe[furnace_signal_column_a], color = 'red')
# plt.plot(initial_dataframe[furnace_signal_column_b], color = 'black')
# # plt.legend([target_column, furnace_signal_column_a, furnace_signal_column_b], loc='upper left')
# # plt.xlim(0,initial_dataframe.shape[0]+10)
# # plt.xticks(np.arange(0,initial_dataframe.shape[0],))
# plt.rcParams['figure.figsize'] = (10, 5)


# Taking define number of row from the beginning
# multivariate_data = alter_time(test_new_2, start_pos, test_new.shape[0])
multivariate_data = alter_time(spec_month, start_pos, spec_month.shape[0])
multivariate_data.tail(2)

index_array=[0,-1]
req_column_name = [date_column, target_column]
# req_column_name = [date_column, furnace_signal_column]
rearranged_dataframe = rearrange_frame(multivariate_data,req_column_name,index_array)

dataframe_no_zero_value_blast_furnace = check_A_B_blast_furnace_1(rearranged_dataframe, furnace_signal_column_a, value_A,
                                                                  furnace_signal_column_b, value_B)

dataframe_no_zero_value_target_column = no_zero_value_in_target_1(dataframe_no_zero_value_blast_furnace,target_column, req_drop_value_target)


# import tkinter
# # import matplotlib
# # matplotlib.get_backend()
# import matplotlib
# matplotlib.use('TkAgg')
# plt.plot(dataframe_no_zero_value_target_column[target_column], color = 'blue')
# # plt.show()
# # plt.interactive(False)
# plt.show(block=False)
# plt.ioff()
# # plt.interactive(False)


# plt.plot(dataframe_no_zero_value_blast_furnace[target_column], color = 'blue')
# plt.plot(dataframe_no_zero_value_blast_furnace[furnace_signal_column_a], color = 'red')
# plt.plot(dataframe_no_zero_value_blast_furnace[furnace_signal_column_b], color = 'green')
# # plt.legend([target_column, furnace_signal_column_a, furnace_signal_column_b], loc='upper left')
# # plt.xlim(0,initial_dataframe.shape[0]+10)
# # plt.xticks(np.arange(0,initial_dataframe.shape[0],))
# plt.rcParams['figure.figsize'] = (10, 5)

dataframe_no_zero_value_target_column_2 = dataframe_reset_index(dataframe_no_zero_value_target_column)
print(dataframe_no_zero_value_target_column_2.shape)

plt.plot(dataframe_no_zero_value_target_column_2[target_column], color = 'blue')
plt.plot(dataframe_no_zero_value_target_column_2[furnace_signal_column_a], color = 'red')
plt.plot(dataframe_no_zero_value_target_column_2[furnace_signal_column_b], color = 'green')
# plt.legend([target_column, furnace_signal_column_a, furnace_signal_column_b], loc='upper left')
plt.xlim(0,dataframe_no_zero_value_target_column_2.shape[0]+10)
# plt.xticks(np.arange(0,initial_dataframe.shape[0],))
plt.rcParams['figure.figsize'] = (8, 5)
plt.show(block=False)

# qq = dataframe_no_zero_value_target_column.apply(pd.to_numeric)
# drop_object = dataframe_no_zero_value_target_column.select_dtypes(exclude=['object'])

# Drop the column which has same value in every ROW
# dataframe_drop_column_with_same_value = drop_column_with_same_value(dataframe_no_zero_value_target_column)

# Drop the ROW which has NAN value
multivariate_data_drop_nan = drop_nan_value(dataframe_no_zero_value_target_column_2)

dataframe_drop_column_with_same_value = drop_unique_valued_columns(multivariate_data_drop_nan)

# dataframe_drop_column_with_same_value = multivariate_data_drop_nan.drop(multivariate_data_drop_nan.std()[(multivariate_data_drop_nan.std() == 0)].index, axis=1)
# dataframe_drop_column_with_same_value = drop_column_with_same_value(multivariate_data_drop_nan)

print(dataframe_drop_column_with_same_value.dtypes)

# Drop the column who has 'objet' type value
dataframe_no_string = drop_string_column(dataframe_drop_column_with_same_value)
dataframe_no_string.dtypes

plt.plot(dataframe_no_string[target_column], color = 'blue')
# plt.plot(dataframe_no_zero_value_target_column[furnace_signal_column_a], color = 'red')
# plt.plot(dataframe_no_zero_value_target_column[furnace_signal_column_b], color = 'green')
# plt.legend([target_column, furnace_signal_column_a, furnace_signal_column_b], loc='upper left')
# plt.xlim(0,initial_dataframe.shape[0]+10)
# plt.xticks(np.arange(0,initial_dataframe.shape[0],))
plt.rcParams['figure.figsize'] = (8, 5)


initial_dataframe = None
date_df = None
specific_month_df = None
spec_month = None
rearranged_dataframe = None
multivariate_data = None
dataframe_no_zero_value_blast_furnace = None
dataframe_no_zero_value_target_column = None
multivariate_data_drop_nan = None
dataframe_drop_column_with_same_value = None


# Make dataframe with dateTime index
dataframe_datetime = dataframe_datetime(dataframe_no_string)
print(dataframe_datetime.shape)



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

print(dataframe_high_correlation.shape)
dataframe_high_correlation.describe()


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

from datetime import date
today = date.today()
print("Today's date:", today)


# print(len(dataframe_high_correlation.columns))
# subfolder_1 = 'feature_vs_target'+'_'+str(today)
# draw_feature_vs_target = draw_feature_vs_target(dataframe_high_correlation,final_directory,subfolder_1)

train_input, train_output, test_input, test_output = make_dataset(dataframe_high_correlation,required_number_of_test_data)

#s_array = dataframe_high_correlation.values
model_list = [LinearRegression(),linear_model.Lasso(alpha=0.1),linear_model.Ridge(alpha=.5),
              linear_model.BayesianRidge(), tree.DecisionTreeRegressor(max_depth=2),ExtraTreesRegressor(),
              BaggingRegressor(ExtraTreesRegressor()),GBR()]
name = ['LinearRegression','Lasso','Ridge','BayesianRidge','tree','ExtraTreesRegressor','BaggingRegressor','GBR']

# code for creating text file storing for evaluation metrics
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

model = scikit_learn_model(model_list, name, train_input, train_output, test_input, test_output, final_directory, evaluation_metrics_file_path)

# df = dataframe_date_time_type(dataframe_high_correlation)
# # df = dataframe_date_time_type(temp_frame_1)
#
# dict_of_dates = {k: v for k, v in df.groupby('Date')}
# dict_of_day_type = {k:v for k,v in df.groupby('TypeofDAY')}
# dict_of_day_name = {k:v for k,v in df.groupby('day_name')}
#
#
# date_key_value = collections.OrderedDict(dict_of_dates)
# day_type_key_value = collections.OrderedDict(dict_of_day_type)
# day_name_key_value = collections.OrderedDict(dict_of_day_name)
#
# draw_graph_date = draw_graph(date_key_value,dict_of_dates, target_column,final_directory, subfolder_name = '3_date_fig_target')
# draw_graph_week = draw_graph(day_type_key_value,dict_of_day_type, target_column,final_directory, subfolder_name = '3_week_fig_target')
# draw_graph_day = draw_graph(day_name_key_value,dict_of_day_name, target_column,final_directory, subfolder_name = '3_day_fig_target')

#train_output_cv = np.reshape(train_output,(-1,1))
from sklearn.model_selection import cross_val_score

for index, value in enumerate(model_list):
    
    scores_r2 = cross_val_score(value,train_input,train_output,cv=10, scoring='r2')
    scores = cross_val_score(value,train_input,train_output,cv=10, scoring='neg_mean_squared_error')
    mse_scores = -scores
    rmse_scores = np.sqrt(mse_scores)
    print(name[index],'--'*5,scores_r2.mean())
    print(name[index],'--'*5,rmse_scores.mean())
    f = open(evaluation_metrics_file_path, 'a')
    f.write(str(name[index])+'\t'+'RMSE: '+str(rmse_scores.mean())+'\n')
    f.write(str(name[index])+'\t'+'r_2 square: '+str(scores_r2.mean())+'\n')
    f.write('\n')
    f.close()

#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import KFold
#kf = KFold(n_splits = 5, shuffle = True)
##rf_reg = RandomForestRegressor()
#scores = []
#for i in range(5):
#    result = next(kf.split(train_input), None)
#    print(result)
#    x_train = train_input[result[0]]
#    print(x_train)
#    x_test = train_input[result[1]]
#    y_train = train_output[result[0]]
#    y_test = train_output[result[1]]
#    model = my_model.fit(x_train,y_train)
#    predictions = my_model.predict(x_test)
#    np.append(scores,model.score(x_test,y_test))
#print('Scores from each Iteration: ', scores)
#print('Average K-Fold Score :' , np.mean(scores))


#my_model.fit(train_input, train_output)
#my_pred = my_model.predict(test_input)
#plt.plot((min(test_output), max(test_output)), (min(my_pred), max(my_pred)), color='red')
#plt.scatter(test_output, my_pred, color='blue')

# from sklearn import linear_model
#from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#import math
#print('r_2 statistic: %.2f' % r2_score(test_output,my_pred))
#print("Mean_absolute_error: %.2f" % mean_absolute_error(test_output,my_pred))
#print("Mean squared error: %.2f" % mean_squared_error(test_output,my_pred))
#RMSE=math.sqrt(mean_squared_error(test_output,my_pred))
#print('RMSE: ',RMSE)


# # Neural Network
#lr = 0.01
#
#def lr_schedule(epoch):
#    return lr * (0.1 ** int(epoch / 10))

batch_size = batch_size
epochs= epochs

NN_model=NN_model(train_input)
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error','accuracy'])
NN_model.summary()
NN_model.fit(train_input, train_output, epochs=epochs, batch_size=batch_size)
predicted_output_NN = NN_model.predict(test_input)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
print('r_2 statistic: %.2f' % r2_score(test_output,predicted_output_NN))
print("Mean_absolute_error: %.2f" % mean_absolute_error(test_output,predicted_output_NN))
print("Mean squared error: %.2f" % mean_squared_error(test_output,predicted_output_NN))
RMSE=math.sqrt(mean_squared_error(test_output,predicted_output_NN))
print('RMSE: ',RMSE)


test_output_NN = np.reshape(test_output,(-1,1))
test_output_NN.shape

plot_graph(test_output_NN, predicted_output_NN, final_directory,'Neural_Network')
evaluation_metrices(test_output_NN,predicted_output_NN,final_directory,'Neural Netowrk', evaluation_metrics_file_path)

# # LSTM


#from numpy import array
#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers import Dense, Dropout
#from keras.layers import Flatten
#from keras.layers import ConvLSTM2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
#from keras import callbacks

multiple_ip_train_data, multiple_ip_test_set = make_dataset_LSTM(dataframe_high_correlation, required_number_of_test_data)

# split into samples
X_Conv_Lstm, y_Conv_Lstm = split_sequence(multiple_ip_train_data, number_of_step_lstm)

print('X_Conv_Lstm shape: ',X_Conv_Lstm.shape,'\tX_Conv_Lstm size: ',X_Conv_Lstm.size,'\tX_Conv_Lstm dimension: ',X_Conv_Lstm.ndim)
print('y_Conv_Lstm shape: ', y_Conv_Lstm.shape,' size: ',y_Conv_Lstm.size,' dim: ',y_Conv_Lstm.ndim)

# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]

samples = X_Conv_Lstm.shape[0]
time = number_of_step_lstm
rows = 1
n_features = X_Conv_Lstm.shape[-1]
cols = n_features
channels = 1

# X_Conv_Lstm_train = X_Conv_Lstm.reshape((X_Conv_Lstm.shape[0], n_seq, n_rows, n_steps_2, n_features))
X_Conv_Lstm_train = X_Conv_Lstm.reshape((samples, time, rows, cols, channels)) #last 2 is feature

print('X_Conv_Lstm shape: ',X_Conv_Lstm.shape,'\tX_Conv_Lstm size: ',X_Conv_Lstm.size,'\tX_Conv_Lstm dimension: ',X_Conv_Lstm.ndim)
print('X_Conv_Lstm_train shape: ',X_Conv_Lstm_train.shape,'\tX_Conv_Lstm_train size: ',X_Conv_Lstm_train.size,'\tX_Conv_Lstm_train dimension: ',X_Conv_Lstm_train.ndim)

# define model

# cbks = [callbacks.LearningRateScheduler(lambda x: 1. / (1. + x))]

lr = 0.01

def lr_schedule(epoch):
    print('epoch value: ', epoch)
    lr_1 = lr * (0.1 ** int(epoch / 10))
    print('now lr_1: ', lr_1)
    return lr_1

batch_size = batch_size
epochs = 4
activation_function = 'relu'


# In[75]:


model = LSTM_model(activation_function, time, rows, cols, channels)
## n_seq, 1, n_steps_2, n_features
#model.add(ConvLSTM2D(filters=64,data_format='channels_last', kernel_size=(1,2), activation=str(activation_function), input_shape=(time,rows,cols,channels),return_sequences=False))
## model.add(ConvLSTM2D(filters=64,data_format='channels_last', kernel_size=(1,2), activation=str(activation_function)))
#model.add(Flatten())
#model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

train_model=model.fit(X_Conv_Lstm_train, y_Conv_Lstm, batch_size=batch_size, epochs=epochs, verbose=1,
                      shuffle=True, callbacks=[LearningRateScheduler(lr_schedule)])
#                      ModelCheckpoint('E:/CONV_LSTM_30.h5', save_best_only=True)])


# In[77]:


X_Conv_Lstm_test, y_Conv_Lstm_test = split_sequence(multiple_ip_test_set, number_of_step_lstm)
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

dataframe_datetime.iloc[0:5000].plot(y = dataframe_datetime.columns[-1], use_index=True)
plt.rcParams['figure.figsize'] =(15,5)
