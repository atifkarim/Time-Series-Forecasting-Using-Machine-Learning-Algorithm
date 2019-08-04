
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import datetime
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams


# In[ ]:


# test= pd.read_csv('/media/atif/BE0E05910E0543BD/University of Bremen MSc/masters_thesis/IAT_sebastian/dataset_01_Aug_2019.csv')
test=pd.read_csv('E:/University of Bremen MSc/masters_thesis/IAT_sebastian/dataset_26_april_3.csv')


# In[ ]:


longTime=test.loc[0:,['longTime']]
longTime=longTime.as_matrix()
a=[]
date_time_array=[]
for k in longTime:
    a=np.append(a,k)
str_time=[]
correct_longtime=[]
datetime_time=[]
count=0
for b in a:
    b=int(b) # make plain integer
    str_b=str(b)
    c=str_b[-3:]
    new_str_b=str_b.replace(c, '',1)
    new_str_b_time=int(new_str_b)
    correct_longtime.append(new_str_b_time)
    now_time=datetime.datetime.fromtimestamp(new_str_b_time)
    convert_time=now_time.strftime('%Y-%m-%d %H:%M')
    str_time.append(convert_time)
# print(str_time)
test_new = test.assign(stringTime=str_time,correct_longtime=correct_longtime) # here new column in the panda dataframe for string_time has added


# In[ ]:


# It will print the type of value of each column
long_time = test_new.correct_longtime
print(type(long_time))
string_time = test_new.stringTime
print(type(string_time))


# In[ ]:

#making dateTime column as datetime format
test_new['dateTime'] =  pd.to_datetime(test_new['stringTime'], format='%Y-%m-%d %H:%M')
# test_new['dateTime_column'] =  pd.to_datetime(test_new['stringTime'], format='%Y-%m-%d %H:%M')


# In[ ]:


# making dateTime column as a index for the test_new panda dataframe
# test_new = test_new.set_index('dateTime')


# In[ ]:


test_new.head()


# =============================================================================
# # In[ ]:
# 
# 
# # multivariate_column_label=['DEWIHOBT9_I0','AEDATZ_HO_V0','AEDAHO_T9_V2','AEWIHO_T9AV2','RWWIHOB_HWT0'
# #                            ,'RWWIHOB_MWT0']#rmse error 1.99
# 
# # multivariate_column_label = ['DEWIHOBT9_I0','AEDATZ_HONP0','AEDATZK_ASP0','AEDATZ_HO_V0',
# #                              'AEDAHO_T9_V2','AEWIHO_T9AV2'] #rmse error 1.38
# 
# #RWDAKRWRS8V0
# # 'AEDAHO_T8_V2','AEWIHO_T8AV2',
# multivariate_column_label = ['dateTime_column','DEWIHOBT9_I0','AEDATZ_HONP0','AEDATZK_ASP0','AEDATZ_HO_V0',
#                              'AEDATZ_TZCP2','AEDATZKA_8P0','AEDATZ_HO_P1','AEDAHO_T9_V2',
#                              'RWWIHOB_HWT0','RWWIHOB_MWT0','AEWIHO_T9AV2'] #rmse error 1.94
# size_column = len(multivariate_column_label)
# print(size_column)
# 
# print(type(multivariate_column_label))
# 
# 
# # In[ ]:
# 
# 
# size_column = len(multivariate_column_label)
# print(size_column)
# =============================================================================


# In[ ]:


# multivariate_data=test_new.iloc[0:25000][multivariate_column_label]
multivariate_data=test_new.iloc[0:25000][:]
multivariate_data=multivariate_data.loc[::-1]
multivariate_data.head()


# In[ ]:


# function for changing column order. pass dataframe, column name, which order you want to set for the column
def change_column_order(df, col_name, index):
    cols = df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return df[cols]

multivariate_data_1 = change_column_order(multivariate_data,'AEWIHO_T9AV2',-1)
multivariate_data_2 = change_column_order(multivariate_data_1,'dateTime',0)


# In[ ]:


multivariate_data_drop = multivariate_data_2.drop([multivariate_data_2.columns[1],multivariate_data_2.columns[2],
                                                  multivariate_data_2.columns[-3],multivariate_data_2.columns[-2]], axis=1)


# In[ ]:


multivariate_data_drop.columns.get_loc("AEWIHO_T9AV2")


# In[ ]:


multivariate_data_drop.columns[-1]


# In[ ]:


multivariate_data_drop.tail()


# In[ ]:


multivariate_data_drop.loc[4]['AEAGHOAWE2T1']


# In[ ]:


multivariate_data_drop_dummy = multivariate_data_drop.set_index('dateTime')


# In[ ]:


multivariate_data_drop_dummy.head()


# In[ ]:


s.describe()


# In[ ]:


multivariate_data["a"] = pd.to_datetime(multivariate_data["dateTime_column"])


# In[ ]:


# checking column type
s = multivariate_data['dateTime_column'].dtype
print(s)


# In[ ]:


# multivariate_data["a"] = pd.to_datetime(multivariate_data["dateTime_column"])
tr = multivariate_data.drop(multivariate_data.columns[-1], axis=1)


# In[ ]:


tr['Date'] = multivariate_data['dateTime_column'].dt.strftime('%d/%m/%Y')
tr['Time'] = multivariate_data['dateTime_column'].dt.strftime('%H:%M:%S')

date_type = tr['Date'].dtype
time_type = tr['Time'].dtype
print('date_type: ', date_type)
print('time_type: ',time_type)


# In[ ]:


# converting previously created date and time column to datetime type
tr['Date'] = pd.to_datetime(tr['Date'])
tr['Time'] = pd.to_datetime(tr['Time'])

date_type = tr['Date'].dtype
time_type = tr['Time'].dtype

print("to observe the difference look in the previous cell's output")
print('date_type: ', date_type)
print('time_type: ',time_type)


# In[ ]:


e1 = tr['Date'].dtype
print(type(e1))


# In[ ]:


# it will return a column with weekday name
tr['Weekday_name'] = tr.index.weekday_name


# In[ ]:


# It will return a column with number associated with DAY. like monday =0, Tuesday=1 and so on
tr['weekday'] = multivariate_data['dateTime_column'].apply(lambda x: x.weekday())


# In[ ]:


tr['TypeofDAY'] = np.where(multivariate_data['dateTime_column'].dt.dayofweek < 5, 'Weekday', 'weekend') # if the associated number less than 5 then weekend, otherwise weekday
tr['TypeofDAY_number'] = np.where(multivariate_data['dateTime_column'].dt.dayofweek < 5, 1, 0) # 1 for weekday, 0 for weekend


# In[ ]:


# make all the time(without date) to numeric value
tr['numric_time'] = pd.to_timedelta(tr['Time']).dt.total_seconds()


# In[ ]:


tr.head()


# In[ ]:


tr.columns


# In[ ]:


pr_1.head()


# In[ ]:


my_array=[]
count = 0

for index_2, value_2 in tr.iterrows():
    for i_2 , v_2 in enumerate(value_2):
        if i_2 == 12 and v_2 != 0.0:
#             print(v_2)
            v_2_split = v_2.split('/')
            if v_2_split[0] == '11' and v_2_split[1]=='03' :
                my_array = np.append(my_array,index_2)
#                 print(v_2)
#                 print(count)
            
                count+=1
# print('-------',count)
print(len(my_array))


# In[ ]:


tr_33 = tr.reset_index()


# In[ ]:


df2 = pd.DataFrame()
for idx , v_2 in enumerate(tr_33['Date']):
#     print(idx)
    if v_2 != 0.0:
        v_2_split = v_2.split('/')
        if v_2_split[0] == '11' and v_2_split[1]=='03':
            required_dataframe = pd.DataFrame(tr_33.iloc[idx:(idx+1),:], columns=tr_33.columns)
            df2 = pd.concat([df2, required_dataframe], axis=0)


# In[ ]:


df2.head()


# In[ ]:


vvv = tr.groupby('TypeofDAY')
vvv.head(1)

tr['AEWIHO_T9AV2'].describe()


# In[ ]:


dict_of_day_type = {k:v for k,v in tr.groupby('TypeofDAY')}
# plt.ioff()
key_value = collections.OrderedDict(dict_of_day_type)

for k_1 in key_value:
    print(k_1)
    my_week = dict_of_day_type[k_1]
    my_week.iloc[:].plot(y=['AEWIHO_T9AV2'])
    describe = my_week['AEWIHO_T9AV2'].describe()
    RMSE = 2
    textstr = 'describe=%.2f\nRMSE=%.2f\n'%(1, 2)
    plt.text(0.5, 0.5, textstr, fontsize=14)
    plt.savefig(path_1+str(k_1)+'_'+'.jpg')
#     plt.close()


# In[ ]:


dict_of_dates = {k: v for k, v in tr.groupby('Date')}

import collections
prices  = collections.OrderedDict(dict_of_dates)

path_1 = 'E:/University of Bremen MSc/masters_thesis/forecasting_sensor_data_Salzgitter_AG/univariate_data_for_prediction/figure_from_code/graph_of_target_day_wise/'

for k in prices:
    k_sp = k.replace('/','_')
    my_f = dict_of_dates[k]
    my_f.iloc[:].plot(y=['AEWIHO_T9AV2'])
    plt.savefig(path_1+str(k_sp)+'_'+'date.jpg')
    plt.ioff()
#     print(my_f.iloc[:]['AEWIHO_T9AV2'])
    
    


# In[ ]:


dict_of_dates = {k: v for k, v in tr.groupby('Date')}

import collections
prices  = collections.OrderedDict(dict_of_dates)

# path_1 = 'E:/University of Bremen MSc/masters_thesis/forecasting_sensor_data_Salzgitter_AG/univariate_data_for_prediction/figure_from_code/graph_of_target_day_wise/'

for k in prices:
    k_sp = k.replace('/','_')
    print(k,'\t',k_sp)

# import pprint
# pprint.pprint(dict_of_dates)


# In[ ]:


# tr_group = tr.groupby(pd.Grouper(freq='1Y')).sum()
# tr_group.head()

tr_1 = tr['2019-03-11':'2019-03-11']
tr_1.tail()
print(len(tr_1))


# In[ ]:


morning_shift = tr_1.between_time('06:00', '14:00')
day_shift = tr.between_time('14:01', '22:00')
night_shift = tr.between_time('22:01', '05:59')


# In[ ]:


len(morning_shift)


# In[ ]:


def change_dataframe(dataframe):
    present_data = dataframe
    return present_data
present_data = change_dataframe(tr)


# In[ ]:


type(pr_1)


# In[ ]:


# present_data.plot(y=present_data.columns['AEWIHO_T9AV2'])


# In[ ]:


pr_1 = present_data.drop(multivariate_column_label[0], axis=1)


# In[ ]:


pr_1.head()


# In[ ]:


zero_index=[]
zero_index=np.array(zero_index)
for index, value in pr_1.iterrows():
    for i , v in enumerate(value):
        if i == 0  and v == 0.0:
#             print('index: ',index)
            zero_index=np.append(zero_index,index)

print('size of zero_index: ',zero_index.size)


# In[ ]:


if zero_index.size > 0:
    pr_1_modify = pr_1.drop(zero_index,axis=0) # axis= 0 means row indiated. 1 means column indicated
else:
    pr_1_modify = pr_1


# In[ ]:


cols = [-1,-2,-3,-4,-5,-6,-7]
pr_1_modify_drop = pr_1_modify.drop(pr_1_modify.columns[cols], axis=1)


# In[ ]:


pr_1_modify_drop.head()


# In[ ]:


multivariate_data_modify = pr_1_modify_drop

from numpy import nan
multivariate_data_drop_zero= multivariate_data_modify.replace(0,nan)
multivariate_data_drop_zero=multivariate_data_drop_zero.dropna(how='any',axis=0)


# In[ ]:


print(multivariate_data_modify.shape)
print(multivariate_data_drop_zero.shape)


# In[ ]:


multivariate_data_drop_zero_resample = multivariate_data_drop_zero.resample('1min').mean()
multivariate_data_drop_zero_interpolate = multivariate_data_drop_zero_resample.interpolate('linear')

print('shape of multivariate_data_drop_zero_resample: ', multivariate_data_drop_zero_resample.shape)
print('shape of multivariate_data_drop_zero_interpolate: ',multivariate_data_drop_zero_interpolate.shape)


# In[ ]:


# print(len(multivariate_column_label))
for idx, i in enumerate(multivariate_column_label):
    now_idx = idx+2
    
    if now_idx == len(multivariate_column_label):
        print('terminate')
        break
    print(multivariate_column_label[now_idx])
    drop_consecutive_same_value_zero = multivariate_data_drop_zero.loc[multivariate_data_drop_zero[multivariate_column_label[now_idx]].shift() != multivariate_data_drop_zero[multivariate_column_label[now_idx]]]
    drop_consecutive_same_value_interpolate = multivariate_data_drop_zero_interpolate.loc[multivariate_data_drop_zero_interpolate[multivariate_column_label[now_idx]].shift() != multivariate_data_drop_zero_interpolate[multivariate_column_label[now_idx]]]
    


# In[ ]:


print('shape of drop_consecutive_same_value_zero: ', drop_consecutive_same_value_zero.shape)
print('shape of drop_consecutive_same_value_interpolate: ', drop_consecutive_same_value_interpolate.shape)


# In[ ]:


new_dataframe = drop_consecutive_same_value_zero.drop(['DEWIHOBT9_I0'], axis=1)
# new_dataframe = drop_consecutive_same_value_interpolate.drop(['DEWIHOBT9_I0'], axis=1)


# In[ ]:


new_dataframe.head()


# In[ ]:


# start = 0
# end = 300

# loop = int(len(new_dataframe)/300)+1
# path = 'E:/University of Bremen MSc/masters_thesis/forecasting_sensor_data_Salzgitter_AG/univariate_data_for_prediction/figure_from_code/fig_target_night/'

# for i in range(loop):
#     if end < len(new_dataframe):
        
#         new_dataframe.iloc[start:end].plot(y=new_dataframe.columns[-1])
        
#         plt.savefig(path+str(start)+'_night.jpg')
#         start = end
#         end = end+300
#     else:
#         start = start
#         end = len(new_dataframe)
#         new_dataframe.iloc[start:end].plot(y=new_dataframe.columns[-1])
#         plt.savefig(path+'final_'+str(len(new_dataframe))+'_night.jpg')
        
# #     plt.rcParams['figure.figsize'] =(20,10)


# In[ ]:


new_dataframe.iloc[0:].plot(y = new_dataframe.columns[-1], use_index=True)
plt.rcParams['figure.figsize'] =(20,20)


# In[ ]:


dateRange = pd.date_range(new_dataframe.index[0],new_dataframe.index[10], freq='1min')
print(dateRange)
plt.plot(dateRange,new_dataframe.iloc[0:11,-1])
plt.xlim(dateRange[0],dateRange[-1])
plt.xticks(rotation=25)


# In[ ]:


dateRange[-1]


# In[ ]:


# start_1 = 700
# end_1 = start_1+300
# new_dataframe.iloc[start_1:end_1].plot(x = new_dataframe.index.format(), y=new_dataframe.columns[-1])
plt.plot(new_dataframe.iloc[0:100].index.format(), new_dataframe.iloc[0:100,-1])
plt.show()


# In[ ]:


print(max(new_dataframe.iloc[start_1:end_1][new_dataframe.columns[-1]]))

print(len(new_dataframe))


# In[ ]:


test_new.iloc[0:2000].plot(y=new_dataframe.columns[-1])


# In[ ]:


new_dataframe.plot(y=new_dataframe.columns[-1])


# In[ ]:


dataset = np.array(multivariate_data_drop_dummy)


# In[ ]:


def make_dataset(dataset):
    NumberOfElements=int(len(dataset)*0.95)
    print('Number of Elements for training: ',NumberOfElements)
    print('dataset length: ',len(dataset))

    train_input=dataset[0:NumberOfElements,0:-1]
    print('train_input shape: ',train_input.shape)
    train_output=dataset[0:NumberOfElements,-1]
    print('train_output shape: ',train_output.shape)

    test_input=dataset[NumberOfElements:len(dataset),0:-1]
    print('test_input shape: ',test_input.shape)
    test_output=dataset[NumberOfElements:len(dataset),-1]
    print('test_output shape: ',test_output.shape)
    
    return train_input, train_output, test_input, test_output

train_input, train_output, test_input, test_output = make_dataset(dataset)


# In[ ]:


from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[ ]:


train_model=LinearRegression(fit_intercept = True,normalize=False).fit(train_input,train_output)
print(train_model)

predicted_output=train_model.predict(test_input)


# In[ ]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math


# In[ ]:


print('Slope:' ,train_model.coef_)
print('Intercept:', train_model.intercept_)
print('r_2 statistic: %.2f' % r2_score(test_output,predicted_output))
print("Mean_absolute_error: %.2f" % mean_absolute_error(test_output,predicted_output))
print("Mean squared error: %.2f" % mean_squared_error(test_output,predicted_output))
RMSE=math.sqrt(mean_squared_error(test_output,predicted_output))
print('RMSE: ',RMSE)


# In[ ]:


plt.plot((min(test_output),max(test_output)), (min(predicted_output),max(predicted_output)))
plt.scatter(test_output,predicted_output, color = 'blue')


# In[ ]:


correlation = multivariate_data_drop_dummy.corr()


# In[ ]:


correlation.shape


# In[ ]:


# to display correlation dataframe
# correlation


# In[ ]:


# changing the index value to numerical value
correlation_reset = correlation.reset_index()


# In[ ]:


# correlation_reset


# In[ ]:


high_correlated_array_with_target = []
for index_corr_reset, val_corr_reset in enumerate(correlation_reset['AEWIHO_T9AV2']):
    if val_corr_reset>0.4:
        required_column = correlation_reset.loc[index_corr_reset]['index']
        high_correlated_array_with_target = np.append(high_correlated_array_with_target,required_column)


# In[ ]:


high_correlated_array_with_target


# In[ ]:


# making new dataframe using the column which came from the high correlated array

print(type(high_correlated_array_with_target))
print('shape: ',high_correlated_array_with_target.shape)

new_frame = multivariate_data_drop_dummy.iloc[:][high_correlated_array_with_target]


# In[ ]:


print('length of new_frame: ', len(new_frame))
new_frame.head()


# In[ ]:


new_frame_1 = new_frame.drop(['RWWIHOAG9_V0','RWWIHOBG8_V0'], axis=1)


# In[ ]:


dataset = np.array(new_frame)

def make_dataset(dataset):
    NumberOfElements=int(len(dataset)*0.98)
    print('Number of Elements for training: ',NumberOfElements)
    print('dataset length: ',len(dataset))

    train_input=dataset[0:NumberOfElements,0:-1]
    print('train_input shape: ',train_input.shape)
    train_output=dataset[0:NumberOfElements,-1]
    print('train_output shape: ',train_output.shape)

    test_input=dataset[NumberOfElements:len(dataset),0:-1]
    print('test_input shape: ',test_input.shape)
    test_output=dataset[NumberOfElements:len(dataset),-1]
    print('test_output shape: ',test_output.shape)
    
    return train_input, train_output, test_input, test_output

train_input, train_output, test_input, test_output = make_dataset(dataset)
# print('train data size: ',train_data.shape,'\ntest data size: ',test_data.shape)


# In[ ]:


from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import *
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor as GBR

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math


# In[ ]:


train_model = LinearRegression(fit_intercept = True,normalize=False).fit(train_input,train_output)
#train_model = ExtraTreesRegressor(n_estimators=10000, random_state=0)
train_model.fit(train_input, train_output)
print(train_model)


# In[ ]:


predicted_output=train_model.predict(test_input)


# In[ ]:


# print('Slope:' ,train_model.coef_)
# print('Intercept:', train_model.intercept_)
print('r_2 statistic: %.2f' % r2_score(test_output,predicted_output))
print("Mean_absolute_error: %.2f" % mean_absolute_error(test_output,predicted_output))
print("Mean squared error: %.2f" % mean_squared_error(test_output,predicted_output))
RMSE=math.sqrt(mean_squared_error(test_output,predicted_output))
print('RMSE: ',RMSE)


# In[ ]:


plt.plot((min(test_output),max(test_output)), (min(predicted_output),max(predicted_output)), color='red')
plt.scatter(test_output,predicted_output, color = 'blue')

