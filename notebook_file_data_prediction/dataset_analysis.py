import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from numpy import nan
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import collections
import os
import shutil
# get_ipython().run_line_magic('matplotlib', 'inline')
#from matplotlib.pylab import rcParams

# # function to read the csv file

def create_dataframe(filepath):
    test = pd.read_csv(filepath)  # here the given csv file is reading

    return test


# create datetime and drop rowID column if exists

def create_dateTime(dataframe, col_a, col_b):
    dataframe = dataframe.sort_index()
    dataframe['dateTime'] = pd.to_datetime(dataframe['longTime'], unit='ms')
    dataframe = dataframe.drop(['longTime'], axis=1)
    try:
        dataframe = dataframe.drop([col_b], axis=1)
    except:
        None

    try:
        dataframe = dataframe.drop([col_a], axis=1)
    except:
        None

    return dataframe

# function for converting timestamp to unixtime and return the ready dataframe

def conversion_timestamp_to_unixtime(initial_dataframe):
    ''' now conversion of timestamp to unixtime will start. In the csv file the column name of
    timestamp is longtime.'''

    longTime = initial_dataframe.loc[0:, ['longTime']]
    longTime = longTime.as_matrix()
    a = []
    date_time_array = []
    for k in longTime:
        a = np.append(a, k)
    str_time = []
    correct_longtime = []
    datetime_time = []
    count = 0

    for b in a:
        b = int(b)  # make plain integer
        str_b = str(b)
        c = str_b[-3:]
        new_str_b = str_b.replace(c, '', 1)
        new_str_b_time = int(new_str_b)
        correct_longtime.append(new_str_b_time)
        now_time = datetime.datetime.fromtimestamp(new_str_b_time)
        convert_time = now_time.strftime('%Y-%m-%d %H:%M')
        str_time.append(convert_time)
    test_new = initial_dataframe.assign(stringTime=str_time,
                                        correct_longtime=correct_longtime)  # here new column in the panda dataframe for string_time has added
    test_new['dateTime'] = pd.to_datetime(test_new['stringTime'], format='%Y-%m-%d %H:%M')
    test_new = test_new.drop(['longTime', 'stringTime', 'correct_longtime'], axis=1)

    return test_new


# # be careful here , when perform on a dataframe reset_index then a new column will appear and it is 'index'. No need of it so immediately drop it. for better view please take a look in the previous cell


# create month, year column to observe dataset from a different point of view.

def distinct_month_1(dataframe, target_column, month_key):
    array_df = []
    for i in month_key:
        req_data_1 = dataframe.loc[(dataframe[target_column] == i)]
        req_frame_1 = pd.DataFrame(req_data_1, columns=dataframe.columns)

        array_df.append(req_frame_1)

    return array_df


def draw_month(month_key_value, dict_of_month, target_column):
    for i in month_key_value:
        value = dict_of_month[i]
        value.iloc[:].plot(y=[target_column])
        plt.title('visualization of signal ' + str(target_column) + ' in time of ' + str(i))
        plt.xlabel('range')
        plt.ylabel('value')

        plt.rcParams['figure.figsize'] = (5, 5)
        plt.savefig(str(i) + '.jpg')
        plt.show()


def create_month(dataframe, target_column_month):
    dataframe = dataframe.set_index('dateTime')
    dataframe['year'] = pd.DatetimeIndex(dataframe.index).year
    dataframe['month'] = pd.DatetimeIndex(dataframe.index).month

    dict_of_month = {k: v for k, v in dataframe.groupby('month')}
    month_key_value = collections.OrderedDict(dict_of_month)

    month_array_df = distinct_month_1(dataframe, target_column_month, month_key_value)

    #     draw_month_fig = draw_month(month_key_value, dict_of_month, target_column)

    return dataframe, month_array_df


def specific_month_df(dataframe, target_column_month):
    req_data_month = dataframe.loc[(dataframe[target_column_month] == 4) | (dataframe[target_column_month] == 5)]
    #     req_data_month=dataframe.loc[(dataframe[target_column_month]==2|3) ]
    req_frame_month = pd.DataFrame(req_data_month, columns=dataframe.columns)

    return req_frame_month


def drop_month_year(dataframe):
    #     dataframe = dataframe.drop(['year','month'], axis=1)
    dataframe = dataframe.reset_index()
    return dataframe


# def remove_rw_column(dataframe):
#     new_variable = []
#     for i in dataframe:
#         x = i[:2]
#         if x != 'RW':
#             new_variable = np.append(new_variable, i)
#     return new_variable
#
#
# def remove_rw_column_1(dataframe,req_string):
#     new_variable = []
#     for i in dataframe:
#         x = i[:2]
#         if x != req_string:
#             new_variable = np.append(new_variable, i)
#
#     dataframe = dataframe.iloc[:][new_variable]
#     return dataframe


def alter_time(dataframe, start_pos, end_pos):
    #     multivariate_data=test_new.iloc[start_pos:end_pos][multivariate_column_label] # comment out this line if you pass column label
    dataframe = dataframe.iloc[start_pos:end_pos][:]
    dataframe = dataframe.loc[::-1]

    return dataframe


# # Now target column and dateTime colum will be arranged as a given column index. Here target column is the output of turbine 9's output


def rearrange_frame(dataframe, colname, col_pos):
    list_col = dataframe.columns.to_list()
    temp_list = list_col
    for idx, i in enumerate(colname):
        sacrifice_val = temp_list[col_pos[idx]]
        indx = dataframe.columns.get_loc(i)
        temp_list[col_pos[idx]] = i
        temp_list[indx] = sacrifice_val

    return dataframe.iloc[:][temp_list]


# # Now take in consideration the signal DEWIHOBT9_I0. When the value of it's will be 100 only then target column will work otherwise not. So, choose this signal and drop all of the rows where it's value != 100 and then drop the whole colum as after dropping this column will only contain value 100 and it will affect negatively in the correlation with target signal

# the function will do the following task
# if the blast furnace signal for turbine 9 is zero then no work will be happened.
# so, remove all the rows where this value will be zero

def check_A_B_blast_furnace_1(dataframe,furnace_signal_column_a,value_A, furnace_signal_column_b,value_B):
    req_data=dataframe.loc[(dataframe[furnace_signal_column_a]>=value_A) | (dataframe[furnace_signal_column_b]>=value_B)]
    req_frame=pd.DataFrame(req_data,columns=dataframe.columns)
    
    return req_frame


# def check_A_B_blast_furnace(dataframe,furnace_signal_column_a,value_A, furnace_signal_column_b,value_B):
#     req_data=dataframe.loc[(dataframe[furnace_signal_column_a]>=value_A) | (dataframe[furnace_signal_column_b]>=value_B)].values
#     req_frame=pd.DataFrame(req_data,columns=dataframe.columns)
#
#     return req_frame



# def drop_zero_value_row_of_blast_furnace_signal(dataframe, blast_furnace_signal):
#     #     dataframe = dataframe.reset_index()
#     count = []
#     print(blast_furnace_signal)
#     for idx_blast_furnace, val_blast_furnace in enumerate(dataframe[blast_furnace_signal]):
#         if val_blast_furnace != 100:
#             count = np.append(count, idx_blast_furnace)
#     print('size of count array here: ', count.size)
#
#     if count.size > 0:
#         dataframe_1 = dataframe.drop(count, axis=0)  # axis= 0 means row indiated. 1 means column indicated
#     else:
#         dataframe_1 = dataframe
#     dataframe_1 = dataframe_1.drop([blast_furnace_signal], axis=1)  # dropping the column. because all value are same
#     return dataframe_1


# # Now choose the target colum  and check if any value is zero or not. If zero then drop those rows. here taret column is T9's output, signal name is AEWIHO_T9AV2


def no_zero_value_in_target_1(dataframe, target_column, req_drop_value_target):
#     req_data_1=dataframe.loc[(dataframe[target_column]!=req_drop_value_target)]
    req_data_1 = dataframe.loc[(dataframe[target_column]>=60)]
    req_frame_1=pd.DataFrame(req_data_1,columns=dataframe.columns)
    
    return req_frame_1
    


# def no_zero_value_in_target(dataframe, target_column, req_drop_value_target):
#     req_data_1=dataframe.loc[(dataframe[target_column]!=req_drop_value_target)].values
#     req_frame_1=pd.DataFrame(req_data_1,columns=dataframe.columns)
#
#     return req_frame_1




# def drop_zero_value_row_of_target_signal(dataframe, target_signal):
#     count = []
#     for idx_blast_furnace, val_blast_furnace in enumerate(dataframe[target_signal]):
#         if val_blast_furnace == 0:
#             count = np.append(count, idx_blast_furnace)
#     print(type(count))
#     for i in count:
#         if i > 24222:
#             print(i)
#     print('size of count array: ', len(count))
#
#     if len(count) > 0:
#         dataframe_1 = dataframe.drop(count, axis=0)  # axis= 0 means row indiated. 1 means column indicated
#     else:
#         dataframe_1 = dataframe
#     dataframe_1 = dataframe_1.drop(dataframe_1.columns[0], axis=1)  # generally after resetting index the former index
#     # take place the first place of the column. so removing it.
#     return dataframe_1

def dataframe_reset_index(dataframe):
    dataframe = dataframe.reset_index()
    dataframe = dataframe.drop(['index'], axis=1)

    return dataframe


def drop_column_with_same_value(dataframe):
    cols = dataframe.select_dtypes([np.number]).columns
    diff = dataframe[cols].diff().sum()
    dataframe_drop_column_with_same_value = dataframe.drop(diff[diff == 0].index, axis=1)

    return dataframe_drop_column_with_same_value


# # check on the whole dataframe if there is any NAN value or not. If YES, replace it with zero and drop
# Think twice before using this function
# checking if any column has nan value or not. If YES then replace nan with zero and drop the row

# a = dataframe_no_zero_value_blast_furnace[blast_furnace_signal].isnull().sum()
# print(a)

def drop_nan_value(dataframe):
    for index, column in enumerate(dataframe):
        nan_catcher = dataframe[column].isnull().sum()
        if nan_catcher != 0:
            dataframe_1 = dataframe[column].replace(0, nan)
            dataframe_1 = dataframe.dropna(how='any', axis=0)
        #             print(column,' has total',nan_catcher, 'nan valu')
        else:
            dataframe_1 = dataframe
    #             print(column,' is free from nan value. look it has: ', nan_catcher,' value')

    return dataframe_1

# def drop_row(dataframe):
#     for i in dataframe:
#         #        print(i)
#         dataframe_drop_row_consecutive_same_value = dataframe.loc[dataframe[i].shift() != dataframe[i]]
#
#     return dataframe_drop_row_consecutive_same_value

def drop_unique_valued_columns(dataframe):
    nunique = dataframe.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    dataframe = dataframe.drop(cols_to_drop, axis=1)

    return dataframe



def drop_string_column(dataframe):
    drop_object = dataframe.select_dtypes(exclude=['object'])

    return drop_object

def dataframe_datetime(dataframe):
    dataframe_datetime = dataframe.set_index('dateTime')
    return dataframe_datetime


# # All data cleaning process has done. Now feature selection process will come. Before doing this just make a copy of dataframe and set the index as dateTime


def feature_selection_with_selectKbest(dataframe, max_best_number):
    train_input = dataframe.iloc[:, :-1]
    train_output = dataframe.iloc[:, -1]
    train_output = train_output.to_frame()
    #     train_output = pd.DataFrame(train_output)

    X, y = train_input, train_output
    X = X.astype(int)
    y = y.astype(int)

    bestfeatures = SelectKBest(score_func=chi2, k=2)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    #     print(featureScores.nlargest(20,'Score'))  #print 10 best features
    d = featureScores.nlargest(max_best_number, 'Score')

    e = []
    for i, v in enumerate(d['Specs']):
        e = np.append(e, v)

    e = np.append(e, dataframe.columns[-1])
    final_dataframe = dataframe.iloc[:][e]

    return final_dataframe


# # feature selection with correlation

# find correlated matrix for dataframe which came from sklearn feature selection and the datafarem which has passed
# to sklearn feature selection function

def pearson_correlation(sklearn_dataframe, main_dataframe):
    sklearn_correlation = sklearn_dataframe.corr()
    main_correlation = main_dataframe.corr()
    return sklearn_correlation, main_correlation


# # use the correlation matrix to make the new dataframe where the feature will be the column who has a correlation value with the target in a given range.

# function to make dataframe with high correlated valued column
def make_dataframe_with_high_correlated_value(main_dataframe, correlated_dataframe,
                                              correlation_threshold_value, max_value):
    target_column = main_dataframe.columns[-1]

    dataframe = correlated_dataframe.reset_index()

    high_correlated_array_with_target = []
    for index_corr_reset, val_corr_reset in enumerate(dataframe[target_column]):
        if val_corr_reset > correlation_threshold_value and val_corr_reset < max_value:
            required_column = dataframe.loc[index_corr_reset]['index']
            if required_column != target_column:
                high_correlated_array_with_target = np.append(high_correlated_array_with_target, required_column)
            else:
                print(required_column)
                pass

    final_array = np.append(high_correlated_array_with_target, target_column)
    new_dataframe = main_dataframe.iloc[:][final_array]

    return new_dataframe


def dataframe_date_time_type(dataframe):
    df = pd.DataFrame(index=dataframe.index)
    target_df = dataframe.loc[:, dataframe.columns[-1]]
    df['dateTime_column'] = pd.to_datetime(dataframe.index, format='%Y-%m-%d %H:%M')
    df['day_name'] = df.index.weekday_name
    df['TypeofDAY'] = np.where(df['dateTime_column'].dt.dayofweek < 5, 'Weekday',
                               'Weekend')  # if the associated number less than 5 then weekend, otherwise weekday
    df['TypeofDAY_number'] = np.where(df['dateTime_column'].dt.dayofweek < 5, 1, 0)  # 1 for weekday, 0 for weekend
    df['Date'] = df['dateTime_column'].dt.strftime('%Y-%m-%d')

    df = pd.concat([df, target_df], axis=1)

    return df


def my_sum(x,y):
    s = x+y
    return s


def draw_graph(dictionary_value, dictionary, target, path, subfolder_name):
    fig_location = path + '/' + str(subfolder_name)

    if not os.path.exists(fig_location):
        os.makedirs(fig_location)
    else:
        shutil.rmtree(fig_location, ignore_errors=True)
        os.makedirs(fig_location)
    for i in dictionary_value:
        value = dictionary[i]
        value.iloc[:].plot(y=[target])

        plt.title('visualization of signal ' + str(target) + ' in time of ' + str(i))
        plt.xlabel('range')
        plt.ylabel('value')

        plt.rcParams['figure.figsize'] = (20, 10)
        plt.savefig(fig_location + '/' + str(i) + '.jpg')
        plt.show()


def draw_feature_vs_target(dataframe,final_directory,subfolder):
    fig_location = final_directory + '/' + str(subfolder)

    if not os.path.exists(fig_location):
        os.makedirs(fig_location)
    else:
        shutil.rmtree(fig_location, ignore_errors=True)
        os.makedirs(fig_location)

    for now_num in range(len(dataframe.columns) - 1):
        col_name = dataframe.columns[now_num]
        dataframe.iloc[0:100].plot(dataframe.columns[now_num],dataframe.columns[-1])
        x_axis = dataframe.columns[now_num]
        y_axis = dataframe.columns[-1]
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title('title is ' + str(col_name)+' vs '+str(y_axis))
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.savefig(fig_location + '/' +str(col_name)+' vs '+str(y_axis) + '.jpg')
        plt.show()
