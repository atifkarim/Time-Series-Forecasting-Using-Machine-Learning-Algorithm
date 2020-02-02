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
from sklearn.model_selection import cross_val_score
from scipy import stats
from scipy.stats import norm
# get_ipython().run_line_magic('matplotlib', 'inline')
#from matplotlib.pylab import rcParams


# function to read the csv file

def read_dataframe(filepath):
    dataframe = pd.read_csv(filepath)  # here the given csv file is reading

    return dataframe


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

        plt.rcParams['figure.figsize'] = (12, 5)
        plt.savefig(str(i) + '.jpg')
        plt.show()


def create_month(dataframe, target_column_month,target_column):
    dataframe = dataframe.set_index('dateTime')
    dataframe['year'] = pd.DatetimeIndex(dataframe.index).year
    dataframe['month'] = pd.DatetimeIndex(dataframe.index).month

    dict_of_month = {k: v for k, v in dataframe.groupby('month')}
    month_key_value = collections.OrderedDict(dict_of_month)

    month_array_df = distinct_month_1(dataframe, target_column_month, month_key_value)

#     draw_month_fig = draw_month(month_key_value, dict_of_month, target_column)

    return dataframe, month_array_df


def choose_month(dataframe, target_column_month):
    req_data_month = dataframe.loc[(dataframe[target_column_month] == 2) | (dataframe[target_column_month] == 3)]
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


def ascending_dataframe(dataframe, start_pos, end_pos):
    #     multivariate_data=test_new.iloc[start_pos:end_pos][multivariate_column_label] # comment out this line if you pass column label
    dataframe = dataframe.iloc[start_pos:end_pos][:]
    dataframe = dataframe.loc[::-1]

    return dataframe


# # Now target column and dateTime colum will be arranged as a given column index. Here target column is the output of turbine 9's output


def rearrange_dataframe(dataframe, colname, col_pos):
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

def check_blast_furnace(dataframe,furnace_signal_column_a,value_A, furnace_signal_column_b,value_B):
    req_data=dataframe.loc[(dataframe[furnace_signal_column_a]>=value_A) | (dataframe[furnace_signal_column_b]>=value_B)]
    req_frame=pd.DataFrame(req_data,columns=dataframe.columns)
    dataframe = req_frame.reset_index()
    dataframe = dataframe.drop(['index'], axis=1)
    
    return dataframe


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


def check_target_column(dataframe, target_column, req_drop_value_target):
#     req_data_1=dataframe.loc[(dataframe[target_column]!=req_drop_value_target)]
    dataframe = dataframe.loc[(dataframe[target_column]>=60) & (dataframe[target_column]<=90)]
    dataframe = pd.DataFrame(dataframe,columns=dataframe.columns)
    
    return dataframe
    


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
    dataframe_datetime.describe()
    return dataframe_datetime


# free target column from outlier
def free_target_column_from_outlier(dataframe,target_column):
    dataframe = dataframe[(np.abs(stats.zscore(dataframe[target_column])) < 3)]
    return dataframe


# function to remove outlier from all column
def free_dataframe_from_outlier(dataframe):
    dataframe = dataframe[(np.abs(stats.zscore(dataframe)) < 3).all(axis=1)]
    
    return dataframe


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

def pearson_correlation(dataframe):
    correlation = dataframe.corr()
    return correlation


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


def draw_gaussian_curve(dataframe,target_column,graph_name):
    mean_with_outlier = dataframe.describe()[target_column]['mean']
    std_with_outlier = dataframe.describe()[target_column]['std']
    var_with_outlier = (std_with_outlier)**2
    print(var_with_outlier, std_with_outlier)
    min_value_with_outlier = dataframe.describe()[target_column]['min']
    max_value_with_outlier = dataframe.describe()[target_column]['max']
    
    # calculate the z-transform
    z1 = ( min_value_with_outlier - mean_with_outlier ) / std_with_outlier
    z2 = ( max_value_with_outlier - mean_with_outlier ) / std_with_outlier

    x = np.arange(z1, z2, 0.001) # range of x in spec
    x_all = np.arange(-10, 10, 0.001) # entire range of x, both in and out of spec
    # mean = 0, stddev = 1, since Z-transform was calculated
    y = norm.pdf(x,0,1)
    y2 = norm.pdf(x_all,0,1)
    
    fig, ax = plt.subplots(figsize=(9,6))
    plt.style.use('fivethirtyeight')
    ax.plot(x_all,y2)
    
    ax.fill_between(x,y,0, alpha=0.3, color='b')
    ax.fill_between(x_all,y2,0, alpha=0.1)
    ax.set_xlim([-4,4])
    ax.set_xlabel('# of Standard Deviations Outside the Mean')
    ax.set_yticklabels([])
    ax.set_title('Normal Gaussian Curve')
    plt.savefig(graph_name+'_normal_curve.png', dpi=72, bbox_inches='tight')
    plt.show()
    
def gaussian_curve(dataframe, target_column,name):
    mean = dataframe.describe()[target_column]['mean']
    std = dataframe.describe()[target_column]['std']
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    plt.plot(x, stats.norm.pdf(x, mean, std))
    plt.savefig(name+'gaussian_normal_curve.png',bbox_inches='tight')
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.show()


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
        plt.savefig(fig_location + '/' + str(i) + '.jpg',bbox_inches='tight')
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
        plt.savefig(fig_location + '/' +str(col_name)+' vs '+str(y_axis) + '.jpg',bbox_inches='tight')
        plt.show()


def score_checking_with_cross_validation(model_list, train_input, train_output, evaluation_metrics_file_path,name):
    f = open(evaluation_metrics_file_path, 'a')
    f.write('\n'+'Score checking with Cross Validation')
    f.write('\n')
    f.close()
    for index, value in enumerate(model_list):
        scores_r2 = cross_val_score(value, train_input, train_output, cv=10, scoring='r2')
        scores = cross_val_score(value, train_input, train_output, cv=10, scoring='neg_mean_squared_error')
        mse_scores = -scores
        rmse_scores = np.sqrt(mse_scores)
        print(name[index], '--' * 5, scores_r2.mean())
        print(name[index], '--' * 5, rmse_scores.mean())
        f = open(evaluation_metrics_file_path, 'a')
        f.write(str(name[index]) + '\t' + 'RMSE: ' + str(rmse_scores.mean()) + '\n')
        f.write(str(name[index]) + '\t' + 'r_2 square: ' + str(scores_r2.mean()) + '\n')
        f.write('\n')
    f.write('Score checking finish with Cross validation'+'\n')
    f.close()

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    #     plt.savefig('check_stationarity.jpg')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', regression='c')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    p_value = dfoutput['p-value']
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s-------)' % key] = value
    print(dfoutput)

    if p_value <= 0.05:
        print (p_value,": Rejecting Null Hypothesis.")
        print("Series is Stationary.")
    else:
        print(p_value,": Weak evidence to reject the Null Hypothesis.")
        print("Series is Non-Stationary.")


def tsplot_dataset(df, target_column):
    n_sample = df.shape[0]
    print(n_sample)
    n_train = int(0.995 * n_sample) + 1
    n_forecast = n_sample - n_train

    ts_train = df.iloc[:n_train][target_column]
    ts_test = df.iloc[n_train:][target_column]
    print(ts_train.shape)
    print(ts_test.shape)
    print("Training Series:", "\n", ts_train.head(), "\n")
    print("Testing Series:", "\n", ts_test.head())

    return n_sample, ts_train, ts_test
import statsmodels.tsa.api as smt
import seaborn as sns

def tsplot(y, lags=None, title='', figsize=(14, 8)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    fig.tight_layout()
    plt.savefig('tsplot.png',bbox_inches='tight')
    return ts_ax, acf_ax, pacf_ax