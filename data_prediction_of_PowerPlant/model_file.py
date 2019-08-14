import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import *
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math


def make_dataset(dataframe):
    dataset = np.array(dataframe)
    NumberOfElements = int(len(dataset) * 0.97)
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

#(model_list: object, name: object, train_input: object, train_output: object, test_input: object, test_output: object) -> object
def scikit_learn_model(model_list, name, train_input,train_output, test_input, test_output):
    for idx, i in enumerate(model_list):
        train_model_1 = i
        print('-------', name[idx])
        train_model_1.fit(train_input, train_output)
        predicted_output = train_model_1.predict(test_input)
        print('r_2 statistic: %.2f' % r2_score(test_output, predicted_output))
        print("Mean_absolute_error: %.2f" % mean_absolute_error(test_output, predicted_output))
        print("Mean squared error: %.2f" % mean_squared_error(test_output, predicted_output))
        RMSE = math.sqrt(mean_squared_error(test_output, predicted_output))
        print('RMSE: ', RMSE)
        print('!!!!---------------!!!!----------------!!!!')