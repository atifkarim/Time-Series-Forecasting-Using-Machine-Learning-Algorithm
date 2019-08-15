import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
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
def scikit_learn_model(model_list, name, train_input, train_output, test_input, test_output, final_directory):
    for idx, i in enumerate(model_list):
        train_model_1 = i
        print('-------', name[idx])
        train_model_1.fit(train_input, train_output)
        predicted_output = train_model_1.predict(test_input)
        
        graph = plot_graph(test_output, predicted_output,final_directory,name[idx])
        evaluate_model = evaluation_metrices(test_output, predicted_output)



def plot_graph(test_output, predicted_output, final_directory,subfolder):
    fig_location = final_directory + '/' + str(subfolder)

    if not os.path.exists(fig_location):
        os.makedirs(fig_location)
    else:
        shutil.rmtree(fig_location, ignore_errors=True)
        os.makedirs(fig_location)

    plt.plot((min(test_output), max(test_output)), (min(predicted_output), max(predicted_output)), color='red')
    plt.scatter(test_output, predicted_output, color='blue')
    # plt.savefig(model+'_'+'figure_actual_vs_predicted_with_best_fit_line.jpg')
    plt.xlabel('test_output')
    plt.ylabel('predicted_output')
    plt.title('scatter plotting of predicted_output alongside with the average line of test and predicted output')
    plt.savefig(fig_location + '/' + "scatter_test_pred" + '.jpg')
    plt.show()

    difference_of_value = predicted_output - test_output
    print(type(difference_of_value))

    plt.plot(difference_of_value[:])
    plt.title('observation of the difference of actual and predicted value')

    # plt.rcParams['xtick.labelsize']=2
    # plt.rcParams['ytick.labelsize']=2
    # plt.tick_params(labelsize=20)
    plt.ylabel('difference of value')
    plt.xlabel('range')
    plt.grid(b=None, which='both', axis='both')
    plt.savefig(fig_location + '/' + "difference_test_pred" + '.jpg')
    plt.show()

    plt.hist(difference_of_value, bins=20)
    # plt.xlim(-10,10,1)
    # plt.savefig(model+'_'+'histogram_of_difference_value.jpg')
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title('histogram of value of difference')
    plt.savefig(fig_location + '/' + "error_histogram" + '.jpg')
    plt.show()

    plt.plot(predicted_output[0:len(predicted_output[0:])], color='blue')
    plt.plot(test_output[0:], color='red')
    # plt.xlim(0,40,1)
    # plt.ylim(50,70,1)
    # plt.savefig(model+'_'+'figure_difference_between_actual_and_predicted_value.jpg')
    plt.xlabel('range')
    plt.ylabel('value of test and predicted output')
    plt.title('Visualization of test and predicted output in the same timestamp')
    plt.savefig(fig_location + '/' + "test_and_pred" + '.jpg')
    plt.show()



def evaluation_metrices(test_output,predicted_output):
    print('r_2 statistic: %.2f' % r2_score(test_output, predicted_output))
    print("Mean_absolute_error: %.2f" % mean_absolute_error(test_output, predicted_output))
    print("Mean squared error: %.2f" % mean_squared_error(test_output, predicted_output))
    RMSE = math.sqrt(mean_squared_error(test_output, predicted_output))
    print('RMSE: ', RMSE)
    print('!!!!---------------!!!!----------------!!!!')