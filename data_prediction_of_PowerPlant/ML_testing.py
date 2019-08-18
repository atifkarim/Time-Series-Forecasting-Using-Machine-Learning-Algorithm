print('what !!!!')
from main import dataframe_high_correlation, final_directory
from model_file import make_dataset, scikit_learn_model
from sklearn.ensemble import ExtraTreesRegressor

print('AM I HERE NOW !!!!!')
print(type(dataframe_high_correlation))
dataframe_high_correlation.head()

print(final_directory)

print('I will be here now')
train_input, train_output, test_input, test_output = make_dataset(dataframe_high_correlation)

#s_array = dataframe_high_correlation.values
model_list = [ExtraTreesRegressor()]
name = ['ExtraTreesRegressor']

model = scikit_learn_model(model_list, name, train_input, train_output, test_input, test_output, final_directory)
