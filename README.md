# Forecasting of different sensors' data of a power plant

This respository contains code to foecast data of numerous sensors. Data cleaning and prediction task is done step by step to perform the whole work. In [develop](https://github.com/atifkarim/Time-Series-Forecasting-of-a-Power-Plant/tree/develop) branch all of the code will be found.

## Requirements
* Python 3.6
* Code is tested on Ubuntu 16.04, 18.04, Windows 10  
* Use [requirements.txt](https://github.com/atifkarim/Time-Series-Forecasting-Using-Machine-Learning-Algorithm/blob/master/requirements.txt) file to install necessary library for this project by using ```pip install -r requirements.txt``` command in the terminal. [To make this requirements.txt file I have used the information from this repository](https://github.com/bndr/pipreqs).

## Idea
The idea behind the task is to observe reaction of different machine learning models to the provided data from Salzgitter AG, a reputed steel industry of Germany. Provided data contains information regarding the integrated power plant of Salzgitter AG. Here, I have tried to forecast Turbine data for each minute. Provided data is in time-series format. Initially, all of the raw data is cleaned and visualized using Pandas, NumPy etc. Then, stationarity of time series is checked by ADF test. ARIMA, Linear regression, Decision Tree Regression, Neural Network, Long Short Term Memory(LSTM) are used to do the forecasting.

## Useful links for theoretical knowledge
* [Time series explanation by **IBM**](https://www.ibm.com/support/knowledgecenter/en/SS3RA7_17.0.0/clementine/timeseriesnode_general.html)
* [TIme series analysis by **Duke**](https://people.duke.edu/~rnau/411arim3.htm)
* [LSTM from **Cohla's blog**](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [LSTM explanation by **Shi Yan**](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)
* [LSTM explanation by Michael Nguyen](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

## Code to execute
* After cloning this repository you have to set the name of the **csv file** where your data is stored. For doing this just take a look in this [configuration file](https://github.com/atifkarim/Time-Series-Forecasting-Using-Machine-Learning-Algorithm/blob/master/data_analysis_and_forecasting/variable_config.json)
* In [main.py](https://github.com/atifkarim/Time-Series-Forecasting-Using-Machine-Learning-Algorithm/blob/master/data_analysis_and_forecasting/main.py#L117) replace the variable name. If this line is not found just try to find **dataframe read** variable name and use that to read your **csv file**
* This file is used for [data preprocessing](https://github.com/atifkarim/Time-Series-Forecasting-Using-Machine-Learning-Algorithm/blob/master/data_analysis_and_forecasting/dataset_analysis.py)
* All of the machine learning model is demonstrated [here](https://github.com/atifkarim/Time-Series-Forecasting-Using-Machine-Learning-Algorithm/blob/master/data_analysis_and_forecasting/model_file.py)
* To run code in your favourite IDE/ terminal execute **python main.py**

<!--- **testing bold**\--->
<!--- *testing italic*--->
<!--- \--->
<!--- check list--->
<!--- * Item 1--->
<!---* Item 2--->
 <!--- * Item 2a--->
  <!---* Item 2b--->

<!---ordered list\--->
<!---1. Item 1--->
<!---1. Item 2--->
<!---1. Item 3--->
  <!--- 1. Item 3a--->
   <!---1. Item 3b--->
<!---      1. Item e--->
<!---        <!--- 1.klkl--->
       
      
      
      

<!---comment--->
<!--- comment this line --->
