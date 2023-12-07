import pandas as pd
from pvlib.location import Location

# Data files
DATA_FILES_LOC = "Data-Files/"
WEATHER_DATA = DATA_FILES_LOC + "weather_data.csv"
SOLAR_5K_DATA = DATA_FILES_LOC + "solar_5k_generation.csv"
SOLAR_30K_DATA = DATA_FILES_LOC + "solar_30k_generation.csv"
CONSUMPTION_DATA = DATA_FILES_LOC + "consumption_data.csv"
DATETIME_COLS = ['datetime_unix', 'datetime_local']

# Result files
RESULT_FILES_LOC = "Model-Results/"
NN_RESULTS_FILE = RESULT_FILES_LOC + 'nn_results.csv'
PROPHET_RESULTS_FILE = RESULT_FILES_LOC + 'prophet_results.csv'
BASELINE_RESULTS_FILE = RESULT_FILES_LOC + 'baseline_results.csv'

# Plot Files
PLOT_FILES_LOC = "Model-Results/Plots/"

# Location of a Solar Power plants
LOCATION = Location(54.925976, 25.371658, tz='Europe/Vilnius')

# Target columns
GEN_TARGET_COLUMN = 'generation'
CON_TARGET_COLUMN = 'consumption'

# Testing intervals for dividing data into training and testing
DATE_FORMAT = '%Y-%m-%d'
TESTING_INTERVALS = [
    (pd.to_datetime('2021-03-20', format=DATE_FORMAT), pd.to_datetime('2021-03-22', format=DATE_FORMAT)),
    (pd.to_datetime('2021-07-10', format=DATE_FORMAT), pd.to_datetime('2021-07-12', format=DATE_FORMAT)),
    (pd.to_datetime('2021-10-28', format=DATE_FORMAT), pd.to_datetime('2021-10-30', format=DATE_FORMAT)),
    (pd.to_datetime('2022-02-26', format=DATE_FORMAT), pd.to_datetime('2022-02-28', format=DATE_FORMAT))
]

GENERATION_OBJECTS = ["solar_30k", "solar_5k"]
CONSUMPTION_OBJECTS = ["1", "5"]

# Neural networks parameter tuning
OBJECTS = GENERATION_OBJECTS + CONSUMPTION_OBJECTS
OPTIMIZERS = ['Adam', 'RMSProp']
LOSS_FUNCTIONS = ['MAE', 'MSE', 'MAPE']
LEARNING_RATES = [0.001, 0.0001]
ARCHITECTURES = ['FEEDFORWARD', 'LSTM', 'GRU']
SEQUENCE_LENGHTS = [12, 24, 36]
NUMBER_OF_LAYERS = [2, 4]

# 4 Different training steps, model will be trained for 400 (50 + 50 + 100 + 200) epochs in total
NUMBER_OF_EPOCHS = [50, 50, 100, 200]
