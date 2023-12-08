from config import *
from data_preparation import get_keras_dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from os.path import exists
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Training and testing Keras LSTM models
def train_test_keras(dataframes_dict: dict):
    model_metrics_df = None

    for object_name in OBJECTS:
        print("Trying Keras LSTM model with the object %s" % object_name)

        # Get dataset according to the object
        target = None
        if object_name in GENERATION_OBJECTS:
            target = GEN_TARGET_COLUMN
        elif object_name in CONSUMPTION_OBJECTS:
            target = CON_TARGET_COLUMN
        else:
            raise ValueError('Unknown target!')

        dataset = get_keras_dataset(dataframes_dict[object_name].copy(), TESTING_INTERVALS, target)

        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(dataset.df_train[['ds', target]])

        # Create sequences for LSTM
        def create_sequences(data, seq_length):
            sequences = []
            for i in range(len(data) - seq_length):
                seq = data[i:i+seq_length]
                sequences.append(seq)
            return np.array(sequences)

        seq_length = 10  # Adjust as needed
        X_train = create_sequences(scaled_data, seq_length)
        y_train = scaled_data[seq_length:, 1]  # Assuming the target column is the second column after normalization

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32)

        # Test the model with testing data
        test_data = scaler.transform(dataset.df_test[['ds', target]])
        X_test = create_sequences(test_data, seq_length)
        y_test_real = test_data[seq_length:, 1]
        y_test_pred = model.predict(X_test).reshape(-1)

        # Inverse transform to get real values
        y_test_real = scaler.inverse_transform(np.expand_dims(y_test_real, axis=1)).reshape(-1)
        y_test_pred = scaler.inverse_transform(np.expand_dims(y_test_pred, axis=1)).reshape(-1)

        # Calculate metrics
        r2 = r2_score(y_test_real, y_test_pred)
        mse = mean_squared_error(y_test_real, y_test_pred)

        model_dict = {
            'model': ['LSTM'],
            'object': [object_name],
            'test_r2': [r2],
            'test_mse': [mse],
        }

        model_metrics_df = pd.DataFrame.from_dict(model_dict) if model_metrics_df is None \
            else pd.concat([model_metrics_df, pd.DataFrame.from_dict(model_dict)])

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(dataset.df_test['ds'], y_test_real, label='Real Values')
        plt.plot(dataset.df_test['ds'], y_test_pred, label='Predicted Values')
        plt.title("Comparison (Model LSTM %s)" % object_name)
        plt.xlabel('Datetime')
        plt.ylabel(target)
        plt.legend()
        plt.savefig(PLOT_FILES_LOC + "model_lstm_%s.png" % (object_name))

        plt.close('all')
        plt.cla()
        plt.clf()

    model_metrics_df.to_csv(KERAS_RESULTS_FILE, mode='a', index=False, header=not (exists(KERAS_RESULTS_FILE)))
