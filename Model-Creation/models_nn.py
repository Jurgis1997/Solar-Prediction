from config import *
from data_preparation import get_nn_dataset

import gc
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics as metrics

from itertools import product
from os.path import exists


_NUMBER_OF_UNITS = 64


def construct_lstm(num_timesteps: int, num_features: int, num_layers: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(_NUMBER_OF_UNITS, input_shape=(num_timesteps, num_features), return_sequences=True)
    ])
    if num_layers == 4:
        model.add(tf.keras.layers.LSTM(_NUMBER_OF_UNITS, return_sequences=True))
        model.add(tf.keras.layers.LSTM(_NUMBER_OF_UNITS, return_sequences=True))

    model.add(tf.keras.layers.LSTM(_NUMBER_OF_UNITS))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(0.2)))
    return model


def construct_gru(num_timesteps: int, num_features: int, num_layers: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(_NUMBER_OF_UNITS, input_shape=(num_timesteps, num_features), return_sequences=True)
    ])
    if num_layers == 4:
        model.add(tf.keras.layers.GRU(_NUMBER_OF_UNITS, return_sequences=True))
        model.add(tf.keras.layers.GRU(_NUMBER_OF_UNITS, return_sequences=True))

    model.add(tf.keras.layers.GRU(_NUMBER_OF_UNITS))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(0.2)))
    return model


def construct_feed_forward(num_features: int, num_layers: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(_NUMBER_OF_UNITS, input_shape=(None, num_features), activation='relu'),
        tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'),
        tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'),
    ])
    if num_layers == 4:
        model.add(tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'))
        model.add(tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'))
        model.add(tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'))
        model.add(tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'))

    model.add(tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(0.2)))
    return model


def construct_model(architecture: str, num_timesteps: int, num_features: int, num_layers: int) -> tf.keras.Sequential:
    if architecture == 'FEEDFORWARD':
        return construct_feed_forward(num_features, num_layers)

    elif architecture == 'LSTM':
        return construct_lstm(num_timesteps, num_features, num_layers)

    elif architecture == 'GRU':
        return construct_gru(num_timesteps, num_features, num_layers)

    else:
        raise ValueError('Unknown architecture!')


# Training and testing neural network models
def train_test_nn(dataframes_dict: dict):
    params_product = product(OBJECTS, OPTIMIZERS, LOSS_FUNCTIONS, LEARNING_RATES,
                             ARCHITECTURES, SEQUENCE_LENGHTS, NUMBER_OF_LAYERS)
    for idx, parameters in enumerate(params_product, start=1):
        (object_name, optimizer_str, loss_str, lr, architecture, sequence_len, number_of_layers) = parameters
        print("Trying model %d with the following parameters:" % idx)
        print(parameters)

        # Code below makes sure that the Feed-Forward models aren't trained with different sequence lengths
        if architecture == 'FEEDFORWARD':

            # Skip the training process as the Feed-Forward model doesn't have a sequence length
            if sequence_len != SEQUENCE_LENGHTS[0]:
                print("Model %d is already trained. Skipping..." % idx)
                continue

            # Set the sequence_len to 1 for Feed-Forward models
            else:
                sequence_len = 1

        # Get dataset according to the object
        target = None
        if object_name in GENERATION_OBJECTS:
            target = GEN_TARGET_COLUMN
        elif object_name in CONSUMPTION_OBJECTS:
            target = CON_TARGET_COLUMN
        else:
            raise ValueError('Unknown target!')
        dataset = get_nn_dataset(dataframes_dict[object_name].copy(), TESTING_INTERVALS, sequence_len, target)

        # Construct the tensorflow sequential model according to the current parameters
        model = construct_model(architecture, sequence_len, dataset.num_features, number_of_layers)

        # Set the optimizer
        optimizer = None
        if optimizer_str == 'Adam':
            optimizer = tf.keras.optimizers.Adam(lr)
        elif optimizer_str == 'RMSProp':
            optimizer = tf.keras.optimizers.RMSprop(lr)
        else:
            raise ValueError('Unknown optimizer!')

        # Set the loss function
        loss = None
        if loss_str == 'MAE':
            loss = tf.keras.losses.MeanAbsoluteError()
        elif loss_str == 'MSE':
            loss = tf.keras.losses.MeanSquaredError()
        elif loss_str == 'MAPE':
            loss = tf.keras.losses.MeanAbsolutePercentageError()
        else:
            raise ValueError('Unknown loss function!')

        model.compile(optimizer=optimizer, loss=loss)
        model.summary()

        model_metrics_df = None

        total_epochs = 0
        for num_of_epochs in NUMBER_OF_EPOCHS:
            total_epochs += num_of_epochs

            model.fit(dataset.x_train, dataset.y_train, batch_size=256, epochs=num_of_epochs)

            # Test the model with training data (in-sample)
            y_train_pred = model.predict(dataset.x_train).reshape(-1)
            y_train_real = dataset.y_train.reshape(-1)

            # Test the model with testing data (out of sample)
            y_test_pred = model.predict(dataset.x_test).reshape(-1)
            y_test_real = dataset.y_test.reshape(-1)

            # Model results dictionary
            model_dict = {
                'id': [idx],
                'number_of_epochs': [total_epochs],
                'model': [target],
                'architecture': [architecture],
                'object': [object_name],
                'optimizer': [optimizer_str],
                'loss': [loss_str],
                'learning_rate': [lr],
                'sequence_len': [sequence_len],
                'number_of_layers': [number_of_layers],
                'test_r2': [metrics.r2_score(y_test_real, y_test_pred)],
            }

            # Model results with the same parameters but different number of epochs are store in the same dataframe
            model_metrics_df = pd.DataFrame.from_dict(model_dict) if model_metrics_df is None \
                else pd.concat([model_metrics_df, pd.DataFrame.from_dict(model_dict)])

            # Plot the comparison with the testing dataset
            fig, axs = plt.subplots(2, 2, figsize=(20, 20), dpi=160)
            fig.suptitle("Comparison (Model ID %d, epochs - %d)" % (idx, total_epochs))
            for n, interval in enumerate(TESTING_INTERVALS):
                start_date, end_date = interval
                hours = pd.date_range(start_date, end_date + pd.DateOffset(1), freq='H').tolist()[:-1]
                axs[n // 2, n % 2].plot(
                    hours, y_test_real[n * len(hours): (n + 1) * len(hours)], alpha=0.5, label='Real Values')
                axs[n // 2, n % 2].plot(
                    hours, y_test_pred[n * len(hours): (n + 1) * len(hours)], alpha=0.5, label='Predicted Values')

                title = 'Testing %s - %s' % (start_date.strftime(DATE_FORMAT), end_date.strftime(DATE_FORMAT))
                axs[n // 2, n % 2].set_title(title)
                axs[n // 2, n % 2].legend(loc="lower left")
                days = pd.date_range(start_date, end_date + pd.DateOffset(1))
                axs[n // 2, n % 2].set_xticks(days)
                axs[n // 2, n % 2].set_xticklabels([day.strftime(DATE_FORMAT) for day in days])

            for ax in axs.flat:
                ax.set(xlabel='Datetime', ylabel=target)

            plt.savefig(PLOT_FILES_LOC + "model_%d_%d.png" % (idx, total_epochs))

            # Cleanup plotting environment after each loop
            plt.close('all')
            plt.cla()
            plt.clf()

        model_metrics_df.to_csv(NN_RESULTS_FILE, mode='a', index=False, header=not (exists(NN_RESULTS_FILE)))

        # Free up memory
        del dataset, model, model_dict, model_metrics_df, loss, optimizer
        del y_train_real, y_train_pred, y_test_pred, y_test_real, total_epochs
        gc.collect()
