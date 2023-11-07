from config import *
from data_preparation import get_prophet_dataset

import gc
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from itertools import product
from os.path import exists
from prophet import Prophet


# Training and testing Prophet models
def train_test_prophet(dataframes_dict: dict):
    model_metrics_df = None
    for object_name in OBJECTS:
        print("Trying Prophet model with the object %s" % object_name)
        model = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=True, growth='logistic')

        # Get dataset according to the object
        target = None
        if object_name in GENERATION_OBJECTS:
            target = GEN_TARGET_COLUMN
        elif object_name in CONSUMPTION_OBJECTS:
            target = CON_TARGET_COLUMN
        else:
            raise ValueError('Unknown target!')
        dataset = get_prophet_dataset(dataframes_dict[object_name].copy(), TESTING_INTERVALS, target)

        for column in dataset.df_train.columns:
            if column not in ['y', 'ds', 'cap', 'floor']:
                model.add_regressor(column)

        model.fit(dataset.df_train)

        # Test the model with training data (in-sample)
        train_result_df = model.predict(dataset.df_train)
        train_result_df = pd.merge(train_result_df[['ds', 'yhat']], dataset.df_train, on='ds')

        # Test the model with testing data (out of sample)
        test_results_df = model.predict(dataset.df_test)
        test_results_df = pd.merge(test_results_df[['ds', 'yhat']], dataset.df_test, on='ds')

        y_train_real = train_result_df['y'].tolist()
        y_train_pred = train_result_df['yhat'].tolist()
        y_test_real = test_results_df['y'].tolist()
        y_test_pred = test_results_df['yhat'].tolist()

        model_dict = {
            'model': [target],
            'object': [object_name],
            'test_r2': [metrics.r2_score(y_test_real, y_test_pred)],
        }

        model_metrics_df = pd.DataFrame.from_dict(model_dict) if model_metrics_df is None \
            else pd.concat([model_metrics_df, pd.DataFrame.from_dict(model_dict)])

        fig, axs = plt.subplots(2, 2, figsize=(20, 20), dpi=160)
        fig.suptitle("Comparison (Model Prophet %s)" % (object_name))
        for n, interval in enumerate(TESTING_INTERVALS):
            start_date, end_date = interval
            curr_df = test_results_df.loc[
                (test_results_df['ds'].dt.date >= start_date) & (test_results_df['ds'].dt.date <= end_date)]
            axs[n // 2, n % 2].plot(curr_df['ds'], curr_df['y'], alpha=0.5, label='Real Values')
            axs[n // 2, n % 2].plot(curr_df['ds'], curr_df['yhat'], alpha=0.5, label='Predicted Values')
            title = 'Testing %s - %s' % (start_date.strftime(DATE_FORMAT), end_date.strftime(DATE_FORMAT))
            axs[n // 2, n % 2].set_title(title)
            axs[n // 2, n % 2].legend(loc="lower left")
            days = pd.date_range(start_date, end_date + pd.DateOffset(1))
            axs[n // 2, n % 2].set_xticks(days)
            axs[n // 2, n % 2].set_xticklabels([day.strftime(DATE_FORMAT) for day in days])

        for ax in axs.flat:
            ax.set(xlabel='Datetime', ylabel=target)

        plt.savefig(PLOT_FILES_LOC + "model_prophet_%s.png" % (object_name))

        # Cleanup plotting environment after each loop
        plt.close('all')
        plt.cla()
        plt.clf()

    model_metrics_df.to_csv(PROPHET_RESULTS_FILE, mode='a', index=False, header=not (exists(PROPHET_RESULTS_FILE)))
