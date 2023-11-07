import pandas as pd
from pvlib.location import Location
from feature_encodings import get_solar_noon
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from os.path import exists
from config import *

# Solar generation data taken from https://www.saulesgraza.lt/saules-elektrines-generacija
_ORIGINAL_POWER = 10.7
_DATA = [
    [2014, 1, 385], [2015, 1, 191], [2016, 1, 260], [2017, 1, 140], [2018, 1, 160], [2019, 1, 147],
    [2014, 2, 392], [2015, 2, 434], [2016, 2, 293], [2017, 2, 315], [2018, 2, 475], [2019, 2, 375],
    [2014, 3, 942], [2015, 3, 898], [2016, 3, 696], [2017, 3, 716], [2018, 3, 1050], [2019, 3, 809],
    [2014, 4, 1466], [2015, 4, 1130], [2016, 4, 933], [2017, 4, 1062], [2018, 4, 1052], [2019, 4, 1685],
    [2014, 5, 1390], [2015, 5, 1415], [2016, 5, 1588], [2017, 5, 1557], [2018, 5, 1860], [2019, 5, 1263],
    [2014, 6, 1294], [2015, 6, 1625], [2016, 6, 1559], [2017, 6, 1304], [2018, 6, 1528], [2019, 6, 1721],
    [2014, 7, 1688], [2015, 7, 1472], [2016, 7, 1170], [2017, 7, 1303], [2018, 7, 1343], [2019, 7, 1403],
    [2014, 8, 1284], [2015, 8, 1734], [2016, 8, 1202], [2017, 8, 1332], [2018, 8, 1395], [2019, 8, 1357],
    [2014, 9, 1107], [2015, 9, 1005], [2016, 9, 1082], [2017, 9, 853], [2018, 9, 1132], [2019, 9, 1013],
    [2014, 10, 876], [2015, 10, 845], [2016, 10, 431], [2017, 10, 394], [2018, 10, 851], [2019, 10, 592],
    [2014, 11, 230], [2015, 11, 172], [2016, 11, 161], [2017, 11, 131], [2018, 11, 127], [2019, 11, 155],
    [2014, 12, 193], [2015, 12, 198], [2016, 12, 124], [2017, 12, 105], [2018, 12, 51], [2019, 12, 113],
]

_INVERTERS_DICT = {
    "solar_5k": {
        "max_power": 5.5,
        "data_file": SOLAR_5K_DATA
    },
    "solar_30k": {
        "max_power": 30,
        "data_file": SOLAR_30K_DATA
    }
}

_MONTH_ADDENDS_DICT = {
    "Last_Month_Middle_Day": -1,
    "This_Month_Middle_Day": 0,
    "Next_Month_Middle_Day": 1,
}


# Calculate expected daily generation based on an average monthly values
def calculate_daily_gen(row):
    if row['Date'] == row['This_Month_Middle_Day']:
        return row['This_Month_Average_Gen']
    else:
        if row['Date'] < row['This_Month_Middle_Day']:
            day_span = (row['This_Month_Middle_Day'] - row['Last_Month_Middle_Day']).days
            left_span = (row['Date'] - row['Last_Month_Middle_Day']).days
            right_span = (row['This_Month_Middle_Day'] - row['Date']).days
            left_avg, right_avg = row['Last_Month_Average_Gen'], row['This_Month_Average_Gen']

        else:  # row['Date'] > row['This_Month_Middle_Day']:
            day_span = (row['Next_Month_Middle_Day'] - row['This_Month_Middle_Day']).days
            left_span = (row['Date'] - row['This_Month_Middle_Day']).days
            right_span = (row['Next_Month_Middle_Day'] - row['Date']).days
            left_avg, right_avg = row['This_Month_Average_Gen'], row['Next_Month_Average_Gen']

        return (left_span * right_avg) / day_span + (right_span * left_avg) / day_span


def test_baseline():
    model_metrics_df = None
    for object in GENERATION_OBJECTS:

        # Calculate day average generation for each month
        df_averages = pd.DataFrame(_DATA, columns=['Year', 'Month', 'Generation'])
        df_averages['Avg_Generation'] = df_averages.apply(
            lambda x: x['Generation'] / pd.Period(year=x['Year'], month=x['Month'], freq='m').days_in_month,
            axis=1
        )
        df_averages = df_averages.drop('Year', axis=1).groupby(['Month'], as_index=False).mean()
        averages_dict = df_averages.set_index('Month')['Avg_Generation'].to_dict()
        print(averages_dict)

        gen_df = pd.read_csv(_INVERTERS_DICT[object]["data_file"], parse_dates=['datetime_unix', 'datetime_local'])
        gen_df = gen_df.rename(columns={'datetime_local': 'datetime'})
        gen_df['Months'] = gen_df.apply(lambda x: x['datetime'].replace(day=1, hour=0, minute=0, second=0), axis=1)

        months_only_df = gen_df['Months']
        months_only_df = months_only_df.drop_duplicates().to_frame(name='Months')

        for month_col in _MONTH_ADDENDS_DICT:
            # Find target month
            if _MONTH_ADDENDS_DICT[month_col] == -1:
                months_only_df[month_col] = months_only_df.apply(lambda x: x['Months'] - pd.Timedelta(days=1), axis=1)
            elif _MONTH_ADDENDS_DICT[month_col] == 1:
                months_only_df[month_col] = months_only_df.apply(lambda x: x['Months'] + pd.Timedelta(days=31), axis=1)
            else:
                months_only_df[month_col] = months_only_df['Months']

            # Calculate middle days of the previous, current and following months
            months_only_df[month_col] = months_only_df.apply(
                lambda x: x[month_col].replace(
                    day=pd.Period(
                        year=x[month_col].year, month=x[month_col].month, freq='m'
                    ).days_in_month // 2
                ), axis=1)

            # Set average daily generation for each month
            months_only_df[month_col.replace('Middle_Day', 'Average_Gen')] = months_only_df.apply(
                lambda x: averages_dict[x[month_col].month], axis=1
            )

        gen_df = gen_df.merge(months_only_df, on='Months')
        gen_df = gen_df.drop('Months', axis=1)
        gen_df['Date'] = gen_df['datetime'].dt.normalize()
        gen_df['Daily_Peak_Average'] = gen_df.apply(lambda x: calculate_daily_gen(x), axis=1)

        dt_index = pd.DatetimeIndex(gen_df['Date']).tz_localize(tz=LOCATION.tz).drop_duplicates()
        sun_points_df = LOCATION.get_sun_rise_set_transit(dt_index, method='spa').reset_index()

        # Remove timezone mark from the timestamps
        for column in list(sun_points_df):
            sun_points_df[column] = sun_points_df[column].dt.tz_localize(None)

        # Merge calculated dataframe with the main one
        gen_df = gen_df.merge(sun_points_df, on='Date', how='left')
        gen_df['encoded_noon'] = gen_df.apply(
            lambda x: get_solar_noon(x['datetime'], x['sunrise'], x['transit'], x['sunset']), axis=1)

        df_temp = gen_df[['Date', 'encoded_noon']].groupby(['Date'], as_index=False).sum()
        gen_df = gen_df.merge(df_temp.rename(columns={'encoded_noon': 'encoded_noon_sum'}), on='Date')
        gen_df['generation_predicted'] = gen_df['Daily_Peak_Average'] * \
                                         (gen_df['encoded_noon'] / gen_df['encoded_noon_sum']) * \
                                         (_INVERTERS_DICT[object]["max_power"] / _ORIGINAL_POWER)
        gen_df = gen_df[['datetime', 'generation', 'generation_predicted']]

        testing_dataframes = []
        for start_date, end_date in TESTING_INTERVALS:
            testing_df = gen_df.loc[(gen_df['datetime'].dt.date >= start_date) &
                                    (gen_df['datetime'].dt.date <= end_date)]
            testing_dataframes.append(testing_df)

        comparison_df = pd.concat(testing_dataframes)
        print(comparison_df.to_string())

        y_test_real = comparison_df['generation'].tolist()
        y_test_pred = comparison_df['generation_predicted'].tolist()

        model_dict = {
            'model': [GEN_TARGET_COLUMN],
            'object': [object],
            'test_r2': [metrics.r2_score(y_test_real, y_test_pred)],
        }

        model_metrics_df = pd.DataFrame.from_dict(model_dict) if model_metrics_df is None \
            else pd.concat([model_metrics_df, pd.DataFrame.from_dict(model_dict)])

        fig, axs = plt.subplots(2, 2, figsize=(20, 20), dpi=160)
        fig.suptitle("Comparison (Model Baseline %s)" % (object), fontsize="x-large")
        for n, interval in enumerate(TESTING_INTERVALS):
            start_date, end_date = interval
            curr_df = gen_df.loc[(gen_df['datetime'].dt.date >= start_date) & (gen_df['datetime'].dt.date <= end_date)]
            axs[n // 2, n % 2].plot(
                curr_df['datetime'], curr_df['generation'], alpha=0.5, label='Real Values'
            )
            axs[n // 2, n % 2].plot(
                curr_df['datetime'], curr_df['generation_predicted'], alpha=0.5, label='Predicted Values'
            )
            title = 'Testing %s - %s' % (start_date.strftime(DATE_FORMAT), end_date.strftime(DATE_FORMAT))
            axs[n // 2, n % 2].set_title(title)
            axs[n // 2, n % 2].legend(loc="lower left")
            days = pd.date_range(start_date, end_date + pd.DateOffset(1))
            axs[n // 2, n % 2].set_xticks(days)
            axs[n // 2, n % 2].set_xticklabels([day.strftime(DATE_FORMAT) for day in days])

        for ax in axs.flat:
            ax.set(xlabel='Datetime', ylabel='Generation')

        plt.savefig(PLOT_FILES_LOC + "model_baseline_%s.png" % (object))

        # Cleanup plotting environment after each loop
        plt.close('all')
        plt.cla()
        plt.clf()

    model_metrics_df.to_csv(BASELINE_RESULTS_FILE, mode='a', index=False, header=not (exists(BASELINE_RESULTS_FILE)))
