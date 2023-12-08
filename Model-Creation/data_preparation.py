import pandas as pd
import numpy as np

from pvlib.location import Location
from feature_encodings import *
from scaler import CustomScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

_DATETIME_COLUMN = 'datetime_local'

_COLS_TO_DROP = [
    'datetime_unix', 'datetime_local', 'date_only', 'sunrise', 'sunset', 'transit', 'WindDirection10m',
    'WindSpeed10m', 'Dni', 'Dhi', 'Ebh', 'GtiTracking'
]

_COLS_TO_DROP_CONSUMPTION = [
    'Azimuth', 'CloudOpacity', 'DewpointTemp', 'Ghi', 'GtiFixedTilt', 'PrecipitableWater', 'RelativeHumidity',
    'SnowWater', 'SurfacePressure', 'Zenith', 'AlbedoDaily', 'encoded_noon'
]

_COLS_TO_DROP_GENERATION = ['%d/4_%s' % (i, day) for i in range(1, 5) for day in ['workday', 'weekend']] + \
                            ['Winter', 'Spring', 'Summer', 'Autumn']


class Dataset:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.num_features = None


class Dataset_Prophet:
    def __init__(self):
        self.df_train = None
        self.df_test = None

class Dataset_Keras:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.num_features = None


def prepare_weather_data(df: pd.DataFrame, location: Location):
    """
    Adds new features and encodes existing ones
    :param df: dataframe with historical weather data
    :param location: location for which the solar transition points will be calculated
    :return: prepared weather dataframe
    """
    # Sunrise, sunset and solar noon calculation
    df['date_only'] = df[_DATETIME_COLUMN].dt.normalize()
    dt_index = pd.DatetimeIndex(df['date_only']).tz_localize(tz=location.tz).drop_duplicates()
    sun_points_df = location.get_sun_rise_set_transit(dt_index, method='spa').reset_index()

    # Remove timezone mark from the timestamps
    for column in list(sun_points_df):
        sun_points_df[column] = sun_points_df[column].dt.tz_localize(None)

    # Merge calculated dataframe with the main one
    df = df.merge(sun_points_df, on='date_only', how='left')

    df['sun_activity'] = df.apply(lambda x: encode_sun_activity(x[_DATETIME_COLUMN], x['sunrise'], x['sunset']), axis=1)
    df['encoded_noon'] = df.apply(
        lambda x: get_solar_noon(x[_DATETIME_COLUMN], x['sunrise'], x['transit'], x['sunset']), axis=1)
    df['encoded_date'] = df.apply(lambda x: encode_date(x['date_only']), axis=1)

    # Zenith and azimuth angles encoding (The Sun’s position in the sky)
    df['Azimuth'] = df.apply(lambda x: abs(x['Azimuth']), axis=1)
    df['Zenith'] = df.apply(lambda x: abs(x['Zenith']), axis=1)

    # Wind speed and direction encoding
    df['Wind_encoded'] = df.apply(lambda x: abs(180 - x['WindDirection10m']) * x['WindSpeed10m'], axis=1)

    df = encode_part_of_day(df, _DATETIME_COLUMN)
    df = encode_part_of_year(df, _DATETIME_COLUMN)

    return df


def get_nn_dataset(df: pd.DataFrame, testing_intervals: list, sequence_len: int, target: str) -> Dataset:
    """
    Removes unnecessary data from the dataframe, scales the data with the Min-Max scaler and divides the data
    into the testing and training datasets
    :param df: prepared dataframe
    :param testing_intervals: list of date tuples indicating intervals for the testing dataset
    :param sequence_len: sequence length used in LSTM and GRU models
    :param target: target value of the dataset to prepare
    :return: Dataset class object with data prepared for training and testing
    """
    columns_to_drop = _COLS_TO_DROP.copy()

    if target == 'consumption':
        columns_to_drop.extend(_COLS_TO_DROP_CONSUMPTION)

    elif target == 'generation':
        columns_to_drop.extend(_COLS_TO_DROP_GENERATION)

    scaler = CustomScaler()
    df = scaler.fit_transform(df, columns_to_drop)

    x_train, y_train = [], []
    x_test, y_test = [], []

    # Distribute the data using the sliding window method
    columns_to_drop.append(target)
    print("drop col", columns_to_drop)
    for end_dt in df[_DATETIME_COLUMN].tolist():
        start_dt = end_dt - pd.DateOffset(hours=sequence_len - 1)
        current_seq_df = df.loc[(df[_DATETIME_COLUMN] >= start_dt) & (df[_DATETIME_COLUMN] <= end_dt)]

        if len(current_seq_df.index) == sequence_len:
            is_testing = False
            current_date = end_dt.replace(hour=0, minute=0, second=0)

            for start_date, end_date in testing_intervals:
                if start_date <= current_date <= end_date:
                    is_testing = True

            if is_testing:
                x_test.append(current_seq_df.drop(columns_to_drop, axis=1).to_numpy())
                y_test.append(current_seq_df[target].iloc[-1])

            else:
                x_train.append(current_seq_df.drop(columns_to_drop, axis=1).to_numpy())
                y_train.append(current_seq_df[target].iloc[-1])

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    if sequence_len == 1:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[2])

    print("Dataframe len: ", len(df.index))
    print("x train shape: ", x_train.shape)
    print("y train shape: ", y_train.shape)
    print("x test  shape: ", x_test.shape)
    print("y test  shape: ", y_test.shape)

    dataset = Dataset()
    dataset.x_train, dataset.y_train = x_train, y_train
    dataset.x_test, dataset.y_test = x_test, y_test
    dataset.num_features = x_train.shape[1] if sequence_len == 1 else x_train.shape[2]

    return dataset


def get_prophet_dataset(df: pd.DataFrame, testing_intervals: list, target: str) -> Dataset_Prophet:
    columns_to_drop = _COLS_TO_DROP.copy()

    if target == 'consumption':
        columns_to_drop.extend(_COLS_TO_DROP_CONSUMPTION)

    if target == 'generation':
        columns_to_drop.extend(_COLS_TO_DROP_GENERATION)

    columns_to_drop.remove(_DATETIME_COLUMN)

    df = df.drop(columns_to_drop, axis=1)
    df['is_testing'] = False

    for start_date, end_date in testing_intervals:
        df.loc[(df[_DATETIME_COLUMN].dt.date >= start_date) & (
                df[_DATETIME_COLUMN].dt.date <= end_date), 'is_testing'] = True

    df['cap'] = df[target].max()
    df['floor'] = 0
    df = df.rename(columns={target: 'y', _DATETIME_COLUMN: 'ds'})

    print(df.info())
    dataset = Dataset_Prophet()
    dataset.df_train = df.loc[df['is_testing'] == False]
    dataset.df_test = df.loc[df['is_testing']]
    return dataset


def get_keras_dataset(df: pd.DataFrame, testing_intervals: list, sequence_len: int, target: str) -> Dataset_Keras:
    columns_to_drop = _COLS_TO_DROP.copy()

    # Add any additional columns to drop for Keras dataset preparation

    scaler = MinMaxScaler()
    df = scaler.fit_transform(df, columns_to_drop)

    x, y = [], []

    # Distribute the data using the sliding window method
    columns_to_drop.append(target)
    print("drop col", columns_to_drop)
    for end_dt in df[_DATETIME_COLUMN].tolist():
        start_dt = end_dt - pd.DateOffset(hours=sequence_len - 1)
        current_seq_df = df.loc[(df[_DATETIME_COLUMN] >= start_dt) & (df[_DATETIME_COLUMN] <= end_dt)]

        if len(current_seq_df.index) == sequence_len:
            is_testing = False
            current_date = end_dt.replace(hour=0, minute=0, second=0)

            for start_date, end_date in testing_intervals:
                if start_date <= current_date <= end_date:
                    is_testing = True

            if is_testing:
                x.append(current_seq_df.drop(columns_to_drop, axis=1).to_numpy())
                y.append(current_seq_df[target].iloc[-1])

    x = np.asarray(x)
    y = np.asarray(y)

    if sequence_len == 1:
        x = x.reshape(x.shape[0], x.shape[2])

    print("Dataframe len: ", len(df.index))
    print("x shape: ", x.shape)
    print("y shape: ", y.shape)

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    dataset = Dataset_Keras()
    dataset.x_train, dataset.x_test = x_train, x_test
    dataset.y_train, dataset.y_test = y_train, y_test
    dataset.num_features = x_train.shape[1] if sequence_len == 1 else x_train.shape[2]

    return dataset