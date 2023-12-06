from config import *
from data_preparation import *
from models_baseline import test_baseline
from models_prophet import train_test_prophet
from models_nn import train_test_nn

if __name__ == '__main__':
    weather_df = prepare_weather_data(pd.read_csv(WEATHER_DATA, parse_dates=DATETIME_COLS), LOCATION)
    consumption_df = pd.read_csv(CONSUMPTION_DATA, parse_dates=['datetime_local'])

    # Dictionary for storing dataframes of the objects
    dataframes_dict = {
        "solar_30k": pd.read_csv(SOLAR_30K_DATA, parse_dates=DATETIME_COLS),
        "solar_5k": pd.read_csv(SOLAR_5K_DATA, parse_dates=DATETIME_COLS),
        "1": consumption_df.loc[consumption_df['object_id'] == 1].drop('object_id', axis=1),
        "5": consumption_df.loc[consumption_df['object_id'] == 5].drop('object_id', axis=1)
    }

    # Dictionary for info about fetched data
    fetch_data_dict = {
        'Data type': ["Weather"],
        'First timestamp': [weather_df['datetime_local'].min()],
        'Last timestamp': [weather_df['datetime_local'].max()],
        'Number of rows': [len(weather_df.index)]
    }

    # Dictionary for info about prepared data
    prep_data_dict = {
        'Object': [],
        'First timestamp': [],
        'Last timestamp': [],
        'Number of rows': []
    }

    for obj, dataframe in dataframes_dict.items():
        fetch_data_dict['Data type'].append(obj)
        fetch_data_dict['First timestamp'].append(dataframe['datetime_local'].min())
        fetch_data_dict['Last timestamp'].append(dataframe['datetime_local'].max())
        fetch_data_dict['Number of rows'].append(len(dataframe.index))

        if obj in GENERATION_OBJECTS:
            dataframes_dict[obj] = generate_previous_values(dataframe, 'datetime_local', GEN_TARGET_COLUMN)
            dataframes_dict[obj] = dataframes_dict[obj].drop('datetime_unix', axis=1)

        elif obj in CONSUMPTION_OBJECTS:
            dataframes_dict[obj] = generate_previous_values(dataframe, 'datetime_local', CON_TARGET_COLUMN)

        else:
            raise ValueError('Unknown target!')

        dataframes_dict[obj] = pd.merge(weather_df, dataframes_dict[obj], on='datetime_local')

        prep_data_dict['Object'].append(obj)
        prep_data_dict['First timestamp'].append(dataframes_dict[obj]['datetime_local'].min())
        prep_data_dict['Last timestamp'].append(dataframes_dict[obj]['datetime_local'].max())
        prep_data_dict['Number of rows'].append(len(dataframes_dict[obj].index))

    print("Following data fetched:")
    print(pd.DataFrame.from_dict(fetch_data_dict).to_string(), '\n')

    print("Following data prepared:")
    print(pd.DataFrame.from_dict(prep_data_dict).to_string(), '\n')

    test_baseline()
    train_test_prophet(dataframes_dict)
    train_test_nn(dataframes_dict)
