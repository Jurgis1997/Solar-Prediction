import pandas as pd


def encode_date(ts: pd.Timestamp) -> float:
    """
    :param ts: current timestamp
    :return: Distance to the summer solstice feature value
    """

    # Calculate approximate solstice days
    solstice_days = {
        "summer_curr_year": pd.Timestamp(year=ts.year, month=6, day=22),
        "summer_foll_year": pd.Timestamp(year=ts.year + 1, month=6, day=22),
        "winter_prev_year": pd.Timestamp(year=ts.year - 1, month=12, day=22),
        "winter_curr_year": pd.Timestamp(year=ts.year, month=12, day=22),
    }

    if solstice_days["winter_prev_year"] < ts < solstice_days["summer_curr_year"]:
        return 1 - (solstice_days["summer_curr_year"] - ts).days / (solstice_days["summer_curr_year"] -
                                                                    solstice_days["winter_prev_year"]).days
    elif solstice_days["summer_curr_year"] < ts < solstice_days["winter_curr_year"]:
        return 1 - (ts - solstice_days["summer_curr_year"]).days / (solstice_days["winter_curr_year"] -
                                                                    solstice_days["summer_curr_year"]).days
    elif solstice_days["winter_curr_year"] < ts < solstice_days["summer_foll_year"]:
        return 1 - (solstice_days["summer_foll_year"] - ts).days / (solstice_days["summer_foll_year"] -
                                                                    solstice_days["winter_curr_year"]).days
    elif ts == solstice_days["summer_curr_year"]:
        return 1
    elif ts == solstice_days["winter_curr_year"]:
        return 0


def encode_sun_activity(datetime_value: pd.Timestamp, sunrise: pd.Timestamp, sunset: pd.Timestamp) -> float:
    """
    :param datetime_value: current timestamp
    :param sunrise: sunrise timestamp of the day
    :param sunset: sunset timestamp of the day
    :return: Solar activity feature value
    """
    if sunrise.hour < datetime_value.hour < sunset.hour:  # Sun is up the whole hour
        return 1

    elif datetime_value.hour < sunrise.hour or datetime_value.hour > sunset.hour:  # Sun is down the whole hour
        return 0

    elif datetime_value.hour == sunrise.hour:  # Sun is up, but not the whole hour (before SUNRISE)
        minutes = (sunrise.second / 60) + sunrise.minute
        return (60 - minutes) / 60

    elif datetime_value.hour == sunset.hour:  # Sun is up, but not the whole hour (before SUNSET)
        minutes = (sunset.second / 60) + sunset.minute
        return minutes / 60


def get_solar_noon(datetime_value: pd.Timestamp, sunrise: pd.Timestamp, noon: pd.Timestamp, sunset: pd.Timestamp) \
        -> float:
    """
    :param datetime_value: current timestamp
    :param sunrise: sunrise timestamp of the day
    :param noon: solar noon timestamp of the day
    :param sunset: sunset timestamp of the day
    :return: Distance to the solar noon feature value
    """
    datetime_value = datetime_value.replace(minute=30)

    if datetime_value < sunrise or datetime_value > sunset:
        return 0

    elif sunrise <= datetime_value <= noon:
        return (datetime_value - sunrise) / (noon - sunrise)

    elif noon <= datetime_value < sunset:
        return 1 - (noon - datetime_value) / (noon - sunset)


def encode_part_of_day(consumption_df: pd.DataFrame, dt_column: str) -> pd.DataFrame:
    """
    :param consumption_df: dataframe with consumption data
    :param dt_column: datetime column name in the dataframe
    :return: dataframe with time of day feature value added
    """
    parts_of_workday_dict = {
        "1/4_workday": (0, 6),
        "2/4_workday": (7, 12),
        "3/4_workday": (13, 16),
        "4/4_workday": (17, 23)
    }

    parts_of_weekend_dict = {
        "1/4_weekend": (0, 6),
        "2/4_weekend": (7, 10),
        "3/4_weekend": (11, 21),
        "4/4_weekend": (22, 23)
    }

    for key, interval in parts_of_workday_dict.items():
        consumption_df[key] = consumption_df.apply(
            lambda x: 1 if x[dt_column].day_of_week < 5 and (interval[0] <= x[dt_column].hour <= interval[1]) else 0,
            axis=1
        )

    for key, interval in parts_of_weekend_dict.items():
        consumption_df[key] = consumption_df.apply(
            lambda x: 1 if x[dt_column].day_of_week >= 5 and (interval[0] <= x[dt_column].hour <= interval[1]) else 0,
            axis=1
        )

    return consumption_df


def encode_part_of_year(consumption_df: pd.DataFrame, dt_column: str) -> pd.DataFrame:
    """
    :param consumption_df: dataframe with consumption data
    :param dt_column: datetime column name in the dataframe
    :return: dataframe with time of year feature values added
    """
    parts_of_year_dict = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11]
    }

    for key in parts_of_year_dict:
        consumption_df[key] = consumption_df.apply(
            lambda x: 1 if x[dt_column].month in parts_of_year_dict[key] else 0, axis=1
        )
    return consumption_df


def generate_previous_values(dataframe: pd.DataFrame, dt_column: str, target: str) -> pd.DataFrame:
    """
    Generates the target value of the last hour and the difference between the last two target values
    :param dataframe:
    :param dt_column: datetime column name in the dataframe
    :param target: target column name in the dataframe
    :return: dataframe with mentioned features added
    """
    df_org = dataframe.copy()
    for num in reversed(range(1, 3)):
        df_temp = df_org[[dt_column, target]].copy(True)
        df_temp[dt_column] = df_temp[dt_column] + pd.DateOffset(hours=num)
        df_temp = df_temp.rename(columns={target: "prev_{}_{}".format(num, target)})
        dataframe = dataframe.merge(df_temp, on=dt_column)

    # Generation difference between the previous and current target values
    dataframe['diff_21_prev'] = dataframe['prev_1_{}'.format(target)] - dataframe['prev_2_{}'.format(target)]
    dataframe = dataframe.drop('prev_2_{}'.format(target), axis=1)
    return dataframe
