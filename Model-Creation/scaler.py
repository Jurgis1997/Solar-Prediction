import pickle

import pandas as pd
from datetime import datetime


class CustomScaler:
    """
    Custom Min-Max scaler with an intention to use it in a C++ environment
    """
    def __init__(self):
        self.scaling_dictionary = {}
        self.range_min, self.range_max = 0, 0

    def min_max_scale(self, value, min_val, max_val):
        std = (value - min_val) / (max_val - min_val)
        return std * (self.range_max - self.range_min) + self.range_min

    def fit_transform(self, df: pd.DataFrame, columns_skip: list, feature_range=(0, 1)):
        self.range_min, self.range_max = feature_range
        for column in df.columns:
            if column not in columns_skip:
                min_val, max_val = df[column].min(), df[column].max()
                df[column] = df.apply(lambda x: self.min_max_scale(x[column], min_val, max_val), axis=1)
                self.scaling_dictionary[column] = [min_val, max_val]

        print("Scaler transformed:")
        self.print_scaler()
        return df

    def fit(self, df: pd.DataFrame):
        for column, feature_range in self.scaling_dictionary.items():
            min_val, max_val = feature_range
            df[column] = df.apply(lambda x: self.min_max_scale(x[column], min_val, max_val), axis=1)

        return df

    def print_scaler(self):
        for column, feature_range in self.scaling_dictionary.items():
            min_val, max_val = feature_range
            print("%20s: %10.5f / %10.5f" % (column, min_val, max_val))

    def print_cpp(self, target_column: str):
        features_type_str, scaler_str = "", ""

        for column, feature_range in self.scaling_dictionary.items():
            min_val, max_val = feature_range

            if column != target_column:
                if not features_type_str:
                    features_type_str += "typedef enum {\n" + "    " + column.upper() + " = 0"
                    scaler_str += "struct Scaling scaler[NUMBER_OF_FEATURES] = { \n"
                    scaler_str += '    {{{feature}, {min}, {max}}}'.format(feature=column.upper(), min=min_val, max=max_val)

                else:
                    features_type_str += ",\n    " + column.upper()
                    scaler_str += ',\n    {{{feature}, {min}, {max}}}'.format(feature=column.upper(), min=min_val, max=max_val)

        features_type_str += '\n} Features;'
        scaler_str += '\n};'
        print("#define NUMBER_OF_FEATURES {num}".format(num=len(self.scaling_dictionary) - 1))
        print(features_type_str)
        print(scaler_str)

    def save_to_file(self):
        path = "scalers/" + datetime.now().strftime("%m-%d_%H-%M-%S") + ".pkl"
        temp_dict = self.scaling_dictionary.copy()
        temp_dict['range'] = [self.range_min, self.range_max]

        file = open(path, "wb")
        pickle.dump(temp_dict, file)
        file.close()

        print("Scaler saved to file -", path)

    def load_from_file(self, path: str):
        file = open(path, "rb")
        self.scaling_dictionary = pickle.load(file)
        file.close()

        self.range_min, self.range_max = self.scaling_dictionary['range']
        del self.scaling_dictionary['range']

        print("Scaler loaded from file: ")
        self.print_scaler()


