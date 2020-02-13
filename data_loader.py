import pandas as pd
import os
import json


class DataLoader:
    def __init__(self, config_path):
        assert os.path.exists(config_path), "You must provide a valid json configuration file!"
        with open(config_path) as config_file:
            data = json.load(config_file)

        self.base_path = data['base_path']

    def load_csv_data(self, filename="bank.csv", load_first_n=None):
        data_folder = os.path.join(self.base_path, "data")
        input_csv_path = os.path.join(data_folder, filename)
        if not os.path.exists(input_csv_path):
            print("there is no file with this path:", input_csv_path)

        print("loading data from:",input_csv_path)
        data = pd.read_csv(input_csv_path, sep=";", nrows=load_first_n)
        names = list(data.columns.values)
        target_column = names[-1]
        input_names = names[:-1]
        X = data.loc[:, input_names]
        y = data.loc[:, target_column]
        return X, y