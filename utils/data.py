import numpy as np
import math
import os
import pandas as pd
import json


def convert_size(size_bytes):
    '''
    Converts dataframe size from bytes to more readable format.
    Arguments:
        size_bytes (int, numpy.int64): obtained for example with df.memory_usage(deep=True).sum()
    Returns:
        string of formated size
    '''
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def balance_dataset():
    # TODO
    #UNDERSAMPLING
    X_train['y']=y_train
    reaction = X_train[X_train.y==1]
    no_reaction = X_train[X_train.y==0].sample(2*len(reaction))
    X_train=reaction.append(no_reaction)
    X_train = X_train.sample(frac=1).reset_index(drop=True)
    y_train = X_train.y
    X_train = X_train.drop(columns=['y'])
    
def split_data(X, y, test_size=0.2, random_state=1):
    from sklearn.model_selection import train_test_split 
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_tr, X_te, y_tr, y_te

def decrease_memory_occupation(df, verbose=True):
    '''
    converting data types to decrease the memory occupation (float64 to float32, int64 to int32).
    This function modifies the provided argument inplace.
    
    Arguments:
        df (pandas.DataFrame): 
    Returns:
        pandas.DataFrame
    '''
    if verbose:
        print("memory occupation before data type conversion:", convert_size(df.memory_usage(deep=True).sum()))
        
    for c, dtype in zip(df.columns, df.dtypes):
        if dtype == np.float64:
            df[c] = df[c].astype(np.float32) 
        elif dtype == np.int64:
            df[c] = df[c].astype(np.int32)
            
    if verbose:
        print("memory occupation after data type conversion:", convert_size(df.memory_usage(deep=True).sum()))
    
    return df

class DataLoader:
    def __init__(self, config_path, numeric_column_names=None):
        assert os.path.exists(config_path), "You must provide a valid json configuration file!"
        with open(config_path) as config_file:
            data = json.load(config_file)

        self.base_path = data['base_path']
        self.data = None
        #self.numeric_column_names = numeric_column_names 
        self.numeric_column_names = ['age', 'campaign', 'duration', 'pdays', 
                   'previous', 'emp.var.rate', 'cons.price.idx', 
                   'cons.conf.idx', 'euribor3m', 'nr.employed']

        
    def load_csv_data(self, filename="bank-additional-full.csv", test_filename=None, load_first_n=None, decrease_memory=True):
        data_folder = os.path.join(self.base_path, "data")
        input_csv_path = os.path.join(data_folder, filename)
        
        if not os.path.exists(input_csv_path):
            print("there is no file with this path:", input_csv_path)
        
        if test_filename is not None:
            test_csv_path = os.path.join(data_folder, test_filename)
            if not os.path.exists(test_csv_path):
                print("there is no file with this path:", test_csv_path)
            print("loading test data from:", test_csv_path)
            self.test_data = pd.read_csv(test_csv_path, sep=";", nrows=load_first_n)
        else:
            self.test_data = None
            print("all of the data will be loaded into training set")
        
        print("loading data from:", input_csv_path)
        data = pd.read_csv(input_csv_path, sep=";", nrows=load_first_n)
        
        if self.test_data is not None:
            self.data = pd.merge(data, self.test_data, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        else:
            self.data = data
        
        names = list(data.columns.values)
        self.target_column = names[-1]
        self.feature_names = names[:-1]
        
        if decrease_memory:
            decrease_memory_occupation(self.data)
            if self.test_data is not None:
                decrease_memory_occupation(self.test_data)
            
        if self.numeric_column_names is not None:
            self.categorical_column_names = list(set(self.feature_names) - set(self.numeric_column_names))
            
        return self.data
        #X = data.loc[:, input_names]
        #y = data.loc[:, target_column]
        #return X, y
    
    def get_data(self, include_target_column=True):
        if include_target_column:
            return self.data
        return self.data[self.feature_names]
    
    def get_test_data(self, include_target_column=True):
        if include_target_column:
            return self.test_data
        return self.test_data[self.feature_names]
    
    def get_features_and_target(self, df, target_column_name=None):
        if target_column_name is None:
            if self.target_column is not None:
                target_column_name = self.target_column
            else:
                print("target column name not defined")
                return None
        return (df[self.feature_names], df[target_column_name])
    
    def get_numeric_data(self, include_target_column=True, numeric_column_names=None, test_data=False):
        if test_data is True:
            print("Returning testing data")
            data = self.test_data
        else:
            print("Returning training data")
            data = self.data
        if numeric_column_names:
            if include_target_column:
                return data[self.numeric_column_names].join(data[self.target_column])
            return data[self.numeric_column_names]
        
        elif self.numeric_column_names is not None:
            if include_target_column:
                return data[self.numeric_column_names].join(data[self.target_column])
            return data[self.numeric_column_names]
        
        else:
            print("numerical columns not defined")
    
    def get_categorical_data(self, include_target_column=True, categorical_column_names=None, test_data=False):
        if test_data is True:
            print("Returning testing data")
            data = self.test_data
        else:
            print("Returning training data")
            data = self.data
        
        if categorical_column_names:
            if include_target_column:
                return data[self.categorical_column_names].join(data[self.target_column])
            return data[self.categorical_column_names]
        
        elif self.categorical_column_names is not None:
            if include_target_column:
                return data[self.categorical_column_names].join(data[self.target_column])
            return data[self.categorical_column_names]
        
        else:
            print("categorical_column_names not defined")
    