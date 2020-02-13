import sklearn as sk
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

from data_loader import DataLoader

def main():
    dl = DataLoader("../config.json")
    inputs, labels = dl.load_csv_data(load_first_n=10)
    inputs.head()
    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(inputs, labels)

if __name__ == "__main__":
    main()