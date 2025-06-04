from components.load import get_project_root
import pandas as pd

root = get_project_root()

without_emissions = f"{root}/src/no_outliers.csv"
with_emissions = f"{root}/src/with_outliers.csv"


def get_datasets_X():
    datasets = {
        'with-out emissions':  pd.read_csv(without_emissions).drop(columns=['Unnamed: 0']),
        'with emissions': pd.read_csv(with_emissions).drop(columns=['Unnamed: 0']),
    }

    return datasets

def get_true_Y():
    datasets = {
        'with-out emissions': pd.read_csv(f"{root}/src/true_no_outliers.csv").drop(columns=['Unnamed: 0']),
        'with emissions': pd.read_csv(f"{root}/src/true_with_outliers.csv").drop(columns=['Unnamed: 0'])
    }
    return datasets