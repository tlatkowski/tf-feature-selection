import pandas as pd


def read(file_name):
    data = pd.read_csv(file_name, sep='\t', header=None, index_col=0).T
    return data.as_matrix()
