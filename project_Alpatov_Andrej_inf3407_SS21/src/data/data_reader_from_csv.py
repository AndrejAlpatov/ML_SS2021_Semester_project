import pandas as pd
from pathlib import Path

RELATIVE_PATH = './pctdata.csv'
NROWS = 220138
COLUMNS = ('id_plane', 'id_x', 'id_y', 'i_event', 'i_time_stamp')


# Read dataset and create a dataframe from it
def read_data_set(path=RELATIVE_PATH, nrows_in=NROWS, usecols_in=COLUMNS):
    """

    :param path: relative path to file
    :param nrows_in: number of rows in CSV-file
    :param usecols_in: names of columns for dataframe
    :return: Pandas dataframe
    """
    _path = Path(__file__).parent / path
    df = pd.read_csv(_path, nrows=nrows_in, usecols=usecols_in)
    return df

