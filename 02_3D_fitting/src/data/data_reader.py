import pandas as pd
from pathlib import Path

RELATIVE_PATH = './2020-02-04_18-30_Proton_230MeV_Head-Phantom.csv.gz'
NROWS = 1000
COLUMNS = ('posX', 'posY', 'posZ', 'edep', 'parentID', 'eventID', 'PDGEncoding')


def read(path=RELATIVE_PATH, nrows=NROWS, usecols=COLUMNS):
    _path = Path(__file__).parent / path
    df = pd.read_csv(_path, nrows=nrows, usecols=usecols)
    return df
