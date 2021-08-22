import sys
import pandas as pd
sys.path.append('../data/')


# Returns the values of the info(), describe() functions and the dataframe 'shape' attribute.
def dataframe_all_basic_information(df):
    """

    :param df: Pandas dataframe
    :return: NoneType, pandas.core.frame.DataFrame, , tuple
    """
    df_info = df.info()
    df_describe = df.describe()
    df_shape = df.shape

    return df_info, df_describe, df_shape


# Returns the number of unique labels in the columns of dataframe
def number_unique_labels_in_dataframe_columns(df, columns):
    """

    :param df: Pandas dataframe
    :param columns: a List of column names
    :return: int
    """

    num_unique_labels = df[columns].apply(pd.Series.nunique)

    return num_unique_labels
