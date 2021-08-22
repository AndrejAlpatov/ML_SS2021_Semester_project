import pandas as pd


# Counting the frequency of values in the i_event column and adding the result to the dataframe
def add_column_with_count_of_elements(df, column_name, new_column_name):

    """

    :param df: Pandas dataframe to be expand
    :param column_name: Name of the column, the frequency of occurrence of values of which will be counted
    :param new_column_name: The name of the column where the calculated values will be saved
    :return: Pandas dataframe with new column
    """

    # Series of i_event values and frequency of occurrences in the dataframe
    store_counts = df[column_name].value_counts()
    # Convert series to dataframe
    df_temp = pd.DataFrame({column_name: store_counts.index, new_column_name:store_counts.values})
    # Joining a column with a number of values to a dataframe
    return df.merge(df_temp, on=column_name)
