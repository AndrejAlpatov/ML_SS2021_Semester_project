# Group the dataframe by the same values in the specified columns and
# apply functions to defined columns of the dataframe.
def grouping_dataframe_(df, columns, columns_for_function_application, functions):

    """

    :param df: Pandas dataframe to be grouped
    :param columns: columns by which the dataframe will be grouped as a List of column names
    :param columns_for_function_application: columns to which the functions will be applied as a List of column names
    :param functions: functions to be applied as a List
    :return: Pandas dataframe
    """

    df_grouped_by_i_event_and_id_plane = df.groupby(columns)[columns_for_function_application].agg(functions)
    # Reset the index of the DataFrame
    df_grouped_by_i_event_and_id_plane = df_grouped_by_i_event_and_id_plane.reset_index()
    # Where there was a grouping of one element, the value of the standard deviation is equal to NaN. Zero replacement.
    df_grouped_by_i_event_and_id_plane = df_grouped_by_i_event_and_id_plane.fillna(0)

    return df_grouped_by_i_event_and_id_plane

