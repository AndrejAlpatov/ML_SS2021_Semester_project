from sklearn.model_selection import train_test_split


# Create predictor and  target variables: X, y
# Create train and test sets: X_train, X_test, y_train, y_test from dataframe
def create_x_y_train_and_test_sets(df, target, predictor, test_size_in, random_state_in):

    """

    :param df: Dataframe to work on
    :param target: Series with target values
    :param predictor: Dataframe with with values to study
    :param test_size_in: The proportion of the dataframe that will be used to assess the accuracy of the model
    :param random_state_in: Parameter random_state of train_test_split()

    :return: np-arrays with target  and predictor variables, train and test sets with target  and predictor variables
    """

    # Create predictor and  target variables: X, y
    y = df[target].values
    x = df[predictor].values

    # Create train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_in, random_state=random_state_in,
                                                        stratify=y)

    return y, x, x_train, x_test, y_train, y_test

