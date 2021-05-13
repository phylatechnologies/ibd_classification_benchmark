import numpy as np
import pandas as pd

def get_x_y(df, metadata_keys, label):
    '''
    Method to split dataset to Features (without metadata) and Response (label)
    Essentially get the X and y matrices

    Input:
        DataFrame: Standard Data Set
        list: list of strings of metadata columns
        str: response label

    Output:
        DataFrame: X - features
        DataFrame: y - labels
    '''
    df = df.sample(frac=1)
    X = df.drop(metadata_keys, axis=1).astype(float)
    y = df[label].astype(float)
    return X, y


def stratify_by_value(df, column, value):
    '''
    Method to split dataset by a list of studies (or any value of a label feature)

    Input:
        DataFrame: Standard Data Set
        str: col - response label in which the value is contained
        str: value - value to split by
    Output:
        DataFrame: Training Set
        DataFrame List: List of Test Sets
    '''
    # get all unique elements from column
    all_values = set(df[column])
    # left out value
    leave_value = {value}

    remaining_values = list(all_values - leave_value)

    remaining_df = df[df[column].isin(remaining_values)]
    left_out_df = df[df[column].isin(leave_value)]

    return remaining_df, left_out_df


def balance_df(df, column):
    '''
    Method to split dataset by a list of studies (or any value of a label feature)

    Input:
        DataFrame: Standard Data Set
        str: stratify_col - column to statify sampling by
        str: label - column to balance
    Output:
        DataFrame: Balanced Training Set
    '''

    # split the df into healthy and sick
    df_copy = df.copy()
    healthy = df_copy[df_copy['diagnosis'] == 0]
    sick = df_copy[df_copy['diagnosis'] == 1]

    # Figure out which has the most samples
    N = min(healthy.shape[0], sick.shape[0])

    # Upsample the datasets to the same number of samples
    healthy_balanced = healthy.groupby(column, group_keys=False).apply(
        lambda x: x.sample(int(np.rint(N * len(x) / len(healthy))), random_state=26))
    sick_balanced = sick.groupby(column, group_keys=False).apply(
        lambda x: x.sample(int(np.rint(N * len(x) / len(sick))), random_state=26))

    # Join the healthy and sick back together
    ret = pd.concat([healthy_balanced, sick_balanced])

    return ret

