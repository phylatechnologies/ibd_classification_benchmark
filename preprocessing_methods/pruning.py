import pandas as pd
import numpy as np


def features_to_retain(df):
    # Takes in a dataframe where cols = n_features (including metadata) and rows/index = m_samples
    # Finds features with a count greater than 1e-5 and adds them to a list
    feature_list = []
    df = df.apply(pd.to_numeric, errors='coerce')
    for colname in df.columns.values:
        # print(colname)
        colNum = df[colname].tolist()

        if np.sum(colNum) != 0:
            relAb = colNum / np.sum(colNum)

            if (np.sum(x > .00005 for x in relAb) / len(relAb)) >= .1:
                feature_list.append(colname)

    return feature_list


def rareFeatureRemove(df_in, studyid_col_name, metadata_col_names=[]):
    # df is a pandas dataframe where cols = n_features (including metadata) and rows/index = m_samples
    # studyid_col_name is a string noting the name of studyID column which we will use to split the dataframe
    # metadata_col_names is an optional parameter, marking the names of the metadata columns in the dataframe,
    # so that we can split the dataframe and perform our calculation on a dataframe without string values

    print("Removing rare features and samples with low read count...")

    df = df_in.copy()

    initial_shape = df.shape

    dict_of_studies = dict(iter(df.groupby(studyid_col_name, axis=0)))
    metadata_df = df.copy()[metadata_col_names]
    df = df.drop(metadata_col_names, axis=1)

    list_of_features_to_retain = []

    for sk in dict_of_studies.keys():
        temp_df = dict_of_studies[sk]
        temp_df = temp_df.drop(metadata_col_names, axis=1)

        # now we aim to find only features with relative abundance greater than 10e-5 in at least 10% of samples
        list_of_features_to_retain.extend(features_to_retain(temp_df))

    # Some features might have repeated in our list, just making sure they are unique
    unique_features_to_retain = list(set(list_of_features_to_retain))

    df = df[unique_features_to_retain]  # Retain only features with certain abundance
    final_df = pd.concat([df, metadata_df], axis=1)  # We concatenate the metadata back to the main dataframe

    final_shape = final_df.shape
    print("Number of rare features removed: " + str(initial_shape[1] - final_shape[1]))

    return final_df


def check_health(df, verbose=False):
    '''
    Method that takes in a DataFrame, and tells you about the health;
    1. the number of columns that have Nans
    2. the number of columns that have all 0s

    if verbose == True, tell you the column/row that has all 0s or Nans
    '''
    return None
