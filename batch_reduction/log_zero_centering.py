import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
import statistics
import math
import sys
import itertools

np.seterr(all='raise')


def batch_pp(df, batch_column, ignore):
    """This function takes in a df and the batch column
    and it outputs a feature count matrix
    batch dummy matrix (one hot vectors as rows)
    ( batch dummy matrix [X_batch], and count matrix [Y])
    NOTE: this df can be a combination of datasets, or an individual dataset"""

    # df: [dataframe] input with rows as samples and columns as feature counts.
    #                should only have OTU names ,covariates, and batch_column in keyspace
    # batch_column: [string] column that defines the batches in this dataframe
    # ignore: [List] of column names to ignore

    ################################### Check proper input ###################################
    if (batch_column not in df.keys()):
        raise ValueError("Column name " + str(batch_column) + " not found")

    ################################### Turn batch column to one hot vector ###################################
    # note: for all features, batch matrix and covariate matrix will be the same.
    X_batch = pd.get_dummies(df[batch_column], drop_first=False)

    ################################### Build the feature zero inflation matrix ###################################
    # turn numbers to 1 and keep zeroes the way they are
    otu_keys = df.keys().drop([batch_column] + ignore)
    return {"X_batch": X_batch,
            "Y": np.log(df[otu_keys].replace(0, 1)),
            "ignore": df[ignore]}


def center(Y, X_batch):
    """This function takes in a dataframe of counts and the corresponding dummy batch matrix and zero centers the 
    counts for each batch"""

    batch_sizes = X_batch.sum()

    mean_batch_count = (X_batch.T.dot(Y)).T.div(batch_sizes)

    y_bar = mean_batch_count.mean(axis=1)
    zero_centeted = np.exp(Y - X_batch.dot(mean_batch_count.T))

    return zero_centeted


def log_zero_center(df, batch_column, ignore):
    """
    df: (samples x features) otu/genus count matrix
    batch_column: str of column that has the batch id
    ignore: list of columns that are not genus/otu counts 
    """
    t = batch_pp(df, batch_column, ignore)
    out = center(t['Y'], t['X_batch'])
    out[ignore] = t['ignore']
    return out

