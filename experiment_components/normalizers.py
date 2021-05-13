from sklearn.preprocessing import StandardScaler

try:
    from skbio.stats.composition import clr, multiplicative_replacement, ilr
except:
    print('no skbio')

import numpy as np
import pandas as pd

from metadata.getters import get_pwd, get_metadata_label, get_component


# *************************************************************************
# ******************* DATA NORMALIZATION METHODS **************************
# *************************************************************************

def normalize_tss(df_in, ignore):
    df = df_in.copy()
    df_store = df[ignore]
    df = df.drop(ignore, inplace=False, axis=1).astype(float)
    df = df.div(df.sum(axis=1), axis=0).astype(float)
    df[ignore] = df_store
    return df


def normalize_clr(df_in, ignore):
    # shape should be (n_samples, n_features)
    df = df_in.copy()
    df_store = df[ignore]
    df = df.drop(ignore, inplace=False, axis=1).astype(float)
    save_cols = df.columns
    save_keys = df.index
    df = multiplicative_replacement(df)
    df = clr(df)
    df = pd.DataFrame(data=df, index=save_keys, columns=save_cols)
    df[ignore] = df_store
    return df

def normalize_ilr(df_in, ignore):
    # shape should be (n_samples, n_features)
    df = df_in.copy()
    df_store = df[ignore]
    df = df.drop(ignore, inplace=False, axis=1).astype(float)
    save_keys = df.index
    df = multiplicative_replacement(df)
    df = ilr(df)
    df = pd.DataFrame(data=df, index=save_keys)
    df[ignore] = df_store
    return df


# normalize_log and normalized_std are used in LAS
def normalize_log(df_in, ignore):
    df = df_in.copy()
    df_store = df[ignore]
    df = df.drop(ignore, inplace=False, axis=1).astype(float)
    df = df.replace(0, 1)
    df = np.log(df).astype(float)
    df[ignore] = df_store
    return df


def normalize_std(df_in, ignore):
    df = df_in.copy()
    df_store = df[ignore]
    df = df.drop(ignore, inplace=False, axis=1).astype(float)
    scaler = StandardScaler().fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index).astype(float)
    df[ignore] = df_store
    return df


def normalize_las(df_in, ignore):
    '''
    LAS: Log transform followed by Standardization Scaling
    '''
    log_df = normalize_log(df_in, ignore)
    log_std_df = normalize_std(log_df, ignore)
    return log_std_df


def normalize_vst(exp_name, study):
    '''
    from: gsutil cp gs://phylab1/publication_datasets/vst ./pipeline3/datasets/

    VST is done only with BRN and BRZC
    '''
    data_name = get_component(exp_name, component='data')
    # ./pipeline3/datasets/vst/COG/
    path = get_pwd() + '/datasets/vst/{}'.format(data_name)        
    
    BATCH = get_component(exp_name, component='batch')
    if BATCH in ['BRMCC','BRMNCS','BRMCCL']:
        train = '{}-{}-training-left-out-study-{}.pkl'.format(data_name, BATCH, study)
        test = '{}-{}-testing-left-out-study-{}.pkl'.format(data_name, BATCH, study)

        df_train = pd.read_pickle('{}/{}'.format(path, train)).set_index('run_accession')
        df_test = pd.read_pickle('{}/{}'.format(path, test)).set_index('run_accession')
        
        return df_train, df_test
    else:    
        # ./pipeline3/datasets/vst/COG/COG-training-left-out-study-MUC.pkl
        train = '{}-training-left-out-study-{}.pkl'.format(data_name, study)
        test = '{}-testing-left-out-study-{}.pkl'.format(data_name, study)

        df_train = pd.read_pickle('{}/{}'.format(path, train)).set_index('run_accession')
        df_test = pd.read_pickle('{}/{}'.format(path, test)).set_index('run_accession')
        df = pd.concat([df_train, df_test], axis=0)
        return df

def normalize_arcsinsqrt(df_in, ignore):
    df = df_in.copy()
    df = normalize_tss(df, ignore)
    df_store = df[ignore]
    df = df.drop(ignore, inplace=False, axis=1).astype(float)
    df = np.sqrt(df).astype(float)
    df = np.arcsin(df).astype(float)
    df[ignore] = df_store

    return df

def normalize_logtss(df_in, ignore):
    df = df_in.copy()
    df = normalize_tss(df, ignore)
    df_store = df[ignore]
    df = df.drop(ignore, inplace=False, axis=1).astype(float)
    df = df.replace(0, 1)
    df = np.log(df).astype(float)
    df[ignore] = df_store
    return df


def normalize_not(df_in, ignore):
    df = df_in.copy()
    return df


normalizers = {"TSS": normalize_tss,
               "LOG": normalize_logtss,
               "STD": normalize_std,
               "CLR": normalize_clr,
               "LAS": normalize_las,
               "NOT": normalize_not,
               "VST": normalize_vst,
               "ILR": normalize_ilr,
               "ARS": normalize_arcsinsqrt}  # add "VST": normalize_vst,
