# *************************************************************************
# *********************** GETTER METHODS **********************************
# *************************************************************************
import os
import pandas as pd


def get_studies():
    studies = ['AG', 'CVDF', 'GLS', 'GEVERSC', 'GEVERSM', 'HMP', 'MUC',
               'PRJNA418765', 'PRJNA436359', 'QIITA10184', 'QIITA10342', 'QIITA10567',
               'QIITA1448', 'QIITA2202', 'QIITA550']

    return studies


def get_metadata_label():
    '''
    Method to get metadata labels and the response label from data set
    '''
    metadata = ['col_site', 'diagnosis', 'sample_title', 'stool_biopsy', 'studyID', 'uc_cd']
    label = 'diagnosis'
    return metadata, label


def get_studies_by_label():
    dict_studies = {'zero': ['QIITA10567', 'QIITA10184', 'QIITA2202', 'QIITA550', 'QIITA850',
                             'QIITA10342', 'QIITA1448', 'AG', 'CVDF'],
                    'both': ['GEVERSM', 'MUC', 'GLS', 'HMP', 'GEVERSC'],
                    'one': ['PRJNA418765', 'HSCT', 'PRJNA436359', 'Sprockett']}

    return dict_studies


def get_component(exp_name, component):
    '''
    exp_name:   experiment name, ex: PFAM-LAS-BRM-RF
    components: 'data', 'norm', 'batch', 'model'
    '''
    mapping = {'data': 0,
               'norm': 1,
               'batch': 2,
               'model': 3,
               'norm2': 4}

    exp_split = exp_name.split('-')
    return exp_split[mapping[component]]


def get_pwd(level=0):
    '''
    Method that return a UNIX style string of current
    directory if level = 0. Setting level to -1 and
    below goes down by that many levels.
    '''
    sysname = os.name
    work_dir = os.getcwd()
    if sysname == 'nt':  # if a windows system
        w = work_dir.split('\\')
        if level < 0:
            work_dir = '/'.join(w[:level])  # ~/pipeline_v2
        else:
            work_dir = '/'.join(w)
    else:
        w = work_dir.split('/')
        if level < 0:
            work_dir = '/'.join(w[:level])  # ~/pipeline_v2
        else:
            work_dir = '/'.join(w)

    return work_dir


def save_preprocessed_df(df, exp_name, study=None, full=False):
    if full:
        file = '{}-{}-{}-full'.format(get_component(exp_name, component='data'),
                                      get_component(exp_name, component='norm'),
                                      get_component(exp_name, component='batch'))
    else:
        file = '{}-{}-{}'.format(get_component(exp_name, component='data'),
                                get_component(exp_name, component='norm'),
                                get_component(exp_name, component='batch'))
    if study is not None:
        file = '{}-{}'.format(file, study.upper())

    path = get_pwd() + '/datasets/batch_reduced'
    if not os.path.isdir(path):
        os.makedirs(path)

    df.to_pickle('{}/{}.pkl'.format(path, file))


def get_preprocessed_df(exp_name, study=None, full=False):
    if full:
        file = '{}-{}-{}-full'.format(get_component(exp_name, component='data'),
                                      get_component(exp_name, component='norm'),
                                      get_component(exp_name, component='batch'))
    else:
        file = '{}-{}-{}'.format(get_component(exp_name, component='data'),
                                get_component(exp_name, component='norm'),
                                get_component(exp_name, component='batch'))

    if study is not None:
        file = '{}-{}'.format(file, study.upper())

    path = get_pwd() + '/datasets/batch_reduced'
    if not os.path.isdir(path):
        os.makedirs(path)

    df = pd.read_pickle('{}/{}.pkl'.format(path, file))
    return df


def get_ensemble_names():
    ens_names = ['E_RF_MLP_LR', 'E_RF_MLP_SVC', 'E_RF_SVC_LR', 'E_RF_SVC_SVC', 
                 'E_RF_KNN_LR', 'E_RF_KNN_SVC', 'E_MLP_SVC_LR', 'E_MLP_SVC_SVC', 
                 'E_MLP_KNN_LR', 'E_MLP_KNN_SVC', 'E_SVC_KNN_LR', 'E_SVC_KNN_SVC',
                 'E_RF_MLP_SVC_LR', 'E_RF_MLP_SVC_SVC', 'E_RF_MLP_KNN_LR', 'E_RF_MLP_KNN_SVC', 
                 'E_RF_SVC_KNN_LR', 'E_RF_SVC_KNN_SVC', 'E_MLP_SVC_KNN_LR', 'E_MLP_SVC_KNN_SVC']
    return ens_names
