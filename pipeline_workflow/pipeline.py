import os
import pandas as pd
from metadata.getters import get_pwd, get_component, get_metadata_label

from pipeline_workflow.LODO import lodo


# *************************************************************************
# *********************** PROCESSING METHODS ******************************
# *************************************************************************

# pipeline_dir = get_pwd() should be ./pipeline3/pipeline_workflow
pipeline_dir = get_pwd()  # should be ./pipeline3/


def run_experiments(experiments_list):
    '''
    Method to run experiments and the way to process
    Input:
        list: experiments_list
    Output:
        list: list of results dataframes
    '''

    result_dict = {}
    for exp_name in experiments_list:
        result = get_results(exp_name)
        result_dict[exp_name] = result

    print('DONE RUNNNING EXPERIMENTS')
    return result_dict


def get_results(exp_name):
    """
    Method to see if the results are present,
    if not, process said experiment and store it.
    """
    results_path = '{}/results'.format(pipeline_dir)
    file = "{}/{}.csv".format(results_path, exp_name)

    # Make the results directory
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    try:
        res = pd.read_csv(file, delimiter=',', encoding='utf-8', index_col=0, low_memory=False)
        print('SKIPPING {} --- results already present ...'.format(exp_name))
        return res

    except FileNotFoundError:
        print('CREATING {} RESULTS'.format(exp_name))
        res = create_results(exp_name)
        return res


def create_results(exp_name):
    '''
    Runs the process for each experiment
    '''
    df = get_dataset(exp_name)

    # --------- PERFORM LODO CV -----------------------
    # Everything is done by the exp_name recipe
    # Just making sure pipeline3/results directory is made
    res_dir = pipeline_dir+'/results'
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

    res = lodo(df, exp_name)
    # -------------------------------------------------
    return res


def get_dataset(exp_name):
    '''
    :param exp_name: experiment string
    :return: unpickled data set
    '''
    data_name = get_component(exp_name, component='data')
    data_file = '{}.pkl'.format(data_name)
    data_path = '{}/datasets/{}'.format(pipeline_dir, data_file)
    df = pd.read_pickle(data_path)
    return df
