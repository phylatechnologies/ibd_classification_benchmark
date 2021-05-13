import os
import pandas as pd
import numpy as np
from metadata.getters import get_pwd
from analysis.evaluation_methods import get_performance
import random
np.random.seed(26)
random.seed(26)
def get_sample_info(dataset):
    '''
    :param exp_name: experiment string
    :return: unpickled data set
    '''

    data_file = '{}.pkl'.format(dataset)
    data_path = '{}/datasets/{}'.format(get_pwd(), data_file)
    df = pd.read_pickle(data_path)
    info = df[['diagnosis','studyID']]

    return info

if (__name__=="__main__"):

    res_path = get_pwd() + '/results'
    res = os.listdir(res_path)


    for e in res:
        output = '{}/metrics'.format(get_pwd())
        if not os.path.isdir(output):
            os.mkdir(output)
        
        exp_name = e.split('.')[0]
        dataset = exp_name.split('-')[0]

        if not os.path.isfile('{}/{}.csv'.format(output, exp_name)):
            
            info = get_sample_info(dataset)

            file = '{}/{}'.format(res_path, e)

            y_pred = pd.read_csv(file, index_col=0)
            y_pred = np.round(y_pred)

            y = pd.concat([info, y_pred], axis = 1, sort=False)
            y.columns = ['true', 'studyID', 'pred']

            metrics = get_performance(y_df=y, index_name=exp_name)

            print(metrics)
            metrics.to_csv('{}/{}.csv'.format(output, exp_name))
