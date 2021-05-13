from metadata.getters import get_metadata_label
import pandas as pd
import numpy as np
import random
from scipy.cluster.hierarchy import dendrogram, linkage

np.random.seed(26)
random.seed(26)


def preprocessing_chain(df, norm=None, bred=None):
    '''
    single df is passed,
    norm/batch reduction/dimension reduction done on entire dataset

    key = 'training'
    '''
    df_copy = df.copy()

    metadata, label = get_metadata_label()

    if norm != None:
        print('normalizing ...')
        df_copy = norm(df_copy, metadata)

    if bred != None:
        print('reducing batch effects ...')
        reduce_batch, kwargs = bred
        df_copy = reduce_batch(df_copy, **kwargs, ignore=metadata)

    return df_copy


def hac(matrix_path, features):
    cor = pd.read_pickle(matrix_path)
    cor = cor.loc[features, features].copy().to_numpy()

    ## Function from https://github.com/lichen-lab/MDeep
    def mydist(p1, p2):
        x = int(p1)
        y = int(p2)
        return 1.0 - cor[x, y]

    x = list(range(cor.shape[0]))
    X = np.array(x)

    linked = linkage(np.reshape(X, (len(X), 1)), metric=mydist, method='single')

    result = dendrogram(linked,
                        orientation='top',
                        distance_sort='descending',
                        show_leaf_counts=True)
    indexes = result.get('ivl')
    del result
    del linked
    index = []
    for i, itm in enumerate(indexes):
        index.append(int(itm))

    return index
