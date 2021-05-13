import pandas as pd
import numpy as np
import random
from sklearn.metrics import f1_score, roc_curve, accuracy_score, auc, matthews_corrcoef, confusion_matrix
np.random.seed(26)
random.seed(26)

def get_performance(y_df, index_name):
    '''
    Used when there is no std to get.
    '''
    N = 100000
    # First groups by studyID, we want to upsample each study individually
    # Next, groupby the combinations of true (0,1) and predicted values (0,1)
    # We will upsample so that the ratios of these groups remains the same
    sampled_y = y_df.groupby(['studyID'], group_keys=False).apply(lambda x: x.groupby(['true', 'pred'], group_keys=False).apply(lambda y: y.sample(int(np.rint(N * len(y) / len(x))), replace=True, random_state=26)))                    

    y_true = sampled_y['true']
    y_pred = sampled_y['pred']

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    try:
        auc_s = auc(fpr, tpr)
    except:
        # happens when either row in confusion matrix is empty
        auc_s = 0.5

    try:
        mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    except FloatingPointError:
        # happens when either row in confusion matrix is empty
        mcc = 0

    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, normalize='all', labels=[0, 1]).ravel()

    scores = pd.DataFrame({'TP': tp,
                           'TN': tn,
                           'FP': fp,
                           'FN': fn,
                           'F1': f1,
                           'Accuracy': acc,
                           'AUC': auc_s,
                           'MCC': mcc}, index=[index_name]).round(4)
    scores = scores * 100                           
    return scores
from sklearn.metrics import f1_score, roc_curve, accuracy_score, auc, matthews_corrcoef, confusion_matrix

