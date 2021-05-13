import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from experiment_components.classifiers import model_dict
from experiment_components.ensemble import get_ensemble_model
from experiment_components.normalizers import normalizers
from experiment_components.batch_reducers import batch_reducers

from preprocessing_methods.splitting import stratify_by_value, get_x_y, balance_df
from preprocessing_methods.preprocessing import preprocessing_chain, hac
from metadata.getters import get_pwd, get_studies, get_metadata_label, get_component
from metadata.getters import get_preprocessed_df, save_preprocessed_df, get_ensemble_names

from preprocessing_methods.pruning import rareFeatureRemove

import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)

# *************************************************************************
# ********************* CROSS VALIDATION METHODS **************************
# *************************************************************************

def lodo(df_in, exp_name): 
    """
    LODO CV
    """
    lodo_dir = get_pwd()  # should be ./pipeline3/
    metadata, label = get_metadata_label()

    # ------------- MODEL SELECTION ------------------------
    model_name = get_component(exp_name, component='model')
    if model_name in get_ensemble_names():
        pass  # we have to initialize dim when we have it
    else:
        model_components = model_dict[model_name]
        model = model_components['model']  # model function
        init_params = model_components['params']  # model parameters

    # ---------- PREPROCESSING METHODS ---------------------
    # Most of the time, we preprocess the dataset before
    NORM = get_component(exp_name, component='norm')
    norm = normalizers[NORM]

    BATCH = get_component(exp_name, component='batch')
    batch = batch_reducers[BATCH]

    # ------------- LODO CV ---------------------------------
    col = 'studyID'
    studies = get_studies()
    # DF to store study predictions at each iteration
    col_y = [exp_name]
    y_pred = pd.DataFrame(columns=col_y)
    # -------------------------------------------------------

    for study in studies:
        print('- {} -'.format(study))
     
        if BATCH in ['BRMCCL', 'BRMCSL']:
            # In this part, we also want to save the splits by study
            try:
                df_train = get_preprocessed_df(exp_name, study)
                _, df_test = stratify_by_value(df_in, col, study)
  
            except FileNotFoundError:
                df_train, df_test = stratify_by_value(df_in, col, study)

                # --------- PRUNE RARE FEATURES --------------------
                df_train = rareFeatureRemove(df_train, 'studyID', metadata)
                df_train = preprocessing_chain(df_train, norm=norm, bred=batch)
                save_preprocessed_df(df_train, exp_name, study)

            # ---- get columns that are pruned for test set ----
            pruned_cols = df_train.columns
            df_test = df_test[pruned_cols]
            # --------------------------------------------------           
            df_test = preprocessing_chain(df_test, norm=norm, bred=None)
            # --------------------------------------------------

        elif BATCH in ['BRMCCS', 'BRMNCS', 'BRMNSS']:
                        # In this part, we also want to save the splits by study
            try:
                df_train = get_preprocessed_df(exp_name, study)
                df_batch = get_preprocessed_df(exp_name, study, full=True)  
  
            except FileNotFoundError:
                df_train, _ = stratify_by_value(df_in, col, study)

                # --------- PRUNE RARE FEATURES --------------------
                df_train = rareFeatureRemove(df_train, 'studyID', metadata)
                df_train = preprocessing_chain(df_train, norm=norm, bred=batch)
                save_preprocessed_df(df_train, exp_name, study)

                # ---- get columns that are pruned for test set ----
                pruned_cols = df_train.columns
                df_combined = df_in[pruned_cols]
                ## Batch reduce the full dataset
                df_batch = preprocessing_chain(df_combined, norm=norm, bred=batch)
                save_preprocessed_df(df_batch, exp_name, study, full=True)       
            # --------------------------------------------------           
            _, df_test = stratify_by_value(df_batch, col, study)

        else:  # in [ BRMCC, BRMNC, BRZ, BRZL, BRN ]
            try:
                if NORM == 'VST':  # VST only done for BRN and BRZC
                    df_combined = get_preprocessed_df(exp_name, study)
                    df_train, df_test = stratify_by_value(df_combined, col, study)
                else:
                    df_combined = get_preprocessed_df(exp_name, study)
                    df_train, df_test = stratify_by_value(df_combined, col, study)

            except FileNotFoundError:
                # VST only done for BRN and BRZC
                if NORM == 'VST':
                    df_combined = norm(exp_name, study)
                    df_combined = preprocessing_chain(df_combined, norm=None, bred=batch)
                    df_train, df_test = stratify_by_value(df_combined, col, study)
                else:
                    df_train, df_test = stratify_by_value(df_in, col, study)
                    df_train = rareFeatureRemove(df_train, 'studyID', metadata)     

                    # ---- get columns that are pruned for test set ----
                    pruned_cols = df_train.columns
                    df_test = df_test[pruned_cols]
                    # --------------------------------------------------     

                    # Assemble since we want to batch reduce together
                    df_combined = pd.concat([df_train, df_test], axis=0)
                    df_combined = preprocessing_chain(df_combined, norm=norm, bred=batch)
                    save_preprocessed_df(df_combined, exp_name, study=study)
                    # Re-split again
                    df_train, df_test = stratify_by_value(df_combined, col, study)
                    # -------------------------------------------------------

        if BATCH in ['BRMCCS', 'BRMNCS', 'BRMCC', 'BRMCCL', 'BRMNSS', 'BRMCSL']:
            NORM2 = get_component(exp_name, component='norm2')
            norm2 = normalizers[NORM2]
            if NORM2 == 'VST':
                df_train, df_test = norm2(exp_name, study)
            else: 
                if NORM2 in ['ILR','CLR']:
                    if df_train.loc[df_train.drop(metadata,axis=1).sum(axis=1) == 0].shape[0] > 0:
                        df_train.loc[df_train.drop(metadata,axis=1).sum(axis=1) == 0, df_train.columns.difference(metadata)] = 1 / df_train.drop(metadata,axis=1).shape[1]               
                    if df_test.loc[df_test.drop(metadata,axis=1).sum(axis=1) == 0].shape[0] > 0:
                        df_test.loc[df_test.drop(metadata,axis=1).sum(axis=1) == 0, df_test.columns.difference(metadata)] = 1 / df_test.drop(metadata,axis=1).shape[1]               
  
                df_train = preprocessing_chain(df_train, norm=norm2, bred=None)
                df_test = preprocessing_chain(df_test, norm=norm2, bred=None)

        df_train = balance_df(df_train, ['col_site', 'uc_cd', 'stool_biopsy'])

        X_train, y_train = get_x_y(df_train, metadata, label)
        X_test, y_test = get_x_y(df_test, metadata, label)
        X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]
        X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]  
        
        if model_name in ['MDeep', 'MDeep_lcl']:
            ordered_index = hac('MDeep/patristic_distance_correlation.pkl', X_train.columns)
            X_train = X_train.iloc[:, ordered_index]
            X_test = X_test.iloc[:, ordered_index] 

        # If the model is from keras, it has extra parameters;
        # to initialize: init_params
        # to fit: fit_params
        # we have to update the init_params accordingly
        if model_name in ['MDeep', 'MLP']:
            init_params = model_components['params']['init_params']
            init_params['dim_size'] = X_train.shape[1]

        if model_name in get_ensemble_names():
            model, kwargs = get_ensemble_model(exp_name, dim=X_train.shape[1])
            init_params = kwargs['init_params']
            ensemble_fit_params = kwargs['fit_params']

        classifier = model(**init_params)
        # ------------------------------------------------------

        # Taking care of special case for Keras ML model fit_params
        if model_name in ['MDeep', 'MDeep_lcl']:
            fit_params = model_components['params']['fit_params']
            try:
                classifier.fit(X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, **fit_params)
            except:
                fit_params = {'epochs': 100, 'validation_split': 0.05, 'shuffle': True, 'batch_size': 16, 
                              'callbacks': [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10,
                               verbose=0, mode='auto', baseline=None, restore_best_weights=False)]}
                classifier.fit(X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, **fit_params)

            yp = classifier.predict(X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1)))

        elif model_name in ['MLP']:
            fit_params = model_components['params']['fit_params']
            try:
                classifier.fit(X_train, y_train, **fit_params)
            except:
                fit_params = {'epochs': 100, 'validation_split': 0.05, 'shuffle': True, 'batch_size': 16, 
                              'callbacks': [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10,
                               verbose=0, mode='auto', baseline=None, restore_best_weights=False)]}
                classifier.fit(X_train, y_train, **fit_params)
            yp = classifier.predict(X_test)

        elif model_name in get_ensemble_names():
            classifier.fit(X_train, y_train, **ensemble_fit_params)
            yp = classifier.predict(X_test)

        elif model_name == 'LRRBF':
            X = pd.concat([X_train, X_test])
            gamma = 1 / (X.shape[1] * X.to_numpy().var())
            X = rbf_kernel(X, gamma=gamma)
            classifier.fit(X.loc[y_train.index], y_train)
            yp = classifier.predict(X.loc[y_test.index])

        elif model_name == 'XGB':
            classifier.fit(X_train, y_train)
            yp = classifier.predict(X_test)

        else:
            classifier.fit(X_train, y_train)
            yp = classifier.predict(X_test)

        # test set prediction vector
        yp_append = pd.DataFrame({exp_name: yp.ravel()}, index=X_test.index)
        y_pred = pd.concat([y_pred, yp_append])

        del classifier
        # -------------- Loop end -----------------------------------

    # ------- STORING PREDICTION VECTOR ----------------------
    pred_file = '{}/results/{}.csv'.format(lodo_dir, exp_name)
    y_pred.to_csv(pred_file)
    # --------------------------------------------------------

    return y_pred
