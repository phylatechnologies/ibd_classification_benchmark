import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statistics
import math
import sys
import itertools
from argparse import ArgumentParser

from experiment_components.normalizers import normalize_tss

np.seterr(over='warn', under="ignore")


def batch_pp(df, covariates, batch_column, drop_first, ignore):
    """This function takes in a df, the name of the covariate columns, and the batch column
    and it outputs a feature count matrix, feature zero inflation matrix,
    batch dummy matrix (one hot vectors as rows), covariate matrix (concatenated one hot vectors )
    (covariates coefficient matrix [X_ij], batch dummy matrix [X_batch],
    the zero inflation matrix [I_ijk], and count matrix [Y])
    NOTE: this df can be a combination of datasets, or an individual dataset"""

    # df: [dataframe] input with rows as samples and columns as feature counts.
    #                should only have OTU names ,covariates, and batch_column in keyspace
    # covariates: [List] of the covariates to retain and estimate betas for
    # batch_column: [string] column that defines the batches in this dataframe
    # drop_first:   [BOOL] decide whether or not to drop the first col
    # ignore: [List] of column names to ignore

    ################################### Check proper input ###################################
    if (batch_column not in df.keys()):
        raise ValueError("Column name " + str(batch_column) + " not found")

    if (not set(covariates) <= set(df.keys())):
        raise ValueError("Covariate columns not found in dataframe")

    ################################### Turn batch column to one hot vector ###################################
    # note: for all features, batch matrix and covariate matrix will be the same.
    X_batch = pd.get_dummies(df[batch_column], drop_first=False)

    ################################### Turn covariate columns covariate matrix ###################################
    # number of columns is the number of betas to estimate
    X_cov = pd.get_dummies(df[covariates], drop_first=True)
    # intercept=[1 for _ in range(X_cov.shape[0])]
    # adding intercept term
    # X_cov.insert(0, "intercept", intercept)
    print(X_cov.shape[1])
    if (np.linalg.matrix_rank(X_cov) != X_cov.shape[1]):
        raise ValueError("Covariates are counfounded ")

    print(X_batch.shape[1])
    if (np.linalg.matrix_rank(X_batch) != X_batch.shape[1]):
        raise ValueError("Batch is counfounded ")

    ################################### Build the feature zero inflation matrix ###################################
    # turn numbers to 1 and keep zeroes the way they are

    otu_keys = df.keys().drop(ignore)
    I = df[otu_keys].replace('0.0', False).astype(bool).replace(False, 0).replace(True, 1)

    df_dict = {"X_cov": X_cov,
               "X_batch": X_batch,
               "I": I,
               "Y": df[otu_keys],
               "ignore": df[ignore]}

    #print(df_dict)

    return df_dict


def reduce_batch_effects(Y, I, X_cov, X_batch, eb=True, add_mode=0, grand_mean=False, smooth_delta=False,
                         verbose=False):
    """This function takes in the output of batch_pp and does the feature-wise batch reduction

    #INPUT:
    #Y: matrix of feature counts with the columns as features and columns as sample counts as rows
    #I: matrix of feature zero inflation (1s where values are >=1, 0s o.w.)
    #X_cov: covariance matrix (this will give us the betas we need to estimate)
    #X_batch: dummy matrix of batch values
    #eb: boolean flag for the use of empirical bayes estimate rather than SLR for the approximation of batch effect
    #add_mode: int to give the mode for which we recompute the batch-correctd data
    #        {0: add all and sub all,
    #         1: add all and sub cov,
    #         2: add all and sub batch,
    #         3: add batch and sub batch,
    #         4: add batch and sub cov,
    #         5: add cov and sub cov,
    #         6: add cov and sub batch}
    #grand_mean: boolean flag to know if we are calculating Z= (Y- (X * Beta))/sigma or Z= (Y- mean(X * Beta))/sigma (grandmean)
    #smooth_delta: bool flag for whether or not we replace the 0 values in delta_i by 1 (not important if eb is set to false)
    #OUTPUT:
    # corrected matrix

    """

    # merge the dummy variables for the covariates and also for the batch to get the whole design matrix
    X_mat = pd.concat([X_cov, X_batch], axis=1).astype(float)

    if (np.linalg.matrix_rank(X_mat) != X_mat.shape[1]):
        raise ValueError("Covariates and Batch are confounded ")

    # type conversions and index storing
    X_cov = X_cov.astype(float)
    Y = Y.astype(float)
    num_beta_cov = X_cov.shape[1]
    num_beta_batch = X_batch.shape[1]
    num_features = len(Y.keys())
    num_samples = Y.shape[0]
    Z = Y.copy()

    # for each of the features, we will calculate the batch reduction coefficients, then reduce the batch effects
    corr_data = {}
    count = 0
    otu_names = list(Y.keys())
    otu_names = [x for x in otu_names if Y[x][Y[x] > 0].count() > 2]
    sigma_p_store = {}
    beta_params_store = {}
    beta_cov_store = {}
    beta_batch_store = {}
    for p in otu_names:
        # select only the feature as a row
        y_ijp = Y[p]
        y_store = Y[p]  # storing the original column(unchanged)
        y_smooth = y_store.astype(float).replace(0.0, 1.0)
        # original column replacing 0s with 1s so log turns them back to 0
        I_ijp = I[p].astype(float)
        if (count % 100 == 0 and verbose):
            print("Estimating beta_cov, beta_batch, and beta_p for feature {}".format(count))

        ################### Estimate beta_p and beta_batch through OLS regression ####################
        # ignore the keys with zero counts and only fit with non zero samples
        fit_index = list(y_ijp.to_numpy().astype(float).nonzero()[0])
        zero_index = list(set(range(num_samples)) - set(fit_index))
        zero_keys = y_store.keys()[zero_index]

        # use only non zero counts for index to fit our OLS
        y_ijp = y_ijp[fit_index]
        X_design_mat = X_mat.iloc[fit_index, :]
        X_cov_mat = X_cov.iloc[fit_index, :]
        X_batch_mat = X_batch.iloc[fit_index, :]

        # fit ols
        model = sm.OLS(np.log(y_ijp), X_design_mat)
        res = model.fit()
        # print(res.summary())

        ############# Calculate sigma_p using the standard deviation of previous regression ###########
        # using residuals to calculate sigma hat
        residuals = np.log(y_ijp) - X_design_mat.dot(res.params)
        sigma_hat_p = statistics.stdev(residuals)
        # store in feature keyed dictionary of standard deviations
        sigma_p_store[p] = sigma_hat_p

        # separate the beta cov from the beta batch
        beta_params = res.params
        beta_cov = res.params[:num_beta_cov]
        beta_batch = res.params[num_beta_cov:]
        # store list of beta parameters indexed by feature
        beta_params_store[p] = beta_params
        beta_cov_store[p] = beta_cov
        beta_batch_store[p] = beta_batch

        ####################################### Calculate Z_ijp #######################################
        if (grand_mean):
            z_ijp = (np.log(y_ijp) - X_cov_mat.dot(beta_cov).mean()) / sigma_hat_p
        else:
            z_ijp = (np.log(y_ijp) - X_cov_mat.dot(beta_cov)) / sigma_hat_p
        # save the number of samples of this feature that are non zero
        n_batch = z_ijp.shape[0]
        ########################### Estimating the batch effect parameters SLR ############################
        if (not eb):
            # we use the simple linear regression method to estimate the parameters
            model = sm.OLS(z_ijp, X_batch_mat)
            res = model.fit()
            # print(res.summary())
            gamma_star_p = res.params
            residuals = z_ijp - X_batch_mat.dot(res.params)
            delta_star_p = statistics.stdev(residuals)

            # sometimes this gives an overflow in exp.
            # for this try-except block to be effective, we need to set the log level to throw errors instead of warnings
            # not super worried because this is unexplored (according to Siyuan)
            try:
                if (add_mode == 0):
                    X_add = X_mat.dot(beta_params)
                    X_sub = X_mat.dot(beta_params)
                elif (add_mode == 1):
                    X_add = X_mat.dot(beta_params)
                    X_sub = X_cov.astype(float).dot(beta_cov)
                elif (add_mode == 2):
                    X_add = X_mat.dot(beta_params)
                    X_sub = X_batch.astype(float).dot(beta_batch)
                elif (add_mode == 3):
                    X_add = X_batch.astype(float).dot(beta_batch)
                    X_sub = X_batch.astype(float).dot(beta_batch)
                elif (add_mode == 4):
                    X_add = X_batch.astype(float).dot(beta_batch)
                    X_sub = X_cov.astype(float).dot(beta_cov)
                elif (add_mode == 5):
                    X_add = X_cov.astype(float).dot(beta_cov)
                    X_sub = X_cov.astype(float).dot(beta_cov)
                elif (add_mode == 6):
                    X_add = X_cov.astype(float).dot(beta_cov)
                    X_sub = X_batch.astype(float).dot(beta_batch)

                y_adj = I_ijp * np.exp(((np.log(y_smooth) - X_sub - X_batch.astype(float).dot(
                    gamma_star_p) * sigma_hat_p) / delta_star_p) + X_add)

            except FloatingPointError:
                return {"p": p, "y_smooth": y_smooth, "X_mat": X_mat, "beta": beta_params,
                        "X_batch": X_batch, "gamma_star_p": gamma_star_p,
                        "sigma_hat_p": sigma_hat_p, "delta_star_p": delta_star_p,
                        "X_cov": X_cov, "beta_cov": beta_cov}
            Z[p] = y_adj

        else:
            # get the whole Z_ijp row including 0s
            if (grand_mean):
                z_store = I_ijp * (np.log(y_smooth) - (X_cov.dot(beta_cov).astype(float).sum() / n_batch)) / sigma_hat_p
            else:
                z_store = I_ijp * (np.log(y_smooth) - (X_cov.dot(beta_cov).astype(float))) / sigma_hat_p
            Z[p] = z_store

        count += 1
    # if we do not use the EB estimators, we can just return the corrected data as is
    if (not eb):
        return {"Y_tilde": Z}

    ########################### Estimating the batch effect parameters EB ############################
    else:
        beta_params_store = pd.DataFrame.from_dict(beta_params_store)
        beta_cov_store = pd.DataFrame.from_dict(beta_cov_store)
        beta_batch_store = pd.DataFrame.from_dict(beta_batch_store)
        # different matrices to add and substract if we decide to not add all
        if (add_mode == 0):
            X_add = X_mat.dot(beta_params_store)
            X_sub = X_mat.dot(beta_params_store)
        elif (add_mode == 1):
            X_add = X_mat.dot(beta_params_store)
            X_sub = X_cov.astype(float).dot(beta_cov_store)
        elif (add_mode == 2):
            X_add = X_mat.dot(beta_params_store)
            X_sub = X_batch.astype(float).dot(beta_batch_store)
        elif (add_mode == 3):
            X_add = X_batch.astype(float).dot(beta_batch_store)
            X_sub = X_batch.astype(float).dot(beta_batch_store)
        elif (add_mode == 4):
            X_add = X_batch.astype(float).dot(beta_batch_store)
            X_sub = X_cov.astype(float).dot(beta_cov_store)
        elif (add_mode == 5):
            X_add = X_cov.astype(float).dot(beta_cov_store)
            X_sub = X_cov.astype(float).dot(beta_cov_store)
        elif (add_mode == 6):
            X_add = X_cov.astype(float).dot(beta_cov_store)
            X_sub = X_batch.astype(float).dot(beta_batch_store)

        estimates = eb_estimator(X_batch, Z, Y,
                                 beta_X_sub=X_sub,
                                 beta_X_add=X_add,
                                 sigma_p=sigma_p_store, smooth_delta=smooth_delta,
                                 grand_mean=grand_mean, verbose=verbose)
        return estimates


def eb_estimator(X_batch, Z, Y_orig, beta_X_sub, beta_X_add, sigma_p, max_itt=3000, smooth_delta=False,
                 grand_mean=False, verbose=False):
    """This function returns the empirical bayes estimates for gamma_star_p and delta_star_p
    as well as the standerdized OTU counts"""
    # X_batch: Batch effects dummy matrix (n x alpha) matrix
    # Z: Matrix of standerdized data  (n x p ) matrix
    # Y_orig: Original values of countst["Y"]
    # beta_X_sub: Vec to be substraced from counts (before division by delta star)
    #            to normalize (should be X_design_mat.dot(beta_Params))
    # beta_X_add: Vec to be added to normalized counts if add_all from reduce_batch_effects
    #            is set to true this should be X_design_mat.dot(beta_Params) otherwise, X_cov.astype(float).dot(beta_cov)
    # sigma_p: Vec of OTU variances
    # max_itt: Maximum number of iterations until convergence
    # smooth_delta: bool flag for whether or not we replace the 0 values in delta_i by 1

    # Standerdized matrix init
    Y_tilde = Y_orig.copy()
    # number of genes/otus
    G = Z.shape[1]

    # number of samples in each batch
    N = X_batch.sum(axis=0)

    # sample mean for each OTU in each per batch (p X alpha) matrix
    gamma_hat = Z.T.dot(X_batch) / N

    # parameter estimates for batch effect location - gamma
    gamma_bar = gamma_hat.mean(axis=0).astype(float)
    tau_bar = ((gamma_hat.sub(gamma_bar) ** 2).sum(axis=0)) / (G - 1)

    # parameter estimates for batch effect scale - delta (p X alpha) matrix
    delta_hat = (((Z - X_batch.dot(gamma_hat.T)) ** 2).T.dot(X_batch)) / (N - 1)
    if (smooth_delta):
        delta_hat.replace(0, 1)

    v_bar = delta_hat.sum(axis=0) / G
    s_bar = ((delta_hat.sub(v_bar) ** 2).sum(axis=0)) / (G - 1)
    lambda_bar = (v_bar + (2 * s_bar)) / (s_bar)
    theta_bar = (v_bar ** 3 + v_bar * s_bar) / (s_bar)

    # iteratively solve for gamma_star_ip and delta_star_ip

    # initialize the keyed matrices
    gamma_star_mat = gamma_hat.copy()
    delta_star_mat = gamma_hat.copy()

    batches = gamma_hat.keys()
    genes = list(gamma_hat.T.keys())
    genes = [x for x in genes if Y_orig[x][Y_orig[x] > 0].count() > 2]

    for i in batches:
        # get individual variables to focus on
        theta_i = theta_bar[i]
        lambda_i = lambda_bar[i]
        Z_in_batch = 0
        n = N[i]
        tau_i = tau_bar[i]
        gamma_bar_i = gamma_bar[i]

        for p in genes:
            gene_counts_in_batch = X_batch[i] * Z[p]
            gene_counts_in_batch = gene_counts_in_batch[gene_counts_in_batch != 0]

            changed_samples = gene_counts_in_batch.keys()

            gamma_hat_ip = gamma_hat[i][p]

            # initial iteration values
            delta_star_ip_init = delta_hat[i][p]
            gamma_star_ip_init = f_gamma_star_ip(tau_i, gamma_bar_i, gamma_hat_ip, delta_star_ip_init, n)

            # calculate the next step in the iteration
            delta_star_ip_next = f_delta_star_ip(theta_i, lambda_i, gene_counts_in_batch, gamma_star_ip_init, n)
            gamma_star_ip_next = f_gamma_star_ip(tau_i, gamma_bar_i, gamma_hat_ip, delta_star_ip_next, n)

            conv_delta = abs(delta_star_ip_next - delta_star_ip_init)
            conv_gamma = abs(gamma_star_ip_next - gamma_star_ip_init)
            itt = 1
            while ((conv_delta + conv_gamma) > 1e-12):
                # store previous iteration of the values
                delta_star_ip_init = delta_star_ip_next
                gamma_star_ip_init = gamma_star_ip_next

                # take our next "guess" for the values
                delta_star_ip_next = f_delta_star_ip(theta_i, lambda_i, gene_counts_in_batch, gamma_star_ip_init, n)
                gamma_star_ip_next = f_gamma_star_ip(tau_i, gamma_bar_i, gamma_hat_ip, delta_star_ip_init, n)

                # calculate how close we are to convergence
                conv_delta = abs(delta_star_ip_next - delta_star_ip_init)
                conv_gamma = abs(gamma_star_ip_next - gamma_star_ip_init)
                itt += 1
                if (itt == max_itt):
                    raise ValueError("Maximum itteration reached for convergence. Try setting a higher limit")
            if (verbose):
                print("OTU {} on dataset {} Convergence took {} steps".format(p[-15:], i, itt))

            # store found values in the relevant matrices
            gamma_star_mat[i][p] = gamma_star_ip_next

            # normalize the changed values of Y
            if (grand_mean):
                Y_tilde[p][changed_samples] = np.exp(((np.log(Y_orig[p][changed_samples].replace(0, 1)) - (
                        beta_X_sub[p][changed_samples].sum() / n) - gamma_star_ip_next * sigma_p[p])
                                                      / (delta_star_ip_next ** 0.5)) + beta_X_add[p][changed_samples], dtype=np.float128)

            else:
                Y_tilde[p][changed_samples] = np.exp(((np.log(Y_orig[p][changed_samples].replace(0, 1)) - beta_X_sub[p][
                    changed_samples] - gamma_star_ip_next * sigma_p[p])
                                                      / (delta_star_ip_next ** 0.5)) + beta_X_add[p][changed_samples], dtype=np.float128)
    Y_tilde = Y_tilde.div(Y_tilde.sum(axis=1), axis=0).fillna(0)

    return {"gamma_star": gamma_star_mat,
            "delta_star": delta_star_mat,
            "Y_tilde": Y_tilde}


def f_delta_star_ip(theta_bar, lambda_bar, Z_in_batch, gamma_star, n):
    """This is the function to calculate delta star given gamma_star """
    # INPUT
    # theta_bar: theta estimate for batch i (scale estimate for delta star_ip)
    # lambda_bar: lamda estimate for batch i (shape estimate for delta star_ip)
    # Z_in_batch: vector of correctd counts for otu p in in batch o
    # gamma_star: posterior mean for location parameter of OTU p in batch i
    # n: number of samples in batch i
    # OUTPUT
    # delta_star_ip: posterior mean for location parameter of OTU p in batch i
    return (theta_bar + 0.5 * (((Z_in_batch - gamma_star) ** 2).sum())) / ((n / 2) + lambda_bar - 1)


def f_gamma_star_ip(tau_bar, gamma_bar, gamma_hat, delta_star, n):
    """This is the function to calculate gamma star given delta_star"""
    # INPUT
    # tau_bar: tau estimate in batch i
    # gamma_bar: gamma mean estimate for batch i
    # gamma_hat: sample mean for each OTU p in batch i
    # delta_star: posterior mean for scale parameter of OTU p in batch i
    # n: number of samples in batch i
    # OUTPUT
    # gamma_star_ip: posterior mean for location parameter of OTU p in batch i

    return (n * tau_bar * gamma_hat + delta_star * gamma_bar) / (n * tau_bar + delta_star)


def remove_all_batch_effects(df, covariates, drop_first, ignore, batch_order, add_mode, eb, grand_mean=True,
                             save_name="alldata", smooth_delta=True, verbose=False):
    """This function removes all listed batch effects from the dataframe"""
    # df: [dataframe] input with rows as samples and columns as feature counts.
    #                should only have OTU names ,covariates, and batch_column in keyspace
    # covariates: [List] of the covariates to retain and estimate betas for
    # batch_column: [string] column that defines the batches in this dataframe
    # drop_first:   [BOOL] decide whether or not to drop the first col
    # ignore: [List] of column names to ignore
    # batch_order: [Dict] integer indexed dic of the order in which to remove the batches
    #             {0:"sample_type", 1:"sample location"} removes the sample type batch effects first,
    #             then the sample location effects
    # add_mode: [int] to give the mode for which we recompute the batch-corrected data
    #        {0: add all and sub all,
    #         1: add all and sub cov,
    #         2: add all and sub batch,
    #         3: add batch and sub batch,
    #         4: add batch and sub cov,
    #         5: add cov and sub cov,
    #         6: add cov and sub batch}
    # eb: [BOOL] flag for the use of empirical bayes estimate rather than SLR for the approximation of batch effect
    # grand_mean; [BOOL] flag for the standerdization of Z using the mean rather than the actual values
    # smooth_delta: bool flag for whether or not we replace the 0 values in delta_i by 1

    num_batches = len(batch_order.keys())
    # to_ignore=ignore+covariates+list(batch_order.values())

    for batch_col in range(num_batches):
        try:
            batch = batch_order[batch_col]
            print("-- CORRECTING FOR BATCH TYPE {} --".format(batch.upper()))

            t = batch_pp(df, covariates=covariates,
                         batch_column=batch, drop_first=drop_first, ignore=ignore)
            # return t
            r = reduce_batch_effects(Y=t['Y'], X_cov=t['X_cov'], I=t['I'], X_batch=t['X_batch'], eb=eb,
                                     add_mode=add_mode, grand_mean=grand_mean, smooth_delta=smooth_delta,
                                     verbose=verbose)

            df = pd.concat([r["Y_tilde"], t["ignore"]], axis=1)
            print(df[r["Y_tilde"].keys()].isnull().values.any())

        except:

            print(sys.exc_info()[0])
            # identivy all the values that could have made this crash
            return {"prev_df": df, "prev_t": t, "prev_r": r, "batch": batch}

    return df


if __name__ == "__main__":
    path = 'C:/Users/Tima/PycharmProjects/pipeline_v2/datasets/picrust_data/training_picrust_pathway.csv'
    data = pd.read_csv(path, delimiter=',', encoding='utf-8', index_col=0, low_memory=False)
    ignore = ['col_site', 'diagnosis', 'sample_title', 'stool_biopsy', 'studyID', 'uc_cd']
    batch_column = {0: 'col_site'}
    covariates = ['stool_biopsy', 'uc_cd']

    # df, covariates, drop_first, ignore, batch_order, add_mode, eb, grand_mean=True,
    #                              save_name="alldata", smooth_delta=True, verbose=False

    kwargs = {"covariates": ['stool_biopsy', 'uc_cd'],
              "batch_order": {0: 'col_site'}, 'drop_first': False,
              'add_mode': 5, 'eb': True, 'grand_mean': False, 'ignore': ignore}

    data.col_site.fillna(data.studyID, inplace=True)
    data = normalize_tss(data, ignore)
    t = remove_all_batch_effects(data, **kwargs)
    t.to_csv('muffin_test.csv')
