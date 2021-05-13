import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
import statistics
import math
import sys
import itertools
import time

np.seterr(over='raise', under="ignore")

def batch_pp(df, covariates, batch_column, ignore):
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
    intercept = [1 for _ in range(X_cov.shape[0])]
    # adding intercept term
    X_cov.insert(0, "intercept", intercept)

    ################################### Build the feature zero inflation matrix ###################################
    # turn numbers to 1 and keep zeroes the way they are
    otu_keys = df.keys().drop(ignore)
    I = df[otu_keys].replace('0.0', False).astype(bool).replace(False, 0).replace(True, 1)

    df_dict = {"X_cov": X_cov,
                "X_batch": X_batch,
                "I": I,
                "Y": df[otu_keys],
                "ignore": df[ignore]}
    return df_dict


def reduce_batch_effects(Y, I, X_cov, X_batch, verbose=False):
    """This function takes in the output of batch_pp and does the feature-wise batch reduction"""

    # INPUT:
    # Y: matrix of feature counts with the columns as features and columns as sample counts as rows
    # I: matrix of feature zero inflation (1s where values are >=1, 0s o.w.)
    # X_cov: covariance matrix (this will give us the betas we need to estimate)
    # X_batch: dummy matrix of batch values
    # OUTPUT:
    # corrected matrix

    # merge the dummy variables for the covariates and also for the batch to get the whole design matrix
    X_mat = pd.concat([X_cov, X_batch], axis=1).astype(float)

    # type conversions and index storing
    Y = Y.astype(float)
    num_beta_cov = X_cov.shape[1]
    num_beta_batch = X_batch.shape[1]
    num_features = len(Y.keys())
    num_samples = Y.shape[0]
    Z = pd.DataFrame(index=Y.index, columns=Y.columns)

    # for each of the features, we will calculate the batch reduction coefficients, then reduce the batch effects
    count = 0
    otu_names = list(Y.keys())
    otu_names = [x for x in otu_names if Y[x][Y[x] > 0].count() > 2]
    sigma_p_store = {}
    beta_params_store = pd.DataFrame(columns=Y.columns, index=X_mat.columns)
    beta_cov_store = pd.DataFrame(columns=Y.columns, index=X_cov.columns)
    beta_batch_store = {}

    start = time.time()
    for p in otu_names:
        # select only the feature as a row
        y_ijp = Y[p]
        y_store = Y[p]  # storing the original column(unchanged)
        I_ijp = I[p].astype(float)
        if (count % 100 == 0 and verbose):
            print("Estimating β_cov, β_batch, and σ_p for feature {}".format(count))

        # --------- Estimate beta_p and beta_batch through OLS regression --------------
        # ignore the keys with zero counts and only fit with non zero samples
        fit_index = list(y_ijp.to_numpy().astype(float).nonzero()[0])
        zero_index = list(set(range(num_samples)) - set(fit_index))
        zero_keys = y_store.keys()[zero_index]

        # use only non zero counts for index to fit our OLS
        y_ijp = y_ijp.iloc[fit_index]
        # y_ijp = y_ijp[fit_index] # PREVIOUS VERSION

        X_design_mat = X_mat.iloc[fit_index, :]
        X_cov_mat = X_cov.iloc[fit_index, :]
        X_batch_mat = X_batch.iloc[fit_index, :]

        # fit ols
        model = sm.OLS(y_ijp, X_design_mat)
        res = model.fit()

        ############# Calculate sigma_p using the standard deviation of previous regression ###########
        residuals = y_ijp - X_cov_mat.dot(res.params[:num_beta_cov])
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
        z_ijp = (y_ijp - X_cov_mat.dot(res.params[:num_beta_cov])) / sigma_hat_p
        Z[p] = z_ijp
        count += 1

        if count % 25 == 0:
            end = time.time()
            print('{}/{} completed in: {}s'.format(count, len(otu_names), round(end - start, 2)))
        # ------------ LOOP END -----------------------------------------------------------------
    end = time.time()
    print('Total OLS time: {}s'.format(round(end - start, 2)))

    Z = Z.fillna(0)
    beta_params_store = beta_params_store.astype(float)
    # return X_mat.dot(beta_params_store)
    estimates = eb_estimator(X_batch, Z, sigma_p=sigma_p_store, X_add=X_cov.dot(beta_cov_store), verbose=verbose)
    return estimates


def eb_estimator(X_batch, Z, sigma_p, X_add, max_itt=6000, verbose=False):
    """This function returns the empirical bayes estimates for gamma_star_p and delta_star_p
    as well as the standerdized OTU counts"""
    # X_batch: Batch effects dummy matrix (n x alpha) matrix
    # Z: Matrix of standerdized data  (n x p ) matrix
    # sigma_p: Vec of OTU variances
    # X_add: matrix to add back after parameter estimation
    # max_itt: Maximum number of iterations until convergence
    # smooth_delta: bool flag for whether or not we replace the 0 values in delta_i by 1

    # Standardized matrix init
    Z_out = pd.DataFrame(index=Z.index, columns=Z.columns)
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

    v_bar = delta_hat.sum(axis=0) / G
    s_bar = ((delta_hat.sub(v_bar) ** 2).sum(axis=0)) / (G - 1)
    lambda_bar = (v_bar + (2 * s_bar)) / (s_bar)
    theta_bar = (v_bar ** 3 + v_bar * s_bar) / (s_bar)

    # iteratively solve for gamma_star_ip and delta_star_ip

    # initialize the keyed matrices
    gamma_star_mat = pd.DataFrame(index=gamma_hat.index, columns=gamma_hat.columns)
    delta_star_mat = pd.DataFrame(index=gamma_hat.index, columns=gamma_hat.columns)

    batches = gamma_hat.keys()
    genes = list(gamma_hat.T.keys())
    genes = [x for x in genes if Z[x].max() != 0]

    start = time.time()
    count = 0
    for i in batches:
        # get individual variables to focus on
        theta_i = theta_bar[i]
        lambda_i = lambda_bar[i]
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
            while ((conv_delta + conv_gamma) > 1e-8):
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
                    raise ValueError("Maximum iteration reached for convergence. Try setting a higher limit")
            if (verbose):
                print("OTU {} on dataset {} Convergence took {} steps".format(p[-15:], i, itt))

            # store found values in the relevant matrices
            gamma_star_mat[i][p] = gamma_star_ip_next
            delta_star_mat[i][p] = delta_star_ip_next

            a = (sigma_p[p] / delta_star_ip_next)
            b = (Z[p][changed_samples] - gamma_star_ip_next)
            c = X_add[p]
            Z_out[p][changed_samples] = (a * b + c)[changed_samples]

        count += 1
        end = time.time()
        print('{}/{} completed in: {}s'.format(count, len(batches), round(end - start, 2)))
        # ------------ LOOP END -----------------------------------------------------------------

    end = time.time()
    print('Total Batch Reduction Parameter Estimation time: {}s'.format(round(end - start, 2)))

    Z_out = Z_out.fillna(0)

    return {"gamma_star": gamma_star_mat,
            "delta_star": delta_star_mat,
            "BR": Z_out}


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


def combat(in_df, covariates, batches, ignore, verbose=False):
    df = in_df.copy()
    for i in range(len(batches.keys())):
        print("Performing ComBat Batch Correction for {}".format(batches[i].upper()))

        df[df.columns.difference(ignore)] = df[df.columns.difference(ignore)]

        t = batch_pp(df, covariates=covariates,batch_column=batches[i], ignore=ignore)
        r = reduce_batch_effects(Y=t['Y'], X_cov=t['X_cov'], I=t['I'], X_batch=t['X_batch'], verbose=verbose)
        try:
            df = pd.concat([r["BR"], t['ignore']], axis=1)
        except:
            print('Error Occurred - returning original data set')
            return ("error", r["BR"])
    return df

