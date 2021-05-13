# *************************************************************************
# *********************** STATISTICAL MODELS ******************************
# *************************************************************************

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Conv1D, Dropout, Activation, LocallyConnected1D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier


# n = number of jobs (CPUs)
n = 4
# s = random_state
s = 26

unknown = None  # Have to set it up inside LODO.py, depending on the dataset


# KERAS MODELS

def mlp_creator(dim_size, batch_size):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(dim_size,)))
    model.add(Dropout(0.5, seed=26))
    i = 0
    while i < 2:
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5, seed=26))
        i += 1
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def MDeep_creator(dim_size, batch_size):
    model = Sequential()
    model.add(Conv1D(filters=64, strides=4, kernel_size=8, padding='same', name="conv1", input_shape=(dim_size, 1)))
    model.add(Activation('tanh'))
    model.add(Conv1D(filters=64, strides=4, kernel_size=8, padding='same', name="conv2"))
    model.add(Activation('tanh'))
    model.add(Conv1D(filters=32, strides=4, kernel_size=8, padding='same', name="conv3"))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(units=64, name="fc1",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                    bias_initializer=tf.keras.initializers.Constant(value=0.1)))
    model.add(Dropout(0.5, seed=26))
    model.add(Activation('tanh'))
    model.add(Dense(units=8, name="fc2",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                    bias_initializer=tf.keras.initializers.Constant(value=0.1)))
    model.add(Dropout(0.5, seed=26))
    model.add(Activation('tanh'))
    model.add(Dense(units=1, activation="sigmoid", name="fc3"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

call_back = [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10,
                           verbose=0, mode='auto', baseline=None,
                           restore_best_weights=True)]

# Deep Neural Network
deep_params = {'init_params': {'batch_size': 16, 'dim_size': unknown},
              'fit_params': {'epochs': 100, 'validation_split': 0.05, 'shuffle': True, 'batch_size': 16, 'callbacks': call_back},
              'ensemble_params': {'epochs': 30, 'validation_split': 0.05, 'shuffle': True, 'batch_size': 16}}

# Logistic Regression
lr_params = {'solver': 'sag',
             'class_weight': 'balanced',
             'max_iter': 10000,
             'random_state': s,
             'n_jobs': n}

# Linear SVC
svc_params = {'tol': 10e-6,
              'class_weight': 'balanced',
              'random_state': s,
              'max_iter': 100000}

# Random Forest
rf_params = {'criterion': 'gini',
             'n_estimators': 500,
             'max_features': 'sqrt',
             'class_weight': 'balanced',
             'random_state': s,
             'n_jobs': n}

# Bernoulli Naive Bayes
bnb_params = {'binarize': 0.0}

# K-Nearest Neighbours
knn_params = {'n_neighbors': 6,
              'weights': 'distance',
              'metric': 'manhattan',
              'n_jobs': n}

# Stochastic Gradient Descent with Modified Huber Loss
sgd_params = {'loss': 'modified_huber',
              'penalty': 'l2',
              'tol': 10e-5,
              'random_state': s,
              'max_iter': 10000}

xgb_params = {'n_estimators': 500,
              'eval_metric': 'logloss',
             'random_state': s,
             'n_jobs': n,
             'use_label_encoder':False}


model_dict = {'LR': {'model': LogisticRegression, 'params': lr_params},
              'LRRBF': {'model': LogisticRegression, 'params': lr_params},
              'SVC': {'model': SVC, 'params': svc_params},
              'BNB': {'model': BernoulliNB, 'params': bnb_params},
              'RF': {'model': RandomForestClassifier, 'params': rf_params},
              'KNN': {'model': KNeighborsClassifier, 'params': knn_params},
              'SGD': {'model': SGDClassifier, 'params': sgd_params},
              'MLP': {'model': mlp_creator, 'params': deep_params},  
              'MDeep': {'model': MDeep_creator, 'params': deep_params},
              'XGB': {'model': XGBClassifier, 'params': xgb_params},
}
