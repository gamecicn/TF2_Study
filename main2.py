import pandas as pd
import numpy as np
import glob
import pickle
from sklearn.preprocessing import MinMaxScaler, Normalizer
from time import strftime, localtime
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from time import time
from multiprocessing import Pool, cpu_count
import importlib
from os import path
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV
################################################


# Dev Setting
XGB_SEARCH_ITER = 1
XGB_SEARCH_CV = 2
JOB = 4



MODEL_CONF = [

['sklearn.ensemble', 'AdaBoostClassifier', {}],
['sklearn.ensemble', 'AdaBoostClassifier', {"n_estimators" : 80, "learning_rate" : 0.5, "random_state" : 10}],

['sklearn.ensemble', 'BaggingClassifier', {}],
['sklearn.ensemble', 'BaggingClassifier', {"n_estimators" : 20, "max_samples" : 5, "max_features" : 10, "random_state" : 10}],

['sklearn.ensemble', 'ExtraTreesClassifier', {}],
['sklearn.ensemble', 'ExtraTreesClassifier', {"n_estimators" : 100, "max_depth" : 10, "min_samples_split" : 5, "min_samples_leaf" : 2}],

['sklearn.linear_model', 'RidgeClassifier', {}],
['sklearn.linear_model', 'RidgeClassifier', {"alpha" : 0.8, "max_iter" : 1000}],

['sklearn.linear_model', 'SGDClassifier', {}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.0005, "l1_ratio" : 0.01, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],

['sklearn.linear_model', 'SGDClassifier', {}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.0005, "l1_ratio" : 0.01, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],

['sklearn.naive_bayes', 'BernoulliNB', {}],
['sklearn.naive_bayes', 'BernoulliNB', {"alpha" : 0.8, "binarize" : 0}],

['sklearn.tree', 'DecisionTreeClassifier', {}],
['sklearn.tree', 'DecisionTreeClassifier', {"min_samples_split" : 4, "max_depth" : 10}],

['sklearn.svm', 'SVC', {}],
['sklearn.svm', 'SVC', {'degree': 5, 'max_iter': 1000, 'C': 1.0, 'coef0' : 0.1}],

['sklearn.svm', 'NuSVC', {}],
['sklearn.svm', 'NuSVC', {'nu': 0.8, 'degree': 5, 'C': 1.0, 'coef0' : 0.1}],


]

# Real Setting
'''
XGB_SEARCH_ITER = 100
XGB_SEARCH_CV = 2
JOB = cpu_count()
'''

MODEL_CONF_TEST = [

['sklearn.ensemble', 'AdaBoostClassifier', {}],
['sklearn.ensemble', 'AdaBoostClassifier', {"n_estimators" : 80, "learning_rate"  : 0.8, "random_state" : 10}],
['sklearn.ensemble', 'AdaBoostClassifier', {"n_estimators" : 100, "learning_rate" : 0.5, "random_state" : 10}],
['sklearn.ensemble', 'AdaBoostClassifier', {"n_estimators" : 120, "learning_rate" : 0.5, "random_state" : 10}],
['sklearn.ensemble', 'AdaBoostClassifier', {"n_estimators" : 150, "learning_rate" : 0.5, "random_state" : 10}],
['sklearn.ensemble', 'AdaBoostClassifier', {"n_estimators" : 80, "learning_rate"  : 0.1, "random_state" : 10}],
['sklearn.ensemble', 'AdaBoostClassifier', {"n_estimators" : 100, "learning_rate" : 0.2, "random_state" : 10}],
['sklearn.ensemble', 'AdaBoostClassifier', {"n_estimators" : 120, "learning_rate" : 0.3, "random_state" : 10}],
['sklearn.ensemble', 'AdaBoostClassifier', {"n_estimators" : 150, "learning_rate" : 0.5, "random_state" : 10}],

['sklearn.ensemble', 'BaggingClassifier', {}],
['sklearn.ensemble', 'BaggingClassifier', {"n_estimators" : 20, "max_samples" : 5, "max_features" : 20, "random_state" : 10}],
['sklearn.ensemble', 'BaggingClassifier', {"n_estimators" : 20, "max_samples" : 5, "max_features" : 10, "random_state" : 10}],
['sklearn.ensemble', 'BaggingClassifier', {"n_estimators" : 30, "max_samples" : 10, "max_features" : 30, "random_state" : 10}],
['sklearn.ensemble', 'BaggingClassifier', {"n_estimators" : 30, "max_samples" : 5, "max_features" : 10, "random_state" : 10}],
['sklearn.ensemble', 'BaggingClassifier', {"n_estimators" : 50, "max_samples" : 10, "max_features" : 10, "random_state" : 10}],
['sklearn.ensemble', 'BaggingClassifier', {"n_estimators" : 50, "max_samples" : 5, "max_features" : 30, "random_state" : 10}],


['sklearn.ensemble', 'ExtraTreesClassifier', {}],
['sklearn.ensemble', 'ExtraTreesClassifier', {"n_estimators" : 100, "max_depth" : 10, "min_samples_split" : 5, "min_samples_leaf" : 2}],
['sklearn.ensemble', 'ExtraTreesClassifier', {"n_estimators" : 100, "max_depth" : 10, "min_samples_split" : 20, "min_samples_leaf" : 5}],
['sklearn.ensemble', 'ExtraTreesClassifier', {"n_estimators" : 100, "max_depth" : 10, "min_samples_split" : 20, "min_samples_leaf" : 2}],
['sklearn.ensemble', 'ExtraTreesClassifier', {"n_estimators" : 50, "max_depth" : 50, "min_samples_split" : 20, "min_samples_leaf" : 2}],
['sklearn.ensemble', 'ExtraTreesClassifier', {"n_estimators" : 50, "max_depth" : 50, "min_samples_split" : 5, "min_samples_leaf" : 5}],
['sklearn.ensemble', 'ExtraTreesClassifier', {"n_estimators" : 50, "max_depth" : 50, "min_samples_split" : 5, "min_samples_leaf" : 2}],


['sklearn.linear_model', 'RidgeClassifier', {}],
['sklearn.linear_model', 'RidgeClassifier', {"alpha" : 0.8, "max_iter" : 1000}],
['sklearn.linear_model', 'RidgeClassifier', {"alpha" : 0.7, "max_iter" : 1000}],
['sklearn.linear_model', 'RidgeClassifier', {"alpha" : 0.6, "max_iter" : 1000}],
['sklearn.linear_model', 'RidgeClassifier', {"alpha" : 0.5, "max_iter" : 1000}],
['sklearn.linear_model', 'RidgeClassifier', {"alpha" : 0.4, "max_iter" : 1000}],
['sklearn.linear_model', 'RidgeClassifier', {"alpha" : 0.3, "max_iter" : 1000}],
['sklearn.linear_model', 'RidgeClassifier', {"alpha" : 0.2, "max_iter" : 1000}],
['sklearn.linear_model', 'RidgeClassifier', {"alpha" : 0.1, "max_iter" : 1000}],


['sklearn.linear_model', 'SGDClassifier', {}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.0005, "l1_ratio" : 0.01, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.001, "l1_ratio" : 0.01, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.0015, "l1_ratio" : 0.02, "max_iter": 1000, "epsilon" :0.2, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.0020, "l1_ratio" : 0.02, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.0025, "l1_ratio" : 0.02, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.003, "l1_ratio" : 0.05, "max_iter": 1000, "epsilon" :0.2, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.004, "l1_ratio" : 0.01, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.005, "l1_ratio" : 0.01, "max_iter": 1000, "epsilon" :0.2, "eta0" : 0.1, "n_iter_no_change" : 5}],


['sklearn.linear_model', 'SGDClassifier', {}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.0005, "l1_ratio" : 0.01, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.001, "l1_ratio" : 0.01, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.002, "l1_ratio" : 0.02, "max_iter": 1000, "epsilon" :0.2, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.005, "l1_ratio" : 0.03, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.01, "l1_ratio" : 0.04, "max_iter": 1000, "epsilon" :0.2, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.015, "l1_ratio" : 0.02, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.2, "l1_ratio" : 0.01, "max_iter": 1000, "epsilon" :0.2, "eta0" : 0.1, "n_iter_no_change" : 5}],
['sklearn.linear_model', 'SGDClassifier', {"alpha" : 0.3, "l1_ratio" : 0.01, "max_iter": 1000, "epsilon" :0.1, "eta0" : 0.1, "n_iter_no_change" : 5}],



['sklearn.naive_bayes', 'BernoulliNB', {}],
['sklearn.naive_bayes', 'BernoulliNB', {"alpha" : 0.8, "binarize" : 0}],
['sklearn.naive_bayes', 'BernoulliNB', {"alpha" : 0.7, "binarize" : 0}],
['sklearn.naive_bayes', 'BernoulliNB', {"alpha" : 0.6, "binarize" : 0}],
['sklearn.naive_bayes', 'BernoulliNB', {"alpha" : 0.5, "binarize" : 0}],
['sklearn.naive_bayes', 'BernoulliNB', {"alpha" : 0.4, "binarize" : 0}],
['sklearn.naive_bayes', 'BernoulliNB', {"alpha" : 0.3, "binarize" : 0}],
['sklearn.naive_bayes', 'BernoulliNB', {"alpha" : 0.2, "binarize" : 0}],
['sklearn.naive_bayes', 'BernoulliNB', {"alpha" : 0.1, "binarize" : 0}],

['sklearn.tree', 'DecisionTreeClassifier', {}],
['sklearn.tree', 'DecisionTreeClassifier', {"min_samples_split" : 4, "max_depth" : 10}],
['sklearn.tree', 'DecisionTreeClassifier', {"min_samples_split" : 4, "max_depth" : 20}],
['sklearn.tree', 'DecisionTreeClassifier', {"min_samples_split" : 4, "max_depth" : 30}],
['sklearn.tree', 'DecisionTreeClassifier', {"min_samples_split" : 6, "max_depth" : 40}],
['sklearn.tree', 'DecisionTreeClassifier', {"min_samples_split" : 6, "max_depth" : 50}],
['sklearn.tree', 'DecisionTreeClassifier', {"min_samples_split" : 6, "max_depth" : 60}],
['sklearn.tree', 'DecisionTreeClassifier', {"min_samples_split" : 2, "max_depth" : 70}],
['sklearn.tree', 'DecisionTreeClassifier', {"min_samples_split" : 2, "max_depth" : 80}],
['sklearn.tree', 'DecisionTreeClassifier', {"min_samples_split" : 2, "max_depth" : 90}],

['sklearn.svm', 'SVC', {}],
['sklearn.svm', 'SVC', {'degree': 5, 'max_iter': 1000, 'C': 1.0, 'coef0' : 0.1}],
['sklearn.svm', 'SVC', {'degree': 5, 'max_iter': 1000, 'C': 1.0, 'coef0' : 0.1}],
['sklearn.svm', 'SVC', {'degree': 15, 'max_iter': 1000, 'C': 2.0, 'coef0' : 0.2}],
['sklearn.svm', 'SVC', {'degree': 15, 'max_iter': 1000, 'C': 2.0, 'coef0' : 0.2}],
['sklearn.svm', 'SVC', {'degree': 15, 'max_iter': 1000, 'C': 2.0, 'coef0' : 0.2}],
['sklearn.svm', 'SVC', {'degree': 25, 'max_iter': 1000, 'C': 1.0, 'coef0' : 0.1}],
['sklearn.svm', 'SVC', {'degree': 25, 'max_iter': 1000, 'C': 1.0, 'coef0' : 0.1}],

['sklearn.svm', 'NuSVC', {}],
['sklearn.svm', 'NuSVC', {'nu': 0.8, 'degree': 5, 'C': 1.0, 'coef0' : 0.1}],
['sklearn.svm', 'NuSVC', {'nu': 0.8, 'degree': 5, 'C': 1.0, 'coef0' : 0.2}],
['sklearn.svm', 'NuSVC', {'nu': 0.6, 'degree': 10, 'C': 1.0, 'coef0' : 0.1}],
['sklearn.svm', 'NuSVC', {'nu': 0.6, 'degree': 5, 'C': 1.0, 'coef0' : 0.2}],
['sklearn.svm', 'NuSVC', {'nu': 0.4, 'degree': 5, 'C': 1.0, 'coef0' : 0.1}],
['sklearn.svm', 'NuSVC', {'nu': 0.4, 'degree': 10, 'C': 1.0, 'coef0' : 0.2}],

    # MLP
    ['sklearn.neural_network', 'MLPClassifier', {'learning_rate_init': 0.0001, 'hidden_layer_sizes': (20, 20), 'batch_size': 50, 'alpha': 0}],
    ['sklearn.neural_network', 'MLPClassifier', {'learning_rate_init': 0.0001, 'hidden_layer_sizes': (20, 20), 'batch_size': 50, 'alpha': 0.05}],
    ['sklearn.neural_network', 'MLPClassifier', {'learning_rate_init': 0.0001, 'hidden_layer_sizes': (20, 20), 'batch_size': 50, 'alpha': 0.1}],
    ['sklearn.neural_network', 'MLPClassifier', {'learning_rate_init': 0.01, 'hidden_layer_sizes': (10, 10), 'batch_size': 100, 'alpha': 0.05}],
    ['sklearn.neural_network', 'MLPClassifier', {'learning_rate_init': 0.01, 'hidden_layer_sizes': (10, 10), 'batch_size': 100, 'alpha': 0.1}],
    ['sklearn.neural_network', 'MLPClassifier', {'learning_rate_init': 0.01, 'hidden_layer_sizes': (10, 10), 'batch_size': 100, 'alpha': 0.2}],
    ['sklearn.neural_network', 'MLPClassifier', {'learning_rate_init': 0.01, 'hidden_layer_sizes': (10, 10), 'batch_size': 100, 'alpha': 0.3}],
    ['sklearn.neural_network', 'MLPClassifier', {'learning_rate_init': 0.01, 'hidden_layer_sizes': (10, 10), 'batch_size': 100, 'alpha': 0.35}],

    # Linear Regression
    ['sklearn.linear_model', 'LogisticRegression', {'max_iter' : 100, "solver" : 'newton-cg'}],
    ['sklearn.linear_model', 'LogisticRegression', {'max_iter' : 100, "solver" : 'lbfgs'}],
    ['sklearn.linear_model', 'LogisticRegression', {'max_iter' : 100, "solver" : 'liblinear'}],
    ['sklearn.linear_model', 'LogisticRegression', {'max_iter' : 100, "solver" : 'sag'}],
    ['sklearn.linear_model', 'LogisticRegression', {'max_iter' : 100, "solver" : 'saga'}],

    ['sklearn.linear_model', 'LogisticRegression', {'max_iter' : 200, "solver" : 'newton-cg'}],
    ['sklearn.linear_model', 'LogisticRegression', {'max_iter' : 200, "solver" : 'lbfgs'}],
    ['sklearn.linear_model', 'LogisticRegression', {'max_iter' : 200, "solver" : 'liblinear'}],
    ['sklearn.linear_model', 'LogisticRegression', {'max_iter' : 200, "solver" : 'sag'}],
    ['sklearn.linear_model', 'LogisticRegression', {'max_iter' : 200, "solver" : 'saga'}],

    ['sklearn.svm', 'LinearSVC', {}],
    ['sklearn.svm', 'sklearn.svm', 'LinearSVC', {'tol': 1e-02, 'max_iter': 1000, 'C': 1}],
    ['sklearn.svm', 'LinearSVC', {'tol': 1e-03, 'max_iter': 500, 'C': 1.0}],
    ['sklearn.svm', 'LinearSVC', {'tol': 1e-04, 'max_iter': 500, 'C': 1.0}],
    ['sklearn.svm', 'LinearSVC', {'tol': 1e-05, 'max_iter': 500, 'C': 1.0}],
    ['sklearn.svm', 'LinearSVC', {'tol': 1e-06, 'max_iter': 1000, 'C': 1.0}],
    ['sklearn.svm', 'LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 2}],
    ['sklearn.svm', 'LinearSVC', {'tol': 1e-05, 'max_iter': 500, 'C': 4}],
    ['sklearn.svm', 'LinearSVC', {'tol': 1e-05, 'max_iter': 500, 'C': 5}],


    # RF
    ['sklearn.ensemble', 'RandomForestClassifier', {}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 200, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'sqrt', 'max_depth': 45}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 100, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'sqrt', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'sqrt', 'max_depth': 20}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 20, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'auto', 'max_depth': 2}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 20, 'min_samples_split': 10, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'auto', 'max_depth': 2}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 20, 'min_samples_split': 15, 'min_samples_leaf': 2, 'min_impurity_decrease': 0, 'max_features': 'auto', 'max_depth': 2}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 20, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'auto', 'max_depth': 2}],

    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'auto', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'auto', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'auto', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'auto', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'auto', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'log2', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'log2', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'log2', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'log2', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 15, 'min_samples_leaf': 2, 'min_impurity_decrease': 0, 'max_features': 'log2', 'max_depth': 10}],
    ['sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'min_samples_split': 15, 'min_samples_leaf': 4, 'min_impurity_decrease': 0, 'max_features': 'log2', 'max_depth': 10}],


]

XGB_SEARCH_PARA = {
    'eta': [1, 2, 3, 5, 10, 12, 15, 20],
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [30 ,40, 50, 60, 70, 80, 90, 100],
    "min_child_weight": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
    'n_estimators': [10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
}

################################
# Load the data
################################

from sklearn.preprocessing import OneHotEncoder

class LocalOneHotEncoder(object):

  def __init__(self, target_columns):
    '''
    @param: target_columns --- To perform one-hot encoding column name list.
    '''
    self.enc = OneHotEncoder(handle_unknown='ignore')
    self.col_names = target_columns
    print("Success")

  def fit(self, df):
    '''
    @param: df --- pandas DataFrame
    '''
    self.enc.fit(df[self.col_names].values)
    self.labels = np.array(self.enc.categories_).ravel()
    self.new_col_names = self.gen_col_names(df)

  def gen_col_names(self, df):
    '''
    @param:  df --- pandas DataFrame
    '''
    new_col_names = []
    for col in self.col_names:
      for val in df[col].unique():
        new_col_names.append("{}_{}".format(col, val))
    return new_col_names

  def transform(self, df):
     '''
     @param:  df --- pandas DataFrame
     '''
     return pd.DataFrame(data = self.enc.transform(df[self.col_names]).toarray(),
                         columns = self.new_col_names,
                         dtype=int)




def load_data(path = "./data/a5_q1.pkl"):

    data = pickle.load(open(path, "rb"))

    y_train_original = data['y_train']
    X_train_original = data['X_train']  # Original dataset
    X_train_ohe = data['X_train_ohe']  # One-hot-encoded dataset

    X_test_original = data['X_test']
    X_test_ohe = data['X_test_ohe']



    # ========================================
    ONEHOT_COLUMNS = ["arrival_date_year"]

    ## OneHot Year
    ohe_encoder = LocalOneHotEncoder(ONEHOT_COLUMNS)
    ohe_encoder.fit(X_train_ohe)

    X_train_ohe = pd.concat(
        [X_train_ohe.reset_index(drop=True), ohe_encoder.transform(X_train_ohe).reset_index(drop=True)], axis=1)
    X_test_ohe = pd.concat(
        [X_test_ohe.reset_index(drop=True), ohe_encoder.transform(X_test_ohe).reset_index(drop=True)], axis=1)

    X_train_ohe.pop("arrival_date_year")
    X_test_ohe.pop("arrival_date_year")

    ## Add a new column
    X_train_ohe['is_equal_resve'] = (
                X_train_original['reserved_room_type'] == X_train_original['assigned_room_type']).astype(int)
    X_test_ohe['is_equal_resve'] = (
                X_test_original['reserved_room_type'] == X_test_original['assigned_room_type']).astype(int)

    # since there are 2 missing value, just set them to zero
    X_train_ohe.fillna(0, inplace=True)
    X_test_ohe.fillna(0, inplace=True)

    scaler = Normalizer().fit(X_train_ohe)
    X_train_ohe = scaler.transform(X_train_ohe)
    X_test_ohe = scaler.transform(X_test_ohe)

    return X_train_ohe, y_train_original, X_test_ohe,

################################
# Produce submission
################################

def create_submission(confidence_scores, save_path):
    '''Creates an output file of submissions for Kaggle

    Parameters
    ----------
    confidence_scores : list or numpy array
        Confidence scores (from predict_proba methods from classifiers) or
        binary predictions (only recommended in cases when predict_proba is
        not available)
    save_path : string
        File path for where to save the submission file.

    Example:
    create_submission(my_confidence_scores, './data/submission.csv')

    '''
    import pandas as pd

    submission = pd.DataFrame({"score": confidence_scores})
    submission.to_csv(save_path, index_label="id")

def split_array_into_n_subarray(arr, bin_number):
    '''
    Split An list to some sub list evnely,
        from multiprocessing import Pool

        task_list = split_array_into_n_subarray(need_map_authors, int(pnum))
        p = Pool(int(pnum))
        p.map(threadMethod, [(items, args) for items in task_list])

    :param arr:  Original big array
    :param bin_number:  Split to how many sub arrays
    :return:
    '''
    if len(arr) % bin_number == 0:
        bin_size = max(1, int(len(arr) / bin_number))
    else:
        bin_size = max(1, int(len(arr) / (bin_number - 1)))
    return [arr[i * bin_size:(i + 1) * bin_size] for i in range((len(arr) + bin_size - 1) // bin_size)]



#######################################################



def create_class(module_name, class_name, parameter):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**parameter)

def get_hash_id(text):
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def is_model_exist(hash_id):
    return path.exists("./model/{}.joblib".format(hash_id))

def is_feature_exist(hash_id):
    return   path.exists("./feature/{}.testcsv".format(hash_id)) \
           & path.exists("./feature/{}.traincsv".format(hash_id))


def is_need_gen_feature(modle_file):

    hash_id = modle_file[8:-7]

    return (not is_feature_exist(hash_id))



def save_model(hash_id, clf):
    dump(clf, './model/{}.joblib'.format(hash_id))

def load_model(path):
    return load(path)

def get_exist_models():
    """
    :param target_folder:
    :return: image file name list
    """
    model_file = glob.glob(u"./model/*.joblib")
    return model_file

def get_exist_xgb_train_features():
    """
    :param target_folder:
    :return: image file name list
    """
    feature_file = glob.glob(u"./feature/*.traincsv")
    return feature_file

def get_exist_xgb_predict_features():
    """
    :param target_folder:
    :return: image file name list
    """
    feature_file = glob.glob(u"./feature/*.testcsv")
    return feature_file

def get_exist_Xsubtest_features():
    """
    :param target_folder:
    :return: image file name list
    """
    feature_file = glob.glob(u"./feature/*.xsubcsv")
    return feature_file

def get_exist_X_train_all_features():
    """
    :param target_folder:
    :return: image file name list
    """
    feature_file = glob.glob(u"./feature/*.xallcsv")
    return feature_file

def get_exist_models():
    """
    :param target_folder:
    :return: image file name list
    """
    model_file = glob.glob(u"./model/*.joblib")
    return model_file

###################################################

def __generate_model(arg):

    task_list, X_train, y_train = arg

    for index, task in enumerate(task_list):

        hash_id = get_hash_id(str(task))

        try:
            print("generate model {} of {} [{} : {}]".format(index, len(task_list), hash_id, str(task)))

            model = create_class(*task)
            model.fit(X_train, y_train)
            save_model(hash_id, model)
        except:
            print("Genearate model fail {} {}".format(str(task), hash_id))

def generate_model(X_train, y_train):

    exist_model = [str(m)[8:-7] for m in get_exist_models()]
    new_model = []

    for conf in MODEL_CONF:
        hid = get_hash_id(str(conf))
        print("{} hash_id: {}".format(str(conf), hid))

        if hid in exist_model:
            print("Model {} already exist".format(hid))
            continue
        else:
            new_model.append(conf)


    job_num = JOB
    task_list = split_array_into_n_subarray(new_model, job_num)
    p = Pool(job_num)
    p.map(__generate_model, [(tasks, X_train, y_train) for tasks in task_list])


def __generate_feature(arg):

    files, X_train_all, X_test = arg

    for index, model_file in enumerate(files):

        hash_id = model_file[8:-7]
        print("generate feature {} of {} [{}]".format(index, len(files), hash_id))

        if is_feature_exist(hash_id):
            print("Feature already exist")
            continue

        clf = load_model(model_file)
        if hasattr(clf, "predict_proba"):
            y_pred_train = [x[1] for x in clf.predict_proba(X_train_all)]
            y_pred_test = [x[1] for x in clf.predict_proba(X_test)]
        elif hasattr(clf, "predict"):
            y_pred_train = clf.predict(X_train_all)
            y_pred_test = clf.predict(X_test)
        else:
            y_pred_test = y_pred_train= None

        if y_pred_test is not None:
            pd.Series(y_pred_test).to_csv("./feature/{}.testcsv".format(hash_id), index=False)
            pd.Series(y_pred_train).to_csv("./feature/{}.traincsv".format(hash_id), index=False)




def generate_feature(X_test_all):

    model_files = get_exist_models()
    model_files = list(filter(is_need_gen_feature, model_files))

    if len(model_files) == 0:
        print("All feature done")

    job_num = JOB
    p = Pool(job_num)
    task_list = split_array_into_n_subarray(model_files, job_num)

    p.map(__generate_feature, [(task, X_train_all, X_test_all) for task in task_list])


def geature_xgb_train_featurs():
    feature_files = get_exist_xgb_train_features()
    return pd.concat([pd.read_csv(x)['0'] for x in feature_files], axis=1)

def geature_xgb_predict_featurs():
    feature_files = get_exist_xgb_predict_features()
    return pd.concat([pd.read_csv(x)['0'] for x in feature_files], axis=1)

def geature_xgb_sub_Xtest_featurs():
    feature_files = get_exist_Xsubtest_features()
    return pd.concat([pd.read_csv(x)['0'] for x in feature_files], axis=1)

def geature_X_train_all_featurs():
    feature_files = get_exist_X_train_all_features()
    return pd.concat([pd.read_csv(x)['0'] for x in feature_files], axis=1)

def get_timestamp():
    return strftime("%Y_%m_%d_%H_%M_%S", localtime())

def get_timestamped_file_name(filename, path='./', postfix="csv"):
    '''
    description: get a file name with time stamp
    example:
        filename : wanfang_authors
        prosfix  : csv
    return : wanfang_authors_2016_08_22_05_52_51.csv
    '''
    return "%s/%s_%s.%s" % (path, filename, strftime("%Y_%m_%d_%H_%M_%S", localtime()), postfix)

# ========================================


def search_best_XGboost_parameter(X_train, y_train):

    clf = xgb.XGBClassifier(objective = 'binary:logistic')

    print("===== Searching best XGBoost parameter")

    # run randomized search
    n_iter_search = XGB_SEARCH_ITER
    random_search = RandomizedSearchCV(clf,
                                       n_jobs=-1,
                                       param_distributions=XGB_SEARCH_PARA,
                                       n_iter=n_iter_search,
                                       scoring='roc_auc',
                                       cv=min(2, XGB_SEARCH_CV),
                                       verbose=10,)

    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))

    best_para = random_search.best_params_
    print(best_para)

    print("===== Searching best XGBoost parameter finish")

    return best_para


def do_predict(X_train_all, y_train_all, X_test_all):

    X_new_train = geature_xgb_train_featurs()
    X_new_predict = geature_xgb_predict_featurs()


    X_final_test = pd.concat([X_new_predict.reset_index(drop=True), pd.DataFrame(X_test_all).reset_index(drop=True)], axis=1)
    X_final_train = pd.concat([X_new_train.reset_index(drop=True), pd.DataFrame(X_train_all).reset_index(drop=True)], axis=1)

    best_para = search_best_XGboost_parameter(X_final_train.values, y_train_all.values)

    # Traning on all data
    clf = xgb.XGBClassifier(**best_para)

    clf.fit(X_final_train.values, y_train_all.values)

    y_pred = clf.predict_proba(X_final_test.values)

    predictions = [x[1] for x in y_pred]
    submit_file = get_timestamped_file_name('submission', './predict', 'csv')
    create_submission(np.array(predictions), submit_file)

###

if __name__ == "__main__":

    print("Job: {}".format(JOB))

    X_train_all, y_train_all, X_test_all = load_data()

    #X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(X_train_all, y_train_all, test_size=0.1, random_state=42)

    #================================

    generate_model(X_train_all, y_train_all)

    generate_feature(X_train_all)

    do_predict(X_train_all, y_train_all, X_test_all)

























