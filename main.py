import pandas as pd
import numpy as np
import glob
import pickle
from sklearn.preprocessing import MinMaxScaler
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

XGB_SEARCH_ITER = 1
XGB_SEARCH_CV = 2

JOB = cpu_count()

MODEL_CONF = [

    ['sklearn.svm', 'LinearSVC', {}],
    ['LinearSVC', {'tol': 1e-02, 'max_iter': 1000, 'C': 100.0}],
    ['LinearSVC', {'tol': 1e-03, 'max_iter': 1000, 'C': 100.0}],
    ['LinearSVC', {'tol': 1e-04, 'max_iter': 1000, 'C': 100.0}],
    ['LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 100.0}],
    ['LinearSVC', {'tol': 1e-06, 'max_iter': 1000, 'C': 100.0}],
    ['LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 1e+2}],
    ['LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 1e+3}],
    ['LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 1e-1}],
    ['LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 1e-2}],
    ['LinearSVC', {'tol': 1e-05, 'max_iter': 100, 'C': 1e+2}],
    ['LinearSVC', {'tol': 1e-05, 'max_iter': 200, 'C': 1e+3}],
    ['LinearSVC', {'tol': 1e-05, 'max_iter': 500, 'C': 1e-1}],
    ['LinearSVC', {'tol': 1e-04, 'max_iter': 300, 'C': 1e-1}],

    ['sklearn.ensemble', 'RandomForestClassifier', {}],

]

XGB_SEARCH_PARA = {
    'eta': [1, 2, 3, 5, 10, 12, 15, 20],
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [40, 50, 60, 70, 80, 90, 100],
    "min_child_weight": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
    'n_estimators': [2, 5, 10, 15, 20, 25, 30, 40, 50, 60]

}




################################
# Load the data
################################


def load_data(path = "./data/a5_q1.pkl"):

    data = pickle.load(open(path, "rb"))

    y_train_original = data['y_train']
    X_train_original = data['X_train']  # Original dataset
    X_train_ohe = data['X_train_ohe']  # One-hot-encoded dataset

    X_test_original = data['X_test']
    X_test_ohe = data['X_test_ohe']

    # since there are 2 missing value, just set them to zero
    X_train_ohe.fillna(0, inplace=True)
    X_test_ohe.fillna(0, inplace=True)

    scaler = MinMaxScaler().fit(X_train_ohe)
    X_train_ohe_scale = scaler.transform(X_train_ohe)
    X_test_ohe_scale = scaler.transform(X_test_ohe)

    return X_train_ohe_scale, y_train_original, X_test_ohe_scale,

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
        bin_size = int(len(arr) / bin_number)
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
    return   path.exists("./feature/{}.traincsv".format(hash_id)) \
           & path.exists("./feature/{}.testcsv".format(hash_id)) \
           & path.exists("./feature/{}.xsubcsv".format(hash_id)) \
           & path.exists("./feature/{}.xallcsv".format(hash_id))


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

###################################################

def __generate_model(arg):

    task_list, X_train, y_train = arg

    for index, task in enumerate(task_list):

        hash_id = get_hash_id(str(task))
        print("generate model {} of {} [{} : {}]".format(index, len(task_list), hash_id, str(task)))

        try:
            if is_model_exist(hash_id):
                print("{} already exist".format(hash_id))
                continue

            model = create_class(*task)
            model.fit(X_train, y_train)
            save_model(hash_id, model)

        except:
            print("Exception arise {}".format(hash_id))

def generate_model(X_train, y_train):

    job_num = JOB
    task_list = split_array_into_n_subarray(MODEL_CONF, job_num)
    p = Pool(job_num)
    p.map(__generate_model, [(tasks, X_train, y_train) for tasks in task_list])


def __generate_feature(arg):

    files, X_train, X_test, X_subtest, y_sub_test, X_train_all = arg

    for index, model_file in enumerate(files):

        hash_id = model_file[8:-7]
        print("generate feature {} of {} [{}]".format(index, len(files), hash_id))

        if is_feature_exist(hash_id):
            print("Feature already exist")
            continue

        clf = load_model(model_file)
        if hasattr(clf, "predict_proba"):
            y_pred_train = [x[1] for x in clf.predict_proba(X_train)]
            y_pred_test = [x[1] for x in clf.predict_proba(X_test)]
            y_pred_xsub = [x[1] for x in clf.predict_proba(X_subtest)]
            y_pred_all = [x[1] for x in clf.predict_proba(X_train_all)]
        elif hasattr(clf, "predict"):
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            y_pred_xsub = clf.predict(X_subtest)
            y_pred_all = clf.predict(X_train_all)
        else:
            y_pred_train = y_pred_test = y_pred_xsub = y_pred_all = None

        if y_pred_train is not None:

            y_sub_pred = clf.predict(X_subtest)
            print("!!! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  Accuracy: {}  [hash_id]".format(accuracy_score(y_sub_test, y_sub_pred)))

            pd.Series(y_pred_train).to_csv("./feature/{}.traincsv".format(hash_id), index=False)
            pd.Series(y_pred_test).to_csv("./feature/{}.testcsv".format(hash_id), index=False)
            pd.Series(y_pred_xsub).to_csv("./feature/{}.xsubcsv".format(hash_id), index=False)
            pd.Series(y_pred_all).to_csv("./feature/{}.xallcsv".format(hash_id), index=False)




def generate_feature(X_train, X_test, X_subtest, y_sub_test, X_train_all):

    model_files = get_exist_models()

    job_num = JOB
    p = Pool(job_num)
    task_list = split_array_into_n_subarray(model_files, job_num)

    p.map(__generate_feature, [(task, X_train, X_test, X_subtest, y_sub_test, X_train_all) for task in task_list])


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


def search_best_XGboost_parameter(X_train, X_test, y_train, y_test):

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
    random_search.fit(X_train.values, y_train.values)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))

    best_para = random_search.best_params_
    print(best_para)

    print("===== Searching best XGBoost parameter finish")

    #==============================
    #X_test
    #===============================
    clf = xgb.XGBClassifier(**best_para)
    clf.fit(X_train.values, y_train.values)
    y_pred = clf.predict(X_test.values)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> XGB Final Accuracy: {}".format(accuracy_score(y_test, y_pred)))

    return best_para

def do_predict(y_sub_train, y_sub_test, y_train_all):

    X_new_sub_train = geature_xgb_train_featurs()
    X_new_predict = geature_xgb_predict_featurs()
    X_new_sub_test = geature_xgb_sub_Xtest_featurs()
    X_nwe_train_all = geature_X_train_all_featurs()


    best_para = search_best_XGboost_parameter(X_new_sub_train, X_new_sub_test, y_sub_train, y_sub_test)

    # Traning on all data
    clf = xgb.XGBClassifier(**best_para)
    clf.fit(X_nwe_train_all.values, y_train_all.values)
    y_pred = clf.predict_proba(X_new_predict.values)

    predictions = [x[1] for x in y_pred]
    submit_file = get_timestamped_file_name('submission', './predict', 'csv')
    create_submission(np.array(predictions), submit_file)

###

if __name__ == "__main__":

    X_train_all, y_train_all, X_test_all = load_data()

    X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(X_train_all, y_train_all, test_size=0.1, random_state=42)

    #generate_model(X_sub_train, y_sub_train)

    generate_feature(X_sub_train, X_test_all, X_sub_test, y_sub_test, X_train_all)

    do_predict(y_sub_train, y_sub_test, y_train_all)

























