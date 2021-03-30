import pandas as pd
import numpy as np
import glob
import pickle
from sklearn.preprocessing import MinMaxScaler
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
        bin_size = int(len(arr) / (bin_number - 1))
    return [arr[i * bin_size:(i + 1) * bin_size] for i in range((len(arr) + bin_size - 1) // bin_size)]



#######################################################
from multiprocessing import Pool, cpu_count
from sklearn.svm import LinearSVC
import importlib
import os.path
from os import path
from joblib import dump, load

MODEL_CONF = [

['sklearn.svm', 'LinearSVC', {}],
#['LinearSVC', {'tol': 1e-02, 'max_iter': 1000, 'C': 100.0}],
#['LinearSVC', {'tol': 1e-03, 'max_iter': 1000, 'C': 100.0}],
#['LinearSVC', {'tol': 1e-04, 'max_iter': 1000, 'C': 100.0}],
#['LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 100.0}],
#['LinearSVC', {'tol': 1e-06, 'max_iter': 1000, 'C': 100.0}],
#['LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 1e+2}],
#['LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 1e+3}],
#['LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 1e-1}],
#['LinearSVC', {'tol': 1e-05, 'max_iter': 1000, 'C': 1e-2}],

['sklearn.ensemble', 'RandomForestClassifier', {}],

]


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
    return path.exists("./feature/{}.csv".format(hash_id))

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

def get_exist_features():
    """
    :param target_folder:
    :return: image file name list
    """
    feature_file = glob.glob(u"./feature/*.csv")
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

    job_num = cpu_count()
    task_list = split_array_into_n_subarray(MODEL_CONF, job_num)
    p = Pool(job_num)
    p.map(__generate_model, [(tasks, X_train, y_train) for tasks in task_list])


def __generate_feature(arg):

    files, X_test = arg

    for index, model_file in enumerate(files):

        hash_id = model_file[8:-7]
        print("generate model {} of {} [{}]".format(index, len(files), hash_id))

        if is_feature_exist(hash_id):
            print("Feature already exist")
            continue

        clf = load_model(model_file)
        if hasattr(clf, "predict_proba"):
            y_pred = [x[1] for x in clf.predict_proba(X_test)]
        elif hasattr(clf, "predict"):
            y_pred = clf.predict(X_test)
        else:
            y_pred = None

        if y_pred is not None:
            pd.Series(y_pred).to_csv("./feature/{}.csv".format(hash_id), index=False)



def generate_feature(X_test):

    model_files = get_exist_models()

    job_num = 1 # cpu_count()
    p = Pool(job_num)
    task_list = split_array_into_n_subarray(model_files, job_num)

    p.map(__generate_feature, [(task, X_test) for task in task_list])


def geature_featurs():
    feature_files = get_exist_features()
    return pd.concat([pd.read_csv(x)['0'] for x in feature_files], axis=1)



def do_predict(y_train, X_test):

    X_new = geature_featurs()

    

###

if __name__ == "__main__":

    X_train, y_train, X_test = load_data()

    #generate_model(X_train, y_train)

    #generate_feature(X_test)

    do_predict(y_train, X_test)

























