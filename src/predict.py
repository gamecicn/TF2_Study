from __future__ import absolute_import, division, print_function, unicode_literals


import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler


#=================================================

TRAIN_RATIO = 0.9

def get_dataset(data_file):
    '''
    :param data_file:
    :return:  narray
    '''

    df = pd.read_csv(data_file)

    features_considered = ['open', 'high', 'low', 'volumn', 'close']
    features = df[features_considered]

    features.index = df['date']

    dataset = features.values

    return dataset

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      future_target, stride, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - future_target

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, stride)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+future_target])
        else:
            labels.append(target[i:i+future_target])

    return np.array(data), np.array(labels)


if __name__ == '__main__':

    dataset = get_dataset('../data/510050Full.txt')

    # Standlization
    train_size = round(dataset.shape[0]* TRAIN_RATIO)
    future_target = 0
    stride = 1

    scaler = StandardScaler()
    scaler.fit(dataset[: train_size])
    dataset = scaler.transform(dataset)

    # Prepare Data
    multivariate_param = { "dataset"       : dataset,
                            "target"        : dataset[:, 1],
                            "start_index"   : 0,
                            "end_index"     : train_size,
                            "history_size"  : 3,
                            "future_target" : future_target,
                            "stride"        : stride,
                            "single_step"   : True }

    x_train_single, y_train_single = multivariate_data(**multivariate_param)
    x_val_single, y_val_single = multivariate_data(**multivariate_param)

    # Build Training Data
    EPOCHS = 10
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    # https://www.tensorflow.org/guide/data#consuming_numpy_arrays
    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().batch(BATCH_SIZE)

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE)

    # Build Model
    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))
    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    single_step_model.summary()

    # Train
    EVALUATION_INTERVAL = 200

    single_step_history = single_step_model.fit(train_data_single,
                                                epochs=30,
                                                validation_data=val_data_single)


    print("Done")






















