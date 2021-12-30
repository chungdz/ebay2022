import numpy as np
import json
import pandas as pd
from tqdm import trange
import argparse
from keras import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import CSVLogger

def custom_asymmetric_eval(y_train, preds):
    nd_preds = tf.math.round(preds)
    residual = y_train - nd_preds
    loss = tf.where(residual < 0, residual * -0.6, residual * 0.4)
    return tf.math.reduce_mean(loss)

folds = 10
esr = 3

parser = argparse.ArgumentParser()
parser.add_argument("--starti", default=1, type=int)
parser.add_argument("--depth", default=12, type=int)
parser.add_argument("--num_rounds", default=1000, type=int)
parser.add_argument("--border_count", default=254, type=int)
parser.add_argument("--random_strength", default=1, type=float)
parser.add_argument("--esr", default=3, type=int)
parser.add_argument("--l2_leaf", default=3, type=float)
args = parser.parse_args()

loss_and_output = []
all_log = []
for i in trange(args.starti, folds + 1):
    print('model:', i)
    train_set = pd.read_csv('data/subtrain_cat/train_{}.tsv'.format(i), sep='\t')
    valid_set = pd.read_csv('data/subtrain_cat/valid_{}.tsv'.format(i), sep='\t')

    x_train = train_set.drop(['record_number', 'target'], axis=1)
    y_train = train_set.target
    x_valid = valid_set.drop(['record_number', 'target'], axis=1)
    y_valid = valid_set.target

    ## nn model
    inputs = keras.Input(shape=(82,))
    x = layers.Dense(512, activation="sigmoid")(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="sigmoid")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="tanh")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="tanh")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="relu")(x)
    model = keras.Model(inputs, outputs)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='custom_asymmetric_eval', patience=3),
                 CSVLogger('result/log.csv', append=False, separator=';')]
    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=[custom_asymmetric_eval], callbacks=[callbacks])

    batch_size = 32
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=3, validation_data=(x_valid, y_valid),
                        callbacks=[callbacks])
    model.save('para/nn_{}.h5'.format(i))
    logstr = pd.read_csv("result/log.csv", sep=';')
    # only final loss is saved in Keras log
    loss_and_output.append(float(logstr.val_custom_asymmetric_eval))

lao = np.array([1 / x for x in loss_and_output])
lao = lao / lao.sum()

json.dump(list(lao), open('para/nn_weight.json', 'w'))
json.dump(loss_and_output, open('para/all_nn_log.json', 'w'))
print('mean loss:', np.mean(loss_and_output))
