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
args = parser.parse_args()

print('model:', args.starti)
x_train = pd.read_csv('data/subtrain_cat/train_{}.tsv'.format(args.starti), sep='\t')
x_valid = pd.read_csv('data/subtrain_cat/valid_{}.tsv'.format(args.starti), sep='\t')

y_train = x_train.target
x_train = x_train.drop(['record_number', 'target'], axis=1)
y_valid = x_valid.target
x_valid = x_valid.drop(['record_number', 'target'], axis=1)

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
             CSVLogger('result/log_{}.csv'.format(args.starti), append=False, separator=';')]
# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=[custom_asymmetric_eval])

batch_size = 32
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_valid, y_valid),
                    callbacks=[callbacks])
model.save('para/nn_{}.h5'.format(args.starti))