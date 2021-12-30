from keras.models import load_model
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

parser = argparse.ArgumentParser()
parser.add_argument("--starti", default=1, type=int)
args = parser.parse_args()

# load quiz set
print("loading quiz set")
quiz_set = pd.read_csv('data/parsed_quiz_cat.tsv', sep='\t')
quiz_set.drop(['record_number'], axis=1, inplace=True)

def custom_asymmetric_eval(y_train, preds):
    nd_preds = tf.math.round(preds)
    residual = y_train - nd_preds
    loss = tf.where(residual < 0, residual * -0.6, residual * 0.4)
    return tf.math.reduce_mean(loss)

w = json.load(open('para/nn_weight.json', 'r'))

print('predicting of model {}...'.format(args.starti))
model = load_model('para/nn_{}.h5'.format(args.starti), custom_objects={'custom_asymmetric_eval': custom_asymmetric_eval})
ypred = model.predict(quiz_set)
print('finish prediction of model {}'.format(args.starti))
np.save('result/nn_result_{}'.format(args.starti), ypred)
print('save prediction of model {}'.format(args.starti))