#!/usr/bin/env python3
# -*- coding: utf-8 -*-
seed_value= 0
import os, random, pickle
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Model
from keras.layers import LSTM, TimeDistributed, Dense, Dropout, RepeatVector, Input
# from hyperopt import tpe, STATUS_OK, Trials
# from hyperas import optim
from sklearn.metrics.pairwise import cosine_similarity


'''Code for the seq2seq_rf model. '''


def data():
    TEST_ON=4
    data_folder = '../data_proj/'

    ts = pickle.load(open(data_folder+'vectors.pkl', 'rb'))
    train_idx = pickle.load(open(data_folder+'train_idx.pkl', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.pkl', 'rb'))

    trainX = ts[train_idx, 0:TEST_ON, :]
    trainY = ts[train_idx, :, :] 
    testX = ts[test_idx, 0:TEST_ON, :]
    testY = ts[test_idx, :, :]

    return trainX, trainY, testX, testY


def create_lstm_model(trainX, trainY, testX, testY):
    TEST_ON=4

    trainY_past = trainY[:, 0:TEST_ON, :]
    testY_past = testY[:, 0:TEST_ON, :]    
    trainY_future = trainY[:, TEST_ON:, :]
    testY_future = testY[:, TEST_ON:, :]
    
    inputs = Input(shape=(trainX.shape[1],100))
    x = LSTM(128, return_sequences=True)(inputs)
    x = TimeDistributed(Dropout(0.1))(x)
    x = LSTM(64, go_backwards=True)(x)
    x = Dropout(0.1)(x)
    
    y_past = RepeatVector(testY_past.shape[1])(x)
    y_future = RepeatVector(testY_future.shape[1])(x)
    
    y_past = LSTM(64, return_sequences=True)(y_past)
    y_past = TimeDistributed(Dropout(0.1))(y_past)
    y_past = LSTM(128, go_backwards=False, return_sequences=True)(y_past)
    y_past = TimeDistributed(Dropout(0.1))(y_past)
    y_past = TimeDistributed(Dense(100))(y_past)
    
    y_future = LSTM(64, return_sequences=True)(y_future)
    y_future = TimeDistributed(Dropout(0.1))(y_future)
    y_future = LSTM(128, go_backwards=False, return_sequences=True)(y_future)
    y_future = TimeDistributed(Dropout(0.1))(y_future)
    y_future = TimeDistributed(Dense(100))(y_future)
    
    model = Model(inputs=inputs, outputs=[y_past, y_future])
    
    val_start = int(0.75*len(trainX))
    model.compile(loss="mean_squared_error",  optimizer="adam")

    result = model.fit(trainX[0:val_start], [trainY_past[0:val_start], trainY_future[0:val_start]],
              batch_size=32,
              epochs=30,
              verbose=1)

    # Hiyam: added this here
    folder = '../results/results_seq2seq_rf'
    if not os.path.exists(folder):
        os.mkdir(folder)
    pickle.dump(model, open(folder + '/' + str(TEST_ON) + 'model2.p', "wb"))

    valX = trainX[val_start:len(trainX)]
    val_true_past  = trainY_past[val_start:len(trainX)]
    val_true_future  = trainY_future[val_start:len(trainX)]
    
    preds = model.predict(valX)
    preds_past, preds_future = preds[0], preds[1]
    
    cosines_past = np.array([np.average(np.diag(cosine_similarity(preds_past[:, i, :], val_true_past[:, i, :]))) for i in range(trainY_past.shape[1])])
    cosines_future = np.array([np.average(np.diag(cosine_similarity(preds_future[:, i, :], val_true_future[:, i, :]))) for i in range(trainY_future.shape[1])])
    
    val_loss_past = np.average(cosines_past)
    val_loss_future = np.average(cosines_future)
    micro_loss_past = [np.average(cosines_past) for i in range(val_true_past.shape[1])]
    micro_loss_future = [np.average(cosines_future) for i in range(val_true_future.shape[1])]
    
    macro_avg_loss = np.average([val_loss_past, val_loss_future])
    micro_loss_past.extend(micro_loss_future)
    micro_avg_loss = np.average([micro_loss_past])
    print('\n', model.summary(), '\t\tBest val cosine:', val_loss_past, val_loss_future, macro_avg_loss, micro_avg_loss, '\n')
    # return {'loss': -micro_avg_loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    TEST_ON = 4

    trainX, trainY, testX, testY = data()
    create_lstm_model(trainX, trainY, testX, testY)

    # trials = Trials()
    # best_run, best_model = optim.minimize(model=create_lstm_model,
    #                                           data=data,
    #                                           algo=tpe.suggest,
    #                                           max_evals=25,
    #                                           eval_space=True,
    #                                           trials=trials)
    # folder = '../results/results_seq2seq_rf'
    # if not os.path.exists(folder):
    #     os.mkdir(folder)
    # pickle.dump(trials, open(folder+'/'+str(TEST_ON)+'trials2.p', "wb"))
    # pickle.dump(best_run, open(folder+'/'+str(TEST_ON)+'params2.p', "wb"))
    # pickle.dump(best_model, open(folder+'/'+str(TEST_ON)+'model2.p', "wb"))