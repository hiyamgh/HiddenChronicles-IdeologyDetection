#!/usr/bin/env python3
# -*- coding: utf-8 -*-
seed_value = 0
import os, random, pickle
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
from keras.layers import LSTM, TimeDistributed, Dense, Dropout, RepeatVector, Input, Lambda
from sklearn.metrics.pairwise import cosine_similarity

'''Code for the seq2seq_rf model. '''

latent_dim = 1
embedding_dim = 300

def data():
    TEST_ON = 4
    data_folder = '../data_final_proj/'

    ts = pickle.load(open(data_folder + 'vectors.pkl', 'rb'))
    train_idx = pickle.load(open(data_folder + 'train_idx.pkl', 'rb'))
    test_idx = pickle.load(open(data_folder + 'test_idx.pkl', 'rb'))

    trainX = ts[train_idx, 0:TEST_ON, :]
    trainY = ts[train_idx, :, :]
    testX = ts[test_idx, 0:TEST_ON, :]
    testY = ts[test_idx, :, :]

    return trainX, trainY, testX, testY

# # encoder
# inter_dim = 32
# timesteps, features = 100, 1


def sampling(args):
    z_mean, z_log_sigma = args
    batch_size = tf.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return z_mean + z_log_sigma * epsilon


def create_lstm_model(trainX, trainY, testX, testY):

    def vae_loss2(input_x, decoder1, decoder2, z_log_sigma, z_mean):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        # recon1 = K.sum(K.binary_crossentropy(input_x, decoder1), axis=1)
        # recon2 = K.sum(K.binary_crossentropy(input_x, decoder2), axis=1)

        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma)
        return kl
        # return recon1 + recon2 + kl

    # var_x = K.square(sigma_x)
    # reconst_loss = -0.5 * K.sum(K.log(var_x), axis=2) + K.sum(K.square(x - mu_x) / var_x, axis=2)
    # reconst_loss = K.reshape(reconst_loss, shape=(x.shape[0], 1))
    # return K.mean(reconst_loss, axis=0)

    TEST_ON = 4

    trainY_past = trainY[:, 0:TEST_ON, :]
    testY_past = testY[:, 0:TEST_ON, :]
    trainY_future = trainY[:, TEST_ON:, :]
    testY_future = testY[:, TEST_ON:, :]

    # timesteps, features
    inputs = Input(shape=(trainX.shape[1], embedding_dim))

    # intermediate dimension
    x = LSTM(128, return_sequences=True)(inputs)
    x = TimeDistributed(Dropout(0.1))(x)
    x = LSTM(64, go_backwards=True)(x)
    # x = Dropout(0.1)(x)

    # z_layer
    z_mean = Dense(latent_dim)(x)
    z_log_sigma = Dense(latent_dim)(x)
    z = Lambda(sampling)([z_mean, z_log_sigma])

    # decoders
    y_past = RepeatVector(testY_past.shape[1])(z)
    y_future = RepeatVector(testY_future.shape[1])(z)

    y_past = LSTM(64, return_sequences=True)(y_past)
    y_past = TimeDistributed(Dropout(0.1))(y_past)
    y_past = LSTM(128, go_backwards=False, return_sequences=True)(y_past)
    y_past = TimeDistributed(Dropout(0.1))(y_past)
    y_past = TimeDistributed(Dense(embedding_dim))(y_past)

    y_future = LSTM(64, return_sequences=True)(y_future)
    y_future = TimeDistributed(Dropout(0.1))(y_future)
    y_future = LSTM(128, go_backwards=False, return_sequences=True)(y_future)
    y_future = TimeDistributed(Dropout(0.1))(y_future)
    y_future = TimeDistributed(Dense(embedding_dim))(y_future)

    model = Model(inputs=inputs, outputs=[y_past, y_future])

    val_start = int(0.75 * len(trainX))

    # recon1 = K.sum(K.binary_crossentropy(inputs, y_past))
    # recon2 = K.sum(K.binary_crossentropy(inputs, y_future))

    # recon1 = K.sum(K.binary_crossentropy(inputs, y_past))
    # # recon2 = K.sum(K.binary_crossentropy(inputs, y_future))
    #
    # # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    # kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma)
    # model.add_loss(kl)
    # loss = recon1 + recon2 + kl

    # loss = recon1 + kl

    # model.add_loss(vae_loss2(inputs, y_past, y_future, z_log_sigma, z_mean))

    model.compile(loss='mean_squared_error', optimizer="adam")

    # model.metrics_tensors.append(kl)
    # model.metrics_names.append("kl_loss")

    # result = model.fit(trainX[0:val_start], [trainY_past[0:val_start], trainY_future[0:val_start]],
    #                    batch_size=32,
    #                    epochs=30,
    #                    verbose=1)

    model.fit(trainX[0:val_start], [trainY_past[0:val_start], trainY_future[0:val_start]],
                       batch_size=64,
                       epochs=5,
                       verbose=1)

    # Hiyam: added this here
    folder = '../results_final/results_seq2seq_rf_vae/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    pickle.dump(model, open(folder + str(TEST_ON) + 'model2.p', "wb"))

    valX = trainX[val_start:len(trainX)]
    val_true_past = trainY_past[val_start:len(trainX)]
    val_true_future = trainY_future[val_start:len(trainX)]

    preds = model.predict(valX)
    preds_past, preds_future = preds[0], preds[1]

    cosines_past = np.array(
        [np.average(np.diag(cosine_similarity(preds_past[:, i, :], val_true_past[:, i, :]))) for i in
         range(trainY_past.shape[1])])
    cosines_future = np.array(
        [np.average(np.diag(cosine_similarity(preds_future[:, i, :], val_true_future[:, i, :]))) for i in
         range(trainY_future.shape[1])])

    val_loss_past = np.average(cosines_past)
    val_loss_future = np.average(cosines_future)
    micro_loss_past = [np.average(cosines_past) for i in range(val_true_past.shape[1])]
    micro_loss_future = [np.average(cosines_future) for i in range(val_true_future.shape[1])]

    macro_avg_loss = np.average([val_loss_past, val_loss_future])
    micro_loss_past.extend(micro_loss_future)
    micro_avg_loss = np.average([micro_loss_past])
    print('\n', model.summary(), '\t\tBest val cosine:', val_loss_past, val_loss_future, macro_avg_loss, micro_avg_loss,
          '\n')
    # return {'loss': -micro_avg_loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    TEST_ON = 4

    trainX, trainY, testX, testY = data()
    create_lstm_model(trainX, trainY, testX, testY)
