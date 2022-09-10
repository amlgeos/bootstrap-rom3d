"""
(4)
This is the main program that enables AI training and saving of the results.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import os
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from termcolor import colored
import pickle

tf.random.set_seed(1234)

font = {'size': 18, 'family': 'DeJavu Serif', 'serif': ['Palatino']}
plt.rc('font', **font)
params = {'legend.fontsize': 14, 'lines.markersize': 9, 'backend': 'Qt5Agg'}
plt.rcParams.update(params)
plt.rc('text', usetex=True)


def useCPUforTraining(switch):
    # Use CPU because size data is low
    if switch:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif not switch:
        print("Using GPU : Nvidia M1200\n")
    else:
        print("Invalid option!!")


useCPUforTraining(True)


def weighted_mse(yTrue, yPred):
    return K.mean(b * K.square(yTrue - yPred))


def weighted_mser(yTrue, yPred):
    err = (1 * np.square(yTrue - yPred))
    return err


def pureRNN(input_shape, output_shape):
    dropout = 0.0

    # Model
    model = Sequential()
    model.add(LSTM(units=int(input_shape[2] * 2.5),
                   input_shape=(input_shape[1], input_shape[2]),
                   return_sequences=True,
                   go_backwards=True,
                   dropout=dropout, ))
    model.add(LSTM(units=int(input_shape[2] * 2.0),
                   return_sequences=True,
                   go_backwards=True,
                   dropout=dropout))
    model.add(LSTM(units=int(input_shape[2] * 1.5),
                   go_backwards=True,
                   dropout=dropout))
    model.add(Dense(units=output_shape[1]))  # because reshaped and new axis is added
    model.compile(loss=weighted_mse, optimizer='adam')
    return model


def getFromPool2(initialPool_, idx):
    trainset_ = initialPool_[0:idx]
    return initialPool_, trainset_


def getPredResult(trset, trset_Names, set_nub, epochs=20, vsplit=0.01, ncomps=15):
    xtr20, ytr20 = list(), list()
    for i in range(len(trset)):
        trsi = trset[i]
        xtr20.append(trsi[:, :-ncomps])
        ytr20.append(trsi[:, -ncomps:])
    print()

    # add noise
    noisePercent = 30 / 100  # train on noisy data (only prepare noisy training set)
    nversion = 20  #

    for bb, (xtr_, ytr_) in enumerate(zip(xtr20, ytr20)):
        xtr_v = list()
        xtr_v_bcs = list()
        ytr_v = list()
        for bbb in range(nversion):
            if bbb > 0:
                noiseSigma = noisePercent * xtr_[:, :ncomps]
                noise = noiseSigma * np.random.normal(0, 0.5, np.shape(xtr_[:, :ncomps]))
                xtr_v.append(xtr_[:, :ncomps] + noise)
                xtr_v_bcs.append(xtr_[:, ncomps:])
                ytr_v.append(ytr_)
            else:
                # noiseSigma = noisePercent * xtr_[:, :ncomps]
                # noise = noiseSigma * np.random.normal(0, 1, np.shape(xtr_[:, :ncomps]))
                xtr_v.append(xtr_[:, :ncomps] + 0)
                xtr_v_bcs.append(xtr_[:, ncomps:])
                ytr_v.append(ytr_)
        # print()
        xtr20[bb] = np.hstack([np.concatenate(xtr_v, axis=0), np.concatenate(xtr_v_bcs, axis=0)])
        ytr20[bb] = np.concatenate(ytr_v, axis=0)
    print()

    # # print()
    print(colored('\nTRAIN:', 'yellow'), trset_Names)

    # extracting the X and y for training
    x_train_permute = np.expand_dims(np.concatenate(xtr20, axis=0), axis=1)
    y_train_permute = np.concatenate(ytr20, axis=0)

    # splitting all training and testing
    x_train_permute, x_test, y_train_permute, y_test = train_test_split(x_train_permute, y_train_permute, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train_permute, y_train_permute, test_size=0.25)

    history = 0
    saved = 1
    if saved:
        saved_model_name = analysis_folder + "analysis/rnn_permute_" + str(set_nub) + "_.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(saved_model_name, verbose=1, save_best_only=False)
        callbacks = [checkpoint]
        history = model.fit(x=x_train, y=y_train,
                            validation_split=vsplit,
                            shuffle=True,
                            epochs=epochs,
                            callbacks=callbacks,
                            verbose=1)

        # compute on validation error
        y_val_pred = model.predict(x_val)
        err_val = weighted_mser(y_val, y_val_pred)

        if np.mean(err_val) < 1e-2:  #
            # compute on test error
            y_test_pred = model.predict(x_test)
            err_test = weighted_mser(y_test, y_test_pred)
        else:
            err_test = -99
            y_test = -99
            y_test_pred = -99

    else:
        print("Models are not being saved!")

    return history, err_val, err_test, \
           y_val, y_val_pred, \
           y_test, y_test_pred


def makeSupervisedSet(trainset_):
    for i in range(len(trainset_)):
        X = trainset_[i]
        present = X[:-1, :]
        future = X[1:, :]
        Xy = np.hstack([present, future[:, 15:], future[:, :15]])
        trainset_[i] = Xy
    return trainset_


# ############################################ Start program ############################################## #
analysis_folder = os.getcwd() + "/"
state_vars = np.load(analysis_folder + 'normalized/state_.npy', allow_pickle=True)
wts = state_vars[0]
bcs = state_vars[1]
rbs_wts = state_vars[2]
mms_wts = state_vars[3]
rbs_bcs = state_vars[4]
mms_bcs = state_vars[5]
simnames = state_vars[6]

combined = list(np.concatenate([wts, bcs], axis=2))
pca_ = np.load(analysis_folder + "analysis/u/pca_.npy", allow_pickle=True)[0]
b = K.constant(pca_.explained_variance_ratio_)
bnpy = np.array(b)
normalizer = [rbs_wts, mms_wts]

# initialise AI
inp_shape = [1, 1, 23]
out_shape = [1, 15]
model = pureRNN(inp_shape, out_shape)


# Containers and intialisations
errTestInVar = list()
errorInVar = list()
trainedModels = list()
doneidx = 0
init = 0
pool = combined.copy()
snames = simnames.copy()
trainset = 0
trainsetNames = 0
validation_ = 0

lenTrains = list()
y_val_store = list()
y_val_pred_store = list()
y_test_store = list()
y_test_pred_store = list()


# train loop beginning
while init <= len(pool):
    if doneidx == 0:
        init = 19
        initidx = np.arange(init)
        pool, trainset = getFromPool2(pool, init)  # train extract
        poolNames, trainsetNames = getFromPool2(snames, init)  # names extract
        supervisedSet = makeSupervisedSet(trainset.copy())  # make supervised set of the train set

        # Prediction
        history_, val_error_, test_error_, \
        y_val, y_val_pred, \
        y_test, y_test_pred = getPredResult(supervisedSet, trainsetNames.copy(), set_nub=doneidx)
        init = init + 2
        doneidx = doneidx + 1
    else:
        try:
            print("init length = {} == {}".format(init, len(pool)))
            # appending new data
            initidx = np.arange(init)
            pool, trainset = getFromPool2(pool, init)  # train extract
            poolNames, trainsetNames = getFromPool2(snames, init)
            supervisedSet = makeSupervisedSet(trainset.copy())

            # Prediction
            history_, val_error_, test_error_, \
            y_val, y_val_pred, \
            y_test, y_test_pred = getPredResult(supervisedSet, trainsetNames.copy(), set_nub=doneidx)
            init = init + 2
        except:
            print("\nExiting!")
            break

        doneidx += 1

    trainedModels.append(history_.history)
    errorInVar.append(val_error_)
    errTestInVar.append(test_error_)
    lenTrains.append(len(trainsetNames))

    y_val_store.append(y_val)
    y_val_pred_store.append(y_val_pred)
    y_test_store.append(y_test)
    y_test_pred_store.append(y_test_pred)

with open(analysis_folder + 'results/trained_Models_seq.pkl', 'wb') as file_pi:
    pickle.dump(trainedModels, file_pi)
with open(analysis_folder + 'results/errors_Models_seq.pkl', 'wb') as file_pi:
    pickle.dump(errorInVar, file_pi)
with open(analysis_folder + 'results/errors_Test_Models_seq.pkl', 'wb') as file_pi:
    pickle.dump(errTestInVar, file_pi)

with open(analysis_folder + 'results/lenTrains.pkl', 'wb') as file_pi:
    pickle.dump(lenTrains, file_pi)

with open(analysis_folder + 'results/y_val_seq.pkl', 'wb') as file_pi:
    pickle.dump(y_val_store, file_pi)
with open(analysis_folder + 'results/y_val_pred_seq.pkl', 'wb') as file_pi:
    pickle.dump(y_val_pred_store, file_pi)
with open(analysis_folder + 'results/y_test_seq.pkl', 'wb') as file_pi:
    pickle.dump(y_test_store, file_pi)
with open(analysis_folder + 'results/y_test_pred_seq.pkl', 'wb') as file_pi:
    pickle.dump(y_test_pred_store, file_pi)

print("Done!")
