# coding: utf-8
import os
import pickle
import pdb
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *


pred_interval = 3
data_dir = "pred_more_data_alpha_900_10/pred_%d" % pred_interval
save_dir = "dnn_regression_plot_results_900_10/pred_%d" % pred_interval

if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)


def dnn_model(network, norm_train_set_x, train_set_y, norm_test_set_x, test_set_y):
    
    # train_set_y_reverse = 1 - train_set_y
    # train_set_y = np.transpose(np.vstack([train_set_y_reverse, train_set_y]))
    
    model = Sequential()
    
    for layer_idx, layer in enumerate(network):
        neuron_num, dropout = layer
        if layer_idx == 0:
            model.add(Dense(neuron_num, input_dim=norm_train_set_x.shape[1]))
        else:
            model.add(Dense(neuron_num))    
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(dropout))
    
    # output layer
    model.add(Dense(1))
    
    opt = Nadam(lr=0.002)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
    model.compile(optimizer=opt,
                  loss='mae')
    
    import pdb
    pdb.set_trace()

    history = model.fit(norm_train_set_x, train_set_y, epochs = 60,
                        batch_size = 128, verbose=1, validation_data=(norm_test_set_x, test_set_y),
                        shuffle=True, callbacks=[reduce_lr])
    
    return history


def tostr(network):
    network_str = "-->".join(["%d(%.2f)" % (layer[0], layer[1]) for layer in network])
    return network_str


data_files = os.listdir(data_dir)
start_idxes = [int(e.split('_')[0]) for e in data_files]
start_idxes = list(set(start_idxes))
start_idxes.sort()



network_ary = [
    [[64, 0.25], [64, 0.25], [32, 0.25]],
    [[64, 0.5], [64, 0.5], [32, 0.5]],
    [[32, 0.5], [32, 0.5], [16, 0.5]],
    [[16, 0.5], [16, 0.5], [16, 0.5]],
    [[16, 0.5], [16, 0.5]]
]

network_str_ary = [tostr(e) for e in network_ary]

for start_idx in start_idxes:
    
    # prepare training and validation set
    train_x_name =  os.path.join(data_dir, "%d_train_x" % start_idx)
    train_y_name =  os.path.join(data_dir, "%d_train_y" % start_idx)
    test_x_name =  os.path.join(data_dir, "%d_test_x" % start_idx)
    test_y_name =  os.path.join(data_dir, "%d_test_y" % start_idx)
    
    train_x_f = open(train_x_name, 'rb')
    train_y_f = open(train_y_name, 'rb')
    test_x_f = open(test_x_name, 'rb')
    test_y_f = open(test_y_name, 'rb')
    
    train_x = pickle.load(train_x_f)
    train_y = pickle.load(train_y_f)
    test_x = pickle.load(test_x_f)
    test_y = pickle.load(test_y_f)
    
    # train and validate with different networks
    history_ary = []
    history_data = []
    for network in network_ary:
        history = dnn_model(network, train_x, train_y, test_x, test_y)
        history_ary.append(history)
        history_data.append({
            'loss': np.copy(history.history['loss']),
            'val_loss': np.copy(history.history['val_loss'])
        })

    # save the history_ary
    f = open("%s/history_pred_%d_start_%d" % (save_dir, pred_interval, start_idx), 'wb')
    pickle.dump(history_data, f)
    
    # plot and save the results
    color = ['red', 'green', 'blue', 'orange', 'purple']
    
    for epoch_num in [60, 20]:
        plt.figure(figsize=(16, 12))
        legend = []
        for idx, history in enumerate(history_ary):
            plt.plot(history.history['loss'][:epoch_num], color=color[idx], ls='--')
            plt.plot(history.history['val_loss'][:epoch_num], color=color[idx])
            legend.append("train: " + network_str_ary[idx])
            legend.append("val:   " + network_str_ary[idx])
        plt.title('model loss (predict interval=%d, start index=%d)' % (pred_interval, start_idx))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(legend, loc='best')
        fig_path = "%s/loss_pred_%d_start_%d_epoch_%d.jpg" % (save_dir, pred_interval, start_idx, epoch_num)
        plt.savefig(fig_path)
