#!/usr/bin/env python

"""
################################################################################
#                                                                              #
# psychedelic.jupyter_implicit                                                 #
#                                                                              #
################################################################################
#                                                                              #
# LICENCE INFORMATION                                                          #
#                                                                              #
# This program provides imports, environment printout, Keras callbacks and     #
# handy functions suitable for use in a Jupyter notebook.                      #
#                                                                              #
# copyright (C) 2019 William Breaden Madden                                    #
#                                                                              #
# This software is released under the terms of the GNU General Public License  #
# version 3 (GPLv3).                                                           #
#                                                                              #
# This program is free software: you can redistribute it and/or modify it      #
# under the terms of the GNU General Public License as published by the Free   #
# Software Foundation, either version 3 of the License, or (at your option)    #
# any later version.                                                           #
#                                                                              #
# This program is distributed in the hope that it will be useful, but WITHOUT  #
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        #
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     #
# more details.                                                                #
#                                                                              #
# For a copy of the GNU General Public License, see                            #
# <http://www.gnu.org/licenses/>.                                              #
#                                                                              #
################################################################################
"""

import datetime
import math
import os
import pickle
import random
import sqlite3
import sys
import uuid
import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.random.seed(1337)
import graphviz
from   IPython.display import SVG
import keras
from   keras import activations
from   keras import backend as K
from   keras.datasets import mnist
from   keras.layers import (
           concatenate,
           Concatenate,
           Conv1D,
           Conv2D,
           Dense,
           Dropout,
           Embedding,
           Flatten,
           Input,
           MaxPooling1D,
           MaxPooling2D)
from   keras.models import load_model, Model, Sequential
from   keras_tqdm import TQDMNotebookCallback
from   keras.utils import plot_model
from   keras.utils.vis_utils import model_to_dot
from   livelossplot.keras import PlotLossesCallback
import matplotlib
from   matplotlib import gridspec
import matplotlib.patches as mpatches
import matplotlib.pylab as plt
import matplotlib.ticker
from   matplotlib.ticker import NullFormatter, NullLocator, MultipleLocator
import mpl_toolkits.mplot3d
import pandas as pd
from   scipy import stats
from   scipy.stats import zscore
import seaborn as sns
import sklearn.datasets
from   sklearn.datasets import load_iris
import sklearn.decomposition
import sklearn.ensemble
import sklearn.manifold
from   sklearn.model_selection import train_test_split
import sklearn.preprocessing
from   sklearn.preprocessing import MinMaxScaler
import sklearn.tree
from   sklearn.metrics import (
           auc,
           confusion_matrix,
           precision_score,
           roc_curve)
try:
    import talos as ta
except:
    print('error importing Talos')
    pass
import tensorflow as tf
from   tensorflow.python.keras.callbacks import TensorBoard
from   tensorflow.python.client.device_lib import list_local_devices
from   tqdm import tqdm
from   tqdm import tqdm_notebook
import umap
try:
    from   vis.utils import utils
    from   vis.visualization import visualize_activation
    from   vis.visualization import visualize_saliency
except:
    print('error importing keras-viz')
    pass

################################################################################
#                                                                              #
# notebook style                                                               #
#                                                                              #
################################################################################

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows",    500)
sns.set_palette('husl')
sns.set(style='ticks')

################################################################################
#                                                                              #
# environment printout                                                         #
#                                                                              #
################################################################################

def environment_printout(printout_devices=True, preferred_device=None):
    '''
    Print out details about the computing environment. If a printout of devices
    is requested, then a printout of all devices is made unless a preferred
    device is specified, in which case a printout only of that device is made.
    '''
    print('Python version:', sys.version)
    print('Keras version:', keras.__version__)
    print('Matplotlib version:', matplotlib.__version__)
    print('NumPy version:', np.__version__)
    print('TensorFlow version:', tf.__version__)
    if printout_devices:
        if not preferred_device:
            print('\n' + str(list_local_devices()))
        if preferred_device:
            for device in list_local_devices():
                if preferred_device in device.physical_device_desc:
                    print('\n' + str(device))

################################################################################
#                                                                              #
# Keras callbacks                                                              #
#                                                                              #
################################################################################

class EpochProgressBar(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.total_epochs  = self.params['epochs']
        self.current_epoch = 0
        self.pbar = tqdm_notebook(total=self.total_epochs, desc='epochs')
    def on_epoch_end(self, batch, logs={}):
        self.current_epoch += 1
        #print(f'epoch {self.current_epoch} of epochs {self.total_epochs}')
        self.pbar.update(1);
epoch_progress_bar = EpochProgressBar()

class StopAtBeyondAccuracyValue(keras.callbacks.Callback):
    def __init__(self, val_accuracy=None):
        self.val_accuracy = val_accuracy
    def on_epoch_end(self, batch, logs={}):
        if logs.get('val_acc') >= self.val_accuracy:
             self.model.stop_training = True

#class TensorBoardCallback(object):
#    # callbacks = [TensorBoardCallback(model)]
#    def __init__(self, model):
#        layer_names = [layer['config']['name'] for layer in model.get_config()]
#        callback_TensorBoard = keras.callbacks.TensorBoard(
#            log_dir                ='/tmp/tensorboard',
#            histogram_freq         = True,  
#            write_graph            = True,
#            write_images           = True,
#            embeddings_layer_names = layer_names,
#        )
#        return callback_TensorBoard

def TensorBoardCallback():
    # rm /tmp/tensorboard/*
    # tensorboard --logdir /tmp/tensorboard
    # http://127.0.1.1:6006
    return TensorBoard(log_dir=f'/tmp/tensorboard/{datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")}')

class EarlyStoppingWithManualStop(keras.callbacks.Callback):
    """
    Stop training when a monitored quantity has stopped improving or if a
    specified filename is found to exist at the working directory, by default
    restoring the best weights from the epoch with the best monitored quantity.
    This callback is a modified version of Keras 2.2.4
    `keras.callbacks.EarlyStopping`.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 filename='safeword',
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStoppingWithManualStop, self).__init__()

        self.filename = filename
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)
        if os.path.exists(self.filename):
            print('Manual stop file existence detected')
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights:
                if self.verbose > 0:
                    print('Restoring model weights from the end of '
                          'the best epoch')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value

checkpoint_continuous = keras.callbacks.ModelCheckpoint(
    filepath       = 'best_model.{epoch:02d}-{val_loss:.2f}.h5',
    monitor        = 'val_loss',
    save_best_only = True
)

checkpoint_latest = keras.callbacks.ModelCheckpoint(
    filepath       = 'model_latest.h5',
    monitor        = 'val_loss',
    save_best_only = True
)

stop_early = keras.callbacks.EarlyStopping(
    monitor              = 'val_loss',
    min_delta            = 0.0001,
    patience             = 300,
    verbose              = 1,
    mode                 = 'auto',
    baseline             = None,
    restore_best_weights = True
)

stop_early_with_manual = EarlyStoppingWithManualStop(
    monitor              = 'val_loss',
    min_delta            = 0.0001,
    patience             = 300,
    verbose              = 1,
    mode                 = 'auto',
    baseline             = None,
    restore_best_weights = True
)

################################################################################
#                                                                              #
# handy functions                                                              #
#                                                                              #
################################################################################

def timestamp_string():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")

def timestamp():
    print(timestamp_string())

def summary_and_diagram(model):
    model.summary()
    # save model diagrams of varying detail to files
    ## less detailed
    uuid4_min = str(uuid.uuid4()).split('-')[0]
    filepath = timestamp_string() + "_" + uuid4_min + "_model.png"
    print(f"save to {filepath}")
    plot_model(model, to_file=filepath)
    ## more detailed
    uuid4_min = str(uuid.uuid4()).split('-')[0]
    filepath = timestamp_string() + "_" + uuid4_min + "_model.png"
    print(f"save to {filepath}")
    plot_model(model, to_file=filepath, show_shapes=True, show_layer_names=True)
    # display models as SVG in Jupyter
    return SVG(model_to_dot(model).create(prog='dot', format='svg'))
    #return SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))

def save_model(model):
    uuid4_min = str(uuid.uuid4()).split('-')[0]
    filepath = timestamp_string() + "_" + uuid4_min + "_model.ph5"
    model.save(filepath)
    print(f"save to {filepath}")
    return filepath

def model_evaluation(model, x_test, y_test, verbose=False):
    score = model.evaluate(x_test, y_test, verbose=verbose)
    print('max. test accuracy observed:', max(model.history.history['val_acc']))
    print('max. test accuracy history index:', model.history.history['val_acc'].index(max(model.history.history['val_acc'])))
    plt.plot(model.history.history['acc'],     label='train')
    plt.plot(model.history.history['val_acc'], label='validation')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show();

def model_training_plot(history):
    plt.plot(history.history['acc'],     marker='.', label='train')
    plt.plot(history.history['val_acc'], marker='.', label='validation')
    plt.title('accuracy')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show();

def plot_separation(
    s,
    b,
    bins       = 25,
    limits     = (-0.5, 1.5),
    legend_loc = 'upper left'
    ):
    bins = np.linspace(limits[0], limits[1], bins)
    f, ax = plt.subplots()
    _n, _b, _p = plt.hist(
        s,
        bins      = bins,
        bottom    = 0,
        linewidth = 1,
        histtype  = 'stepfilled',
        facecolor = 'none',
        edgecolor = 'red',
        label     = 'signal'
    )
    _n, _b, _p = plt.hist(
        b,
        bins      = bins,
        bottom    = 0,
        linewidth = 1,
        histtype  = 'stepfilled',
        facecolor = 'none',
        edgecolor = 'blue',
        label     = 'background'
    )
    if legend_loc == 'outside right':
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    else:
        plt.legend(loc=legend_loc);
    plt.xlabel('probability')
    plt.ylabel('cases')
    plt.show();

def class_prediction_error_plot(class_0, class_1):
    score_class_0 = len([i for i in class_0 if i<0.5])/len(class_0)
    score_class_1 = len([i for i in class_1 if i>=0.5])/len(class_1)

    cc0 =       score_class_0 * len(class_0)
    ci0 = (1 - score_class_0) * len(class_0)
    cc1 =       score_class_1 * len(class_1)
    ci1 = (1 - score_class_1) * len(class_1)

    fractions_correct   = (cc0, cc1)
    fractions_incorrect = (ci0, ci1)
    width = 0.9
    ind   = np.arange(2)
    p1    = plt.bar(ind, fractions_correct, width, color=['red', 'blue'])
    p2    = plt.bar(ind, fractions_incorrect, width, bottom=fractions_correct, color=['blue', 'red'])

    plt.xlabel('classification by model');
    plt.ylabel('class case relative abundances');
    plt.xticks(ind, ('signal', 'background'));
    plt.yticks([]);
    #plt.legend((p2[0], p1[0]), ('background', 'signal'), loc='center left', bbox_to_anchor=(1, 0.5));
    plt.legend((p2[0], p1[0]), ('background', 'signal'), loc='best');
    plt.show();

def rotate_ROC(points=None, theta=-0.785398, origin=(1, 0)):
    # e.g. rotate_ROC(points=np.array([list(i) for i in zip(fpr_en, tpr_en)]))
    ox, oy = origin
    points_transformed = []
    for point in points:
        px = point[0]
        py = point[1]
        qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
        qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)
        points_transformed.append([qx, qy])
    return np.array(points_transformed)

def scatterplot_two_classes_marginal_histograms(
    class_0_x           = None,
    class_0_y           = None,
    class_1_x           = None,
    class_1_y           = None,
    filepath            = None,
    show_plot           = True,
    return_plot         = False,
    printout_KS2        = True,
    color_0             = 'blue',
    color_1             = 'red',
    size_scatter_points = 0.03,
    binwidth            = 0.25,
    nbins_x             = 50,
    nbins_y             = 50):
    if printout_KS2:
        ksx    = stats.ks_2samp(class_0_x, class_1_x)
        ksy    = stats.ks_2samp(class_0_y, class_1_y)
        ksmean = (ksx[0]+ksy[0])/2
        print('KS2 x:', ksx)
        print('KS2 y:', ksy)
        print('mean of KS2 x and KS2 y:', ksmean)

    # axes definitions
    left    = 0.1
    width   = 0.65
    bottom  = 0.1
    height  = 0.65
    spacing = 0.005

    rect_scatter = [left                  , bottom                   , width, height]
    rect_histx   = [left                  , bottom + height + spacing, width, 0.2]
    rect_histy   = [left + width + spacing, bottom                   , 0.2  , height]

    fig = plt.figure(figsize=(8, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # scatter plot
    ax_scatter.scatter(class_0_x, class_0_y, color=color_0, s=size_scatter_points)
    ax_scatter.scatter(class_1_x, class_1_y, color=color_1, s=size_scatter_points)

    # determine good limits
    lim = np.ceil(np.abs([class_0_x, class_0_y]).max()/binwidth)*binwidth
    ax_scatter.set_xlim((-lim, lim))
    ax_scatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim+binwidth, binwidth)
    # top marginal histograms
    ax_histx.hist(class_1_x, bins=nbins_y, linewidth=1, histtype='stepfilled', facecolor='none', edgecolor=color_1)
    ax_histx.hist(class_0_x, bins=nbins_y, linewidth=1, histtype='stepfilled', facecolor='none', edgecolor=color_0)
    # side marginal histograms
    ax_histy.hist(class_1_y, bins=nbins_x, linewidth=1, histtype='stepfilled', facecolor='none', edgecolor=color_1, orientation='horizontal')
    ax_histy.hist(class_0_y, bins=nbins_x, linewidth=1, histtype='stepfilled', facecolor='none', edgecolor=color_0, orientation='horizontal')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    if filepath:
        plt.savefig(filepath)
    if show_plot:
        plt.show();
    if return_plot:
        return fig

def draw_pie(ax, ratios=[0.333, 0.333, 0.333], X=0, Y=0, size=10, colors=None, legend_names=None):
    ratios = [abs(ratio) for ratio in ratios]
    ratios = [float(ratio)/sum(ratios) for ratio in ratios]
    N = len(ratios)
    xy = []
    start = 0
    for ratio in ratios:
        x = [0]+np.cos(np.linspace(2*math.pi*start, 2*math.pi*(start+ratio), 100)).tolist()
        y = [0]+np.sin(np.linspace(2*math.pi*start, 2*math.pi*(start+ratio), 100)).tolist()
        xy1 = zip(x,y)
        xy.append(xy1)
        start += ratio
    if not colors:
        # https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors
        #colors = ['#ffe119', '#4363d8', '#000000', '#a9a9a9', '#ffffff', '#800000',
        #          '#000075', '#f58231', '#fabebe', '#e6beff', '#e6194B', '#bfef45',
        #          '#3cb44b', '#42d4f4', '#911eb4', '#f032e6', '#9a6324', '#808000',
        #          '#469990', '#ffd8b1', '#fffac8', '#aaffc3']
        colors = ['#e6194B', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#42d4f4',
                  '#4363d8', '#911eb4', '#f032e6', '#a9a9a9', '#800000', '#9a6324',
                  '#808000', '#469990', '#000075', '#000000', '#fabebe', '#ffd8b1',
                  '#fffac8', '#aaffc3', '#e6beff', '#ffffff', '#000000']
    for i, xyi in enumerate(xy):
        ax.scatter([X], [Y] , marker=(list(xyi), 0), s=size, facecolor=colors[i])
    #if legend_names:
    #    patches = []
    #    for color in colors[:len(feature_names)]:
    #        patches.append(mpatches.Rectangle((0, 0), 1, 1, fc=color))
    #    ax.legend(patches, feature_names, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
