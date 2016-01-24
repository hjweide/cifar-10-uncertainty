#!/usr/bin/env python

import model
import network
import utils
import iter_funcs

import matplotlib.pyplot as plt
import numpy as np
import theano

from lasagne import layers
from lasagne.utils import floatX
from os.path import join
from time import time


def train(
        X_train, X_valid, y_train, y_valid, weights_file=None, init_file=None):

    # model parameters
    wd = 0.0005
    bs = 128
    base_lr = 0.01
    gamma = 0.0001
    p = 0.75
    mntm = 0.9
    fixed_bs = True
    mc_dropout = True
    #mc_dropout = False
    lr_update = lambda itr: base_lr * (1 + gamma * itr) ** (-p)
    snapshot_every = 5
    max_epochs = 100000

    print('building model...')
    l_out = model.build(bs, np.unique(y_train).shape[0])
    network.print_layers(l_out)

    # check if we need to load pre-trained weights
    if init_file is not None:
        print('initializing weights from %s...' % (init_file))
        network.init_weights(l_out, init_file)
    else:
        print('initializing weights randomly...')

    # do theano stuff
    print('creating shared variables...')
    lr_shared = theano.shared(floatX(base_lr))

    print('compiling theano functions...')
    train_iter = iter_funcs.create_iter_funcs_train(l_out, base_lr, mntm, wd)
    valid_iter = iter_funcs.create_iter_funcs_valid(
        l_out, bs, N=50, mc_dropout=mc_dropout)

    # prepare to start training
    best_epoch = -1
    best_train_losses_mean, best_valid_losses_mean = np.inf, np.inf
    print('starting training at %s' % (
        network.get_current_time()))
    epoch_train_losses, epoch_valid_losses = [], []
    gradient_updates = 0
    epochs = []

    # start training
    try:
        for epoch in range(1, max_epochs + 1):
            t_epoch_start = time()

            train_losses, train_accs = [], []
            # print run training for each batch
            for train_idx in network.get_batch_idx(
                    X_train.shape[0], bs, fixed=fixed_bs, shuffle=True):
                X_train_batch = X_train[train_idx]
                y_train_batch = y_train[train_idx]

                #print X_train_batch.shape, y_train_batch.shape
                train_loss, train_acc = train_iter(
                    X_train_batch, y_train_batch)
                #train_loss, train_acc = 0, 0

                # learning rate policy
                gradient_updates += 1
                lr = lr_update(gradient_updates)
                lr_shared.set_value(floatX(lr))

                train_losses.append(train_loss)
                train_accs.append(train_acc)

            # run validation for each batch
            valid_losses, valid_accs = [], []
            for valid_idx in network.get_batch_idx(
                    X_valid.shape[0], bs, fixed=fixed_bs, shuffle=False):

                X_valid_batch = X_valid[valid_idx]
                y_valid_batch = y_valid[valid_idx]

                #print X_valid_batch.shape, y_valid_batch.shape
                valid_loss, valid_acc = valid_iter(
                    X_valid_batch, y_valid_batch)
                #valid_loss, valid_acc = 0, 0

                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)

            # average over the batches
            train_losses_mean = np.mean(train_losses)
            train_accs_mean = np.mean(train_accs)
            valid_losses_mean = np.mean(valid_losses)
            valid_accs_mean = np.mean(valid_accs)

            epochs.append(epoch)
            epoch_train_losses.append(train_losses_mean)
            epoch_valid_losses.append(valid_losses_mean)

            # display useful info
            epoch_color = ('', '')
            if valid_losses_mean < best_valid_losses_mean:
                best_epoch = epoch
                best_train_losses_mean = train_losses_mean
                best_valid_losses_mean = valid_losses_mean
                best_weights = layers.get_all_param_values(l_out)
                epoch_color = ('\033[32m', '\033[0m')

            t_epoch_end = time()
            duration = t_epoch_end - t_epoch_start
            print('{}{:>4}{} | {:>10.6f} | {:>10.6f} | '
                  '{:>3.2f}% | {:>3.2f}% | '
                  '{:>1.8} | {:>4.2f}s | '.format(
                      epoch_color[0], epoch, epoch_color[1],
                      train_losses_mean, valid_losses_mean,
                      100 * train_accs_mean, 100 * valid_accs_mean,
                      lr, duration))

            if (epoch % snapshot_every) == 0:
                network.save_weights(best_weights, weights_file)

    except KeyboardInterrupt:
        print('caught ctrl-c... stopped training.')

    # display final results and save weights
    print('training finished at %s\n' % (
        network.get_current_time()))
    print('best local minimum for validation data at epoch %d' % (
        best_epoch))
    print('  train loss = %.6f' % (
        best_train_losses_mean))
    print('  valid loss = %.6f' % (
        best_valid_losses_mean))
    if best_weights is not None:
        print('saving best weights to %s' % (weights_file))
        network.save_weights(best_weights, weights_file)

    # plot the train/val loss over epochs
    print('plotting training/validation loss...')
    plt.plot(epochs, epoch_train_losses, 'b')
    plt.plot(epochs, epoch_valid_losses, 'g')
    plt.legend(('training', 'validation'))
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.xlim((1, epochs[-1]))
    train_val_log = join('logs', '%s.png' % network.get_current_time())
    plt.savefig(train_val_log, bbox_inches='tight')


def main():
    init_file = None
    weights_file = 'nets/weights.pickle'
    print('loading train/valid data...')
    X_train, X_valid, y_train, y_valid = utils.load_train_val(
        'data/cifar-10-batches-py')

    print('  X_train.shape = %r' % (X_train.shape,))
    print('  y_train.shape = %r' % (y_train.shape,))
    print('  X_valid.shape = %r' % (X_valid.shape,))
    print('  y_valid.shape = %r' % (y_valid.shape,))

    X_train, X_valid, X_mean = utils.normalize(X_train, X_valid)
    train(X_train, X_valid, y_train, y_valid, weights_file, init_file)


if __name__ == '__main__':
    main()
