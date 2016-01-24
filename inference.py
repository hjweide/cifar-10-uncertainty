#!/usr/bin/env python

import iter_funcs
import model
import network
import utils

import cv2

from os import makedirs
from os.path import join, isdir


def inference(X_test, X_mean, init_file, outdir):
    bs = 128
    fixed_bs = True

    print('building model...')
    l_out = model.build(bs, 10)

    print('initializing weights from %s...' % (init_file))
    network.init_weights(l_out, init_file)

    test_iter = iter_funcs.create_iter_funcs_test(l_out, bs, N=50)
    for test_idx in network.get_batch_idx(
            X_test.shape[0], bs, fixed=fixed_bs, shuffle=False):

        X_test_batch = X_test[test_idx]

        y_hat = test_iter(X_test_batch)

        # get the test images that were misclassified with low certainty
        for X, y in zip(X_test_batch, y_hat):
            if y.max() > 0.9:
                # undo the initial transformations: shift, scale transpose
                img = ((X + X_mean) * 255.).transpose(1, 2, 0)
                fname = '-'.join('%.5f' % p for p in y)
                cv2.imwrite(join(outdir, '%s.png' % fname), img)


def main():
    outdir = 'images-predicted'
    if not isdir(outdir):
        print('mkdir outdir')
        makedirs(outdir)

    init_file = 'nets/weights_check.pickle'
    print('loading train/valid data...')
    X_train, _, _, _ = utils.load_train_val(
        'data/cifar-10-batches-py')
    X_test, _ = utils.load_cifar100_class('data/cifar-100-python/train', 0)
    X_train, X_test, X_mean = utils.normalize(X_train, X_test)
    print('  X_train.shape = %r' % (X_train.shape,))
    print('  X_test.shape = %r' % (X_test.shape,))

    inference(X_test, X_mean, init_file, outdir)


if __name__ == '__main__':
    main()
