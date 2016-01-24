#!/usr/bin/env python

import cPickle as pickle
import cv2  # NOQA
import numpy as np

from os.path import join


def load_cifar10_batch(path):
    with open(path, 'rb') as ifile:
        batch_dict = pickle.load(ifile)

    data = batch_dict['data']
    labels = batch_dict['labels']

    # convert to opencv BGR
    data = data.reshape(10000, 3, 32, 32)[:, ::-1, :, :]

    #cv2.imwrite('img.png', data[0].transpose(1, 2, 0))
    return data, np.array(labels, dtype=np.int32)


def load_train_val(dname):
    paths = [join(dname, 'data_batch_%d' % (d)) for d in range(1, 6)]
    X_train = np.empty((10000 * len(paths), 3, 32, 32), dtype=np.float32)
    y_train = np.empty(10000 * len(paths), dtype=np.int32)

    for i, path in enumerate(paths):
        data, labels = load_cifar10_batch(path)
        X_train[i * 10000:(i + 1) * 10000] = data / 255.
        y_train[i * 10000:(i + 1) * 10000] = labels

    X_valid = np.empty((10000, 3, 32, 32), dtype=np.float32)
    y_valid = np.empty(10000, dtype=np.int32)
    data, labels = load_cifar10_batch(join(dname, 'test_batch'))
    X_valid[:] = data / 255.
    y_valid[:] = labels

    return X_train, X_valid, y_train, y_valid


def normalize(X_train, X_valid):
    X_mean = np.mean(X_train, axis=0)

    X_train -= X_mean
    X_valid -= X_mean

    return X_train, X_valid, X_mean


def load_cifar100_class(path, label):
    with open(path, 'rb') as ifile:
        batch_dict = pickle.load(ifile)
        data = batch_dict['data']
        labels = np.array(batch_dict['fine_labels'], dtype=np.int32)

        data = data[labels == label]
        labels = labels[labels == label]

        data = data.reshape(-1, 3, 32, 32)[:, ::-1, :, :] / 255.

        return data.astype(np.float32), labels


if __name__ == '__main__':
    X_train, X_valid, y_train, y_valid = load_train_val(
        'data/cifar-10-batches-py')
    print X_train.dtype, X_valid.dtype
    print y_train.dtype, y_valid.dtype

    with open('data/cifar-10-batches-py/batches.meta', 'rb') as ifile:
        d = pickle.load(ifile)
        #for i in range(100):
        #    cv2.imwrite('test-images/%d_%d.png' % (i, y_train[i]), (255. * X_train[i]).transpose(1, 2, 0))
    with open('data/cifar-100-python/meta', 'rb') as ifile:
        d = pickle.load(ifile)
        print d

    data, labels = load_cifar100_class('data/cifar-100-python/train')
    print data.shape, labels.shape
    cv2.imwrite('img.png', (data[0] * 255.).transpose(1, 2, 0))
