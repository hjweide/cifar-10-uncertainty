#!/usr/bin/env python

import cPickle as pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir
from os.path import join, splitext


def make_image_grid(img_list, probs_list, name):
    max_per_row = 10
    if max_per_row >= len(img_list):
        num_cols = max_per_row
        num_rows = 1
    else:
        num_cols = max_per_row
        num_rows = int(np.ceil(len(img_list) / float(max_per_row)))

    fig = plt.figure(1)
    ax1 = plt.axes(frameon=False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    plt.title('apples classified as %s with p > 0.9' % name)
    # share_all=True ==> all grid[i] have same x and y dimensions
    grid = ImageGrid(
        fig, 111, nrows_ncols=(num_rows, num_cols),
        axes_pad=0.3, share_all=True)

    for i in range(num_rows * num_cols):
        if i < len(img_list):
            grid[i].imshow(img_list[i][:, :, ::-1])
            grid[i].set_title(label='%.3f' % float(probs_list[i]))
        grid[i].get_xaxis().set_visible(False)
        grid[i].get_yaxis().set_visible(False)
        grid[i].set_frame_on(False)

    plt.savefig('%s.png' % name, bbox_inches='tight')


def plot_images(dname):
    fnames = [fname for fname in listdir(dname)]
    class_dict = defaultdict(list)
    probs_dict = defaultdict(list)
    for fname in fnames:
        path = join(dname, fname)
        img = cv2.imread(path)
        probs = splitext(fname)[0].split('-')
        label = np.array(probs).argmax()
        probs_dict[label].append(probs[label])
        class_dict[label].append(img)

    # get the strings for the class labels
    with open('data/cifar-10-batches-py/batches.meta', 'rb') as ifile:
        meta_dict = pickle.load(ifile)
        label_names = meta_dict['label_names']

    # plot a grid for each class label
    for k in class_dict.keys():
        print('plotting grid of %d images for class %d (%s)...' % (
            len(class_dict[k]), k, label_names[k]))
        make_image_grid(class_dict[k], probs_dict[k], name=label_names[k])


if __name__ == '__main__':
    plot_images('images-predicted')
