import cPickle as pickle
import numpy as np

from time import strftime
from lasagne import layers


def init_weights(l_out, init_file):
    print('loading weights from %s' % (init_file))
    with open(init_file, 'rb') as ifile:
        src_layers = pickle.load(ifile)
    dst_layers = layers.get_all_params(l_out)
    for i, (src_weights, dst_layer) in enumerate(
            zip(src_layers, dst_layers)):
        print('loading pretrained weights for %s' % (dst_layer.name))
        dst_layer.set_value(src_weights)


def save_weights(weights, weights_file):
    if weights is not None:
        with open(weights_file, 'wb') as ofile:
            pickle.dump(weights, ofile, protocol=pickle.HIGHEST_PROTOCOL)


def get_current_time():
    return strftime('%Y-%m-%d_%H:%M:%S')


def print_layers(l_out):
    all_layers = layers.get_all_layers(l_out)
    print('this network has %d learnable parameters' % (
        (layers.count_params(l_out))))
    for layer in all_layers:
        if hasattr(layer, 'W') and hasattr(layer, 'b'):
            num_params = np.prod(
                layer.W.get_value().shape) + np.prod(layer.b.get_value().shape)
            print('layer %s has output shape %r with %d parameters' % (
                (layer.name, layer.output_shape, num_params)))
        else:
            print('layer %s has output shape %r' % (
                (layer.name, layer.output_shape)))


def get_batch_idx(num_datapoints, bs, fixed=False, shuffle=False):
    num_batches = (num_datapoints + bs - 1) / bs

    batches = range(num_batches)
    # present batches to the network in random order
    if shuffle:
        batches = np.random.permutation(batches)

    for i in np.random.permutation(range(num_batches)):
        start, end = i * bs, (i + 1) * bs
        if end < num_datapoints:
            idx = range(start, end)
        else:
            idx = range(start, num_datapoints)
            # pad with samples from the beginning
            if fixed:
                idx += range(0, end - num_datapoints)
        yield idx
