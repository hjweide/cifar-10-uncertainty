from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax, identity
from lasagne.init import GlorotUniform


def build(bs, num_out):
    conv = {
        'filter_size': (5, 5), 'stride': (1, 1), 'pad': 2, 'num_filters': 192,
        'W': GlorotUniform(gain='relu'), 'nonlinearity': identity,  # for LeNet
    }
    pool = {'pool_size': (2, 2), 'stride': (2, 2)}
    drop = {'p': 0.5}

    l_in = InputLayer((None, 3, 32, 32), name='in')

    l_conv1 = Conv2DLayer(l_in, name='conv1', **conv)
    l_drop1 = DropoutLayer(l_conv1, name='drop1', **drop)
    l_pool1 = Pool2DLayer(l_drop1, name='pool1', **pool)

    l_conv2 = Conv2DLayer(l_pool1, name='conv2', **conv)
    l_drop2 = DropoutLayer(l_conv2, name='drop2', **drop)
    l_pool2 = Pool2DLayer(l_drop2, name='pool2', **pool)

    l_dense3 = DenseLayer(
        l_pool2, name='dense3', num_units=1000,
        W=GlorotUniform(gain='relu'), nonlinearity=rectify)
    l_drop3 = DropoutLayer(l_dense3, name='drop3', **drop)

    l_dense4 = DenseLayer(
        l_drop3, name='out', num_units=num_out,
        W=GlorotUniform(), nonlinearity=softmax)

    return l_dense4
